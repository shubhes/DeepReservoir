# deepreservoir/drl/environs.py
import pickle
import numpy as np
import pandas as pd
from gymnasium import Env
from collections import deque, defaultdict
from gymnasium.spaces import Box

from deepreservoir.drl.rewards import RewardContext
from deepreservoir.data.metadata import project_metadata
from deepreservoir.define_env.hydropower_model import navajo_power_generation_scalar
from deepreservoir.define_env.spring_peak_release.opportunity_index import (
    OIParams,
    precompute_oi_by_wy,
)

m = project_metadata()

# Unit conversions (daily timestep)
# 1 cfs sustained over 1 day -> acre-feet
#   1 acre-foot = 43,560 ft^3
#   1 day       = 86,400 s
#   1 cfs-day   = 86,400 ft^3 = 86,400 / 43,560 acre-feet
CFS_TO_AF_PER_DAY = 86400.0 / 43560.0
AF_PER_DAY_TO_CFS = 1.0 / CFS_TO_AF_PER_DAY

# -----------------------------------------------------------------------------
# Navajo Reservoir physical thresholds (authoritative elevations)
# -----------------------------------------------------------------------------
# NOTE: These are enforced in the environment physics (deadpool release blocking
# and automatic spill). Storage thresholds are derived from the E–S curve.
NAVAJO_DEADPOOL_ELEV_FT = 5775.0
NAVAJO_SPILL_ELEV_FT = 6085.0


class NavajoReservoirEnv(Env):
    """Navajo Reservoir environment (daily timestep).

    One agent, two continuous actions (Box[-1, 1], shape=(2,)):
      - action[0] -> release_sj_main_cfs
      - action[1] -> release_niip_cfs

    Important physical constraints implemented here:
      - Per-outlet capacity caps:
          release_sj_main_cfs <= max_release_sj_main_cfs (default 5000)
          release_niip_cfs    <= max_release_niip_cfs    (default 2500)
        If the agent requests more than the cap, we hard-cap (no proportional rescaling).

      - Deadpool constraint (elevation-based):
          If starting reservoir elevation is at/below NAVAJO_DEADPOOL_ELEV_FT (5775 ft),
          *no releases are physically possible*, so both outlet releases are forced to 0.

      - Spill constraint (elevation-based):
          If ending reservoir elevation would exceed NAVAJO_SPILL_ELEV_FT (6085 ft),
          the excess storage is automatically spilled (uncontrolled) to the San Juan mainstem.

      - Water-available constraint:
          Even above deadpool, releases cannot exceed the water available that day.
          If total requested release exceeds available water, both outlets are scaled down
          proportionally to satisfy mass balance (this preserves the requested split).

    Observation (fixed order, Colab-aligned):
      [ storage_norm, evap_norm, inflow_norm, doy ]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data_raw: pd.DataFrame,
        data_norm: pd.DataFrame,
        norm_stats: pd.DataFrame,
        reward_fn,
        episode_length: int | None = None,
        # Optional: restrict training to one or more allowed index segments.
        # Each tuple is (start_idx, end_idx) inclusive in the provided dataframes.
        allowed_segments: list[tuple[int, int]] | None = None,
        min_release_cfs: float = 0.0,
        max_release_sj_main_cfs: float = 5000.0,
        max_release_niip_cfs: float = 2500.0,
        # NOTE: legacy arg retained so callers don't break; it is no longer used
        # to *define* deadpool. Deadpool is defined by NAVAJO_DEADPOOL_ELEV_FT.
        deadpool_storage_af: float = 500_000.0,
        is_eval: bool = False,
    ):
        super().__init__()
        assert data_raw.index.equals(data_norm.index)

        self.data_raw = data_raw
        self.data_norm = data_norm
        self.norm_stats = norm_stats
        self.reward_fn = reward_fn
        self.is_eval = is_eval

        self.date_index = self.data_raw.index
        self.dates = self.date_index.to_list()
        self.n_steps = len(self.dates)

        # Episode length (in steps); default full series
        self.episode_length = (
            int(episode_length) if episode_length is not None else self.n_steps
        )

        # Optional allowed segments for training (used to avoid "smashing" across
        # excluded gaps while still allowing multiple disjoint periods).
        self.allowed_segments: list[tuple[int, int]] | None = None
        if allowed_segments is not None:
            segs: list[tuple[int, int]] = []
            for a, b in allowed_segments:
                i0 = int(a)
                i1 = int(b)
                if i1 < i0:
                    raise ValueError(f"allowed_segments contains inverted range: {(a, b)}")
                if i0 < 0 or i1 >= self.n_steps:
                    raise ValueError(
                        f"allowed_segments range {(a, b)} falls outside data bounds [0, {self.n_steps-1}]."
                    )
                segs.append((i0, i1))
            # Sort for determinism
            segs = sorted(segs, key=lambda x: x[0])
            self.allowed_segments = segs

        # Release limits (per-outlet) in cfs
        self.min_release_cfs = float(min_release_cfs)
        self.max_release_sj_main_cfs = float(max_release_sj_main_cfs)
        self.max_release_niip_cfs = float(max_release_niip_cfs)

        # Deadpool/spill are defined by *elevation*; storage thresholds are derived
        # from the elevation-storage relationship for convenience/clamping.
        self.deadpool_elev_ft = float(NAVAJO_DEADPOOL_ELEV_FT)
        self.spill_elev_ft = float(NAVAJO_SPILL_ELEV_FT)

        # Keep the provided value only as a legacy/debug field (not authoritative).
        self._deadpool_storage_af_legacy = float(deadpool_storage_af)

        # ACTION SPACE: 2 releases in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # For optional lagged discharge proxy
        self.sj_at_farmington_history = deque(maxlen=3)

        # ---- Observation schema checks ----
        # We REQUIRE raw columns for mass balance:
        required_raw = ["storage_af", "inflow_cfs", "evap_af"]
        for c in required_raw:
            if c not in self.data_raw.columns:
                raise KeyError(f"data_raw is missing required column '{c}'")

        # We REQUIRE normalized columns for observations (inflow/evap):
        required_norm = ["inflow_cfs", "evap_af"]
        for c in required_norm:
            if c not in self.data_norm.columns:
                raise KeyError(
                    f"data_norm is missing required column '{c}'. "
                    f"To include evaporation in observation, ensure preprocessing outputs 'evap_af' in data_norm."
                )

        # Fixed schema across train/eval
        self.obs_cols = ["storage_af", "evap_af", "inflow_cfs", "doy"]
        self.obs_dim = len(self.obs_cols)

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Frequently accessed scalars/arrays for the step hot path.
        storage_mean = float(self.norm_stats.loc["storage_af", "mean"])
        storage_std = float(self.norm_stats.loc["storage_af", "std"])
        self._storage_mean = storage_mean
        self._storage_std = storage_std if storage_std != 0.0 else 1.0

        self._raw_storage_af = self.data_raw["storage_af"].to_numpy(dtype=np.float64, copy=True)
        self._raw_inflow_cfs = self.data_raw["inflow_cfs"].to_numpy(dtype=np.float64, copy=True)
        self._raw_evap_af = self.data_raw["evap_af"].to_numpy(dtype=np.float64, copy=True)
        if "animas_farmington_q_cfs" in self.data_raw.columns:
            self._raw_animas_farmington_q_cfs = self.data_raw[
                "animas_farmington_q_cfs"
            ].to_numpy(dtype=np.float64, copy=True)
        else:
            self._raw_animas_farmington_q_cfs = np.zeros(self.n_steps, dtype=np.float64)

        self._norm_inflow_cfs = self.data_norm["inflow_cfs"].to_numpy(dtype=np.float32, copy=True)
        self._norm_evap_af = self.data_norm["evap_af"].to_numpy(dtype=np.float32, copy=True)
        self._doy_frac = (self.date_index.dayofyear.to_numpy(dtype=np.float32) / 366.0).astype(np.float32, copy=False)

        # Load elevation model
        with open(m.path("elev_area_storage_pickle"), "rb") as f:
            elev_models = pickle.load(f)
        # Elevation-area-storage interpolators (2019 table).
        # We only *require* capacity→elevation, but we also keep elevation→capacity when available.
        self.capacity_to_elev = elev_models["capacity_to_elevation"]
        self.elev_to_capacity = elev_models.get("elevation_to_capacity", None)

        # Ensure we have elevation -> capacity for spill clamping.
        if self.elev_to_capacity is None:
            # Build a numerical inverse from capacity->elevation as a last resort.
            # This is robust and keeps the environment runnable even if the pickle
            # only contains capacity->elevation.
            caps = np.linspace(0.0, 2_000_000.0, 20001)
            elevs = np.asarray(self.capacity_to_elev(caps), dtype=float)
            order = np.argsort(elevs)
            elevs_sorted = elevs[order]
            caps_sorted = caps[order]

            def _elev_to_capacity(elev_ft):
                elev_arr = np.asarray(elev_ft, dtype=float)
                return np.interp(elev_arr, elevs_sorted, caps_sorted)

            self.elev_to_capacity = _elev_to_capacity

        # Fast scalar lookup tables for the env hot path. When the loaded models are
        # scipy interp1d objects, we can reuse their native breakpoint arrays exactly
        # and evaluate them with np.interp, which is much cheaper for scalar calls.
        self._cap_to_elev_x: np.ndarray | None = None
        self._cap_to_elev_y: np.ndarray | None = None
        self._elev_to_cap_x: np.ndarray | None = None
        self._elev_to_cap_y: np.ndarray | None = None

        if hasattr(self.capacity_to_elev, "x") and hasattr(self.capacity_to_elev, "y"):
            self._cap_to_elev_x = np.asarray(self.capacity_to_elev.x, dtype=np.float64)
            self._cap_to_elev_y = np.asarray(self.capacity_to_elev.y, dtype=np.float64)
        else:
            _caps = np.linspace(0.0, 2_000_000.0, 20001, dtype=np.float64)
            _elevs = np.asarray(self.capacity_to_elev(_caps), dtype=np.float64)
            self._cap_to_elev_x = _caps
            self._cap_to_elev_y = _elevs

        if hasattr(self.elev_to_capacity, "x") and hasattr(self.elev_to_capacity, "y"):
            self._elev_to_cap_x = np.asarray(self.elev_to_capacity.x, dtype=np.float64)
            self._elev_to_cap_y = np.asarray(self.elev_to_capacity.y, dtype=np.float64)

        # Storage thresholds derived from the E–S curve (useful for clamping/debug).
        self.deadpool_storage_af = float(self._elev_to_capacity_scalar(self.deadpool_elev_ft))
        self.max_storage_af = float(self._elev_to_capacity_scalar(self.spill_elev_ft))

        # --- SPR Opportunity Index precompute ---
        pm = project_metadata()
        params_path = pm.path("params.spr_oi_params_json")
        self.spring_oi_params: OIParams = OIParams.load(params_path)

        _df_wy = precompute_oi_by_wy(self.data_raw, self.spring_oi_params)
        self._spring_oi_by_wy = _df_wy["oi"]
        self._spring_go_by_wy = _df_wy["go"]

        _wy_by_day = pd.Series(
            (self.data_raw.index.year + (self.data_raw.index.month >= 10)).astype(int),
            index=self.data_raw.index,
            name="wy",
        )
        _oi_map = self._spring_oi_by_wy.to_dict()
        _go_map = self._spring_go_by_wy.to_dict()

        self.spring_oi_daily = _wy_by_day.map(_oi_map).astype(float)
        _go_daily = _wy_by_day.map(_go_map).astype("boolean").fillna(False)
        self.spring_go_daily = _go_daily.astype(bool)
        self._spring_oi_daily = self.spring_oi_daily.to_numpy(dtype=np.float64, copy=True)
        self._spring_go_daily = self.spring_go_daily.to_numpy(dtype=bool, copy=True)
        self._water_year = (
            self.date_index.year + (self.date_index.month >= 10)
        ).to_numpy(dtype=np.int32, copy=True)

        # Internal state
        self.t = 0
        self.start_idx = 0
        self._segment_start_idx = 0
        self._segment_end_idx = self.n_steps - 1
        self._episode_end_idx = self.n_steps - 1
        self.storage_af: float | None = None
        self.episode_step_count = 0
        self._last_obs: np.ndarray | None = None

        self.last_reward_breakdown: dict[str, float] | None = None

        # Episode bookkeeping
        self._episode_reward_sums: dict[str, float] = defaultdict(float)
        self._episode_total_reward: float = 0.0

        # Pass 2: compute only the per-step features needed by the active
        # reward configuration during training. Evaluation still computes the
        # full diagnostic payload for plotting and metrics export.
        self._init_step_feature_flags()

    # ---------------- Helpers ----------------

    def _init_step_feature_flags(self) -> None:
        comps = getattr(self.reward_fn, "components", None)
        active_pairs: set[tuple[str, str]] = set()
        if comps is not None:
            active_pairs = {
                (str(comp.objective), str(comp.variant))
                for comp in comps
            }

        active_objectives = {obj for obj, _ in active_pairs}
        spring_variants = {
            variant for obj, variant in active_pairs if obj == "esa_spring_peak_release"
        }
        dam_variants = {variant for obj, variant in active_pairs if obj == "dam_safety"}

        spring_farmington_variants = {"farmington_10k", "farmington_10k_shaped"}
        spring_bluff_variants = {"bluff_10k"}
        need_spring_farmington = bool(spring_variants & spring_farmington_variants)
        need_spring_bluff = bool(spring_variants & spring_bluff_variants)

        # Eval should keep the full payload intact for downstream analysis.
        compute_full = bool(self.is_eval or comps is None)

        self._compute_full_step_info = compute_full
        self._need_animas_farmington = bool(
            compute_full
            or ("esa_min_flow" in active_objectives)
            or ("flooding" in active_objectives)
            or need_spring_farmington
            or need_spring_bluff
        )
        self._need_sj_at_farmington = bool(
            compute_full
            or ("flooding" in active_objectives)
            or need_spring_farmington
            or need_spring_bluff
        )
        self._need_sj_at_farmington_lag2 = bool(
            compute_full or ("flooding" in active_objectives) or need_spring_bluff
        )
        self._need_end_elev = bool(
            compute_full
            or ("hydropower" in active_objectives)
            or (("dam_safety", "baseline") in active_pairs)
        )
        self._need_hydropower = bool(compute_full or ("hydropower" in active_objectives))
        self._need_storage_bounds = bool(
            compute_full or bool(dam_variants & {"storage_band_env", "storage_fraction"})
        )
        self._need_spring_oi = bool(
            compute_full or bool(spring_variants & {"oi", "farmington_10k_shaped"})
        )
        self._need_spring_meta = bool(compute_full)

        self.step_feature_flags = {
            "compute_full_step_info": bool(self._compute_full_step_info),
            "need_animas_farmington": bool(self._need_animas_farmington),
            "need_sj_at_farmington": bool(self._need_sj_at_farmington),
            "need_sj_at_farmington_lag2": bool(self._need_sj_at_farmington_lag2),
            "need_end_elev": bool(self._need_end_elev),
            "need_hydropower": bool(self._need_hydropower),
            "need_storage_bounds": bool(self._need_storage_bounds),
            "need_spring_oi": bool(self._need_spring_oi),
            "need_spring_meta": bool(self._need_spring_meta),
            "active_reward_components": sorted(active_pairs),
        }

    def _current_global_idx(self) -> int:
        return self.start_idx + self.t

    def _current_date(self) -> pd.Timestamp:
        return self.dates[self._current_global_idx()]

    def _norm_storage(self, storage_af: float) -> float:
        return (float(storage_af) - self._storage_mean) / self._storage_std

    def _capacity_to_elev_scalar(self, storage_af: float) -> float:
        x = self._cap_to_elev_x
        y = self._cap_to_elev_y
        if x is not None and y is not None:
            return float(np.interp(float(storage_af), x, y))
        return float(self.capacity_to_elev(float(storage_af)))

    def _elev_to_capacity_scalar(self, elev_ft: float) -> float:
        x = self._elev_to_cap_x
        y = self._elev_to_cap_y
        if x is not None and y is not None:
            return float(np.interp(float(elev_ft), x, y))
        return float(self.elev_to_capacity(float(elev_ft)))

    def _build_obs(self) -> np.ndarray:
        idx = self._current_global_idx()
        return np.array(
            [
                self._norm_storage(float(self.storage_af)),
                self._norm_evap_af[idx],
                self._norm_inflow_cfs[idx],
                self._doy_frac[idx],
            ],
            dtype=np.float32,
        )

    # ---------------- Gym API ----------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.is_eval:
            self.start_idx = 0
            self._segment_start_idx = 0
            self._segment_end_idx = self.n_steps - 1
            self._episode_end_idx = self.n_steps - 1
            self.max_steps = self.n_steps
        else:
            # Choose an episode start inside allowed segments (if provided).
            if self.allowed_segments is not None:
                segs = self.allowed_segments
                # Weight segments by available start positions.
                weights = []
                for s0, s1 in segs:
                    seg_len = int(s1 - s0 + 1)
                    # Number of feasible episode starts in this segment.
                    # Prefer starts that can accommodate a full-length episode when possible.
                    if seg_len > self.episode_length:
                        n_starts = seg_len - self.episode_length + 1
                    else:
                        # Require at least 2 days so step() can advance once.
                        n_starts = seg_len - 1
                    weights.append(max(int(n_starts), 0))

                total_w = int(sum(weights))
                if total_w <= 0:
                    raise ValueError(
                        "allowed_segments do not contain any segment with length >= 2; cannot sample episode starts."
                    )

                r = int(self.np_random.integers(0, total_w))
                acc = 0
                chosen = 0
                for i, w in enumerate(weights):
                    acc += int(w)
                    if r < acc:
                        chosen = i
                        break

                seg_start, seg_end = segs[chosen]
                self._segment_start_idx = int(seg_start)
                self._segment_end_idx = int(seg_end)

                seg_len = int(seg_end - seg_start + 1)
                # Prefer full-length episodes when possible.
                if seg_len > self.episode_length:
                    start_min = int(seg_start)
                    start_max = int(seg_end - self.episode_length)
                else:
                    start_min = int(seg_start)
                    start_max = int(seg_end - 1)

                self.start_idx = int(self.np_random.integers(start_min, start_max + 1))
                self._episode_end_idx = int(min(self.start_idx + self.episode_length - 1, seg_end))
                self.max_steps = int(self._episode_end_idx - self.start_idx + 1)
            else:
                # Original behavior: sample a contiguous episode within the provided dataframe.
                max_start = max(self.n_steps - self.episode_length, 0)
                self.start_idx = int(self.np_random.integers(0, max_start + 1))
                self._segment_start_idx = 0
                self._segment_end_idx = self.n_steps - 1
                self._episode_end_idx = int(min(self.start_idx + self.episode_length - 1, self.n_steps - 1))
                self.max_steps = int(self._episode_end_idx - self.start_idx + 1)

        self.t = 0
        self.episode_step_count = 0

        # init storage at episode start
        self.storage_af = float(self._raw_storage_af[self.start_idx])
        # Enforce physical spill level at reset as well (historical series should not exceed this,
        # but small inconsistencies/rounding can otherwise start an episode above the spill level).
        self.storage_af = float(min(self.storage_af, self.max_storage_af))


        self.last_reward_breakdown = None
        self.sj_at_farmington_history.clear()

        self._episode_reward_sums = defaultdict(float)
        self._episode_total_reward = 0.0

        obs = self._build_obs()
        self._last_obs = obs
        return obs, {}

    def step(self, action):
        # ---- action shaping ----
        action = np.asarray(action, dtype=np.float32).squeeze()
        if action.ndim != 1 or action.shape[0] != 2:
            raise ValueError(f"Expected action shape (2,), got {action.shape}")
        action = np.clip(action, -1.0, 1.0)

        obs = self._last_obs
        if obs is None:
            obs = self._build_obs()
            self._last_obs = obs
        global_idx = self._current_global_idx()
        date = self.dates[global_idx]

        # Map actions to nonnegative per-outlet releases (cfs)
        frac_sj = (action[0] + 1.0) / 2.0
        frac_niip = (action[1] + 1.0) / 2.0

        requested_release_sj_main_cfs = self.min_release_cfs + frac_sj * (
            self.max_release_sj_main_cfs - self.min_release_cfs
        )
        requested_release_niip_cfs = self.min_release_cfs + frac_niip * (
            self.max_release_niip_cfs - self.min_release_cfs
        )

        # --- Deadpool constraint: no releases below deadpool ---
        # We implement this in *elevation* space (outlet intake constraint), using the
        # starting elevation of the step. This is conservative: if inflow during the day
        # would raise the reservoir above deadpool, releases become possible on the next step.
        start_elev_ft = self._capacity_to_elev_scalar(float(self.storage_af))
        deadpool_block = start_elev_ft <= float(self.deadpool_elev_ft)

        if deadpool_block:
            release_sj_main_cfs = 0.0
            release_niip_cfs = 0.0
            cap_penalty = 0.0
        else:
            # --- Per-outlet hard caps ---
            release_sj_main_cfs = float(
                min(requested_release_sj_main_cfs, self.max_release_sj_main_cfs)
            )
            release_niip_cfs = float(min(requested_release_niip_cfs, self.max_release_niip_cfs))

            # Aggregate cap-penalty = fraction of requested flow clipped by outlet caps
            clipped = (requested_release_sj_main_cfs - release_sj_main_cfs) + (
                requested_release_niip_cfs - release_niip_cfs
            )
            cap_penalty = float(max(clipped, 0.0)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        total_cfs = float(release_sj_main_cfs + release_niip_cfs)

        # --- Physical feasibility (water available) ---
        # IMPORTANT: The reservoir state is volume (acre-feet). We enforce the
        # feasibility constraint in *volume space* to avoid extra AF<->CFS
        # round-tripping and to make the mass balance more explicit.
        inflow_cfs = float(self._raw_inflow_cfs[global_idx])
        inflow_af = inflow_cfs * CFS_TO_AF_PER_DAY
        evap_af = float(self._raw_evap_af[global_idx])

        available_af = max(float(self.storage_af) + inflow_af - evap_af, 0.0)
        requested_total_af = float(total_cfs) * CFS_TO_AF_PER_DAY

        phys_penalty = 0.0
        if requested_total_af > available_af:
            # Scale controlled releases proportionally to satisfy mass balance.
            # (Ratio is identical in cfs or af/day because the timestep is fixed.)
            pre_phys_total = float(total_cfs)
            scale = float(available_af) / (requested_total_af + 1e-9)
            release_sj_main_cfs *= scale
            release_niip_cfs *= scale
            total_cfs = float(release_sj_main_cfs + release_niip_cfs)

            phys_penalty = (pre_phys_total - float(total_cfs)) / (
                (self.max_release_sj_main_cfs + self.max_release_niip_cfs) + 1e-9
            )

        # --- Mass balance update ---
        # --- Mass balance update (controlled releases) ---
        controlled_total_cfs = float(total_cfs)
        controlled_total_af = controlled_total_cfs * CFS_TO_AF_PER_DAY
        new_storage_af = float(self.storage_af) + inflow_af - evap_af - controlled_total_af
        new_storage_af = max(new_storage_af, 0.0)

        # --- Automatic spill (physical): any water above the spill level is released ---
        # We model this as an uncontrolled spill that goes to the San Juan mainstem (not NIIP).
        spill_af = 0.0
        spill_cfs = 0.0
        if new_storage_af > float(self.max_storage_af):
            spill_af = float(new_storage_af - float(self.max_storage_af))
            spill_cfs = float(spill_af * AF_PER_DAY_TO_CFS)
            new_storage_af = float(self.max_storage_af)

        # Totals including spill (actual outflow at the dam)
        sj_main_flow_cfs = float(release_sj_main_cfs + spill_cfs)
        total_cfs = float(sj_main_flow_cfs + release_niip_cfs)
        total_af = float(controlled_total_af + spill_af)

        # Pass 2: during training, only compute the expensive/optional fields
        # required by the active reward configuration. Eval keeps the full
        # payload for plotting/metrics compatibility.
        animas_cfs = 0.0
        if self._need_animas_farmington:
            animas_cfs = float(self._raw_animas_farmington_q_cfs[global_idx])

        sj_at_farm_cfs = None
        sj_at_farm_lag2_cfs = None
        if self._need_sj_at_farmington:
            # IMPORTANT: mainstem outlet contributes to Farmington; NIIP does not.
            sj_at_farm_cfs = animas_cfs + float(sj_main_flow_cfs)
            if self._need_sj_at_farmington_lag2:
                self.sj_at_farmington_history.append(float(sj_at_farm_cfs))
                sj_at_farm_lag2_cfs = (
                    self.sj_at_farmington_history[-3]
                    if len(self.sj_at_farmington_history) >= 3
                    else None
                )

        new_elev_ft = None
        if self._need_end_elev or self._need_hydropower:
            new_elev_ft = self._capacity_to_elev_scalar(new_storage_af)

        hydropower_mwh = None
        if self._need_hydropower:
            # Hydropower uses *controlled* (turbine) San Juan mainstem release.
            # Spill is uncontrolled and should not contribute to generation.
            hydropower_mwh = navajo_power_generation_scalar(
                cfs_value=float(release_sj_main_cfs),
                elevation_ft=float(new_elev_ft),
            )

        wy = None
        oi_val = None
        go_val = None
        if self._need_spring_meta:
            wy = int(self._water_year[global_idx])
            go_val = bool(self._spring_go_daily[global_idx])
        if self._need_spring_oi:
            oi_val = float(self._spring_oi_daily[global_idx])

        info = {
            "date": date,
            "storage_af": float(new_storage_af),
            "prev_storage_af": float(self.storage_af),
            "release_sj_main_cfs": float(release_sj_main_cfs),
            "release_niip_cfs": float(release_niip_cfs),
        }

        if self._need_storage_bounds:
            info["deadpool_storage_af"] = float(self.deadpool_storage_af)
            info["max_storage_af"] = float(self.max_storage_af)

        if self._need_end_elev:
            info["elev_ft"] = float(new_elev_ft)

        if self._need_animas_farmington:
            info["animas_farmington_q_cfs"] = float(animas_cfs)

        if self._need_sj_at_farmington and sj_at_farm_cfs is not None:
            info["sj_at_farmington_cfs"] = float(sj_at_farm_cfs)

        if self._need_sj_at_farmington_lag2:
            info["sj_at_farmington_lag2_cfs"] = (
                None if sj_at_farm_lag2_cfs is None else float(sj_at_farm_lag2_cfs)
            )

        if self._need_hydropower and hydropower_mwh is not None:
            info["hydropower_mwh"] = float(hydropower_mwh)

        if self._need_spring_oi and oi_val is not None:
            info["spring_oi"] = float(oi_val)

        if self._need_spring_meta:
            info["spring_wy"] = wy
            info["spring_go"] = go_val

        if self._compute_full_step_info:
            raw_forcings = {"animas_farmington_q_cfs": float(animas_cfs)}
            info.update(
                {
                    "prev_elev_ft": float(start_elev_ft),
                    "inflow_cfs": float(inflow_cfs),
                    "inflow_af": float(inflow_af),
                    "evap_af": float(evap_af),
                    "available_af": float(available_af),
                    "requested_total_release_af": float(requested_total_af),
                    "total_release_cfs": float(total_cfs),
                    "total_release_af": float(total_af),
                    "spill_cfs": float(spill_cfs),
                    "spill_af": float(spill_af),
                    "sj_main_flow_cfs": float(sj_main_flow_cfs),
                    "total_controlled_release_cfs": float(controlled_total_cfs),
                    "total_controlled_release_af": float(controlled_total_af),
                    "requested_release_sj_main_cfs": float(requested_release_sj_main_cfs),
                    "requested_release_niip_cfs": float(requested_release_niip_cfs),
                    "max_release_sj_main_cfs": float(self.max_release_sj_main_cfs),
                    "max_release_niip_cfs": float(self.max_release_niip_cfs),
                    "deadpool_storage_af": float(self.deadpool_storage_af),
                    "deadpool_elev_ft": float(self.deadpool_elev_ft),
                    "spill_elev_ft": float(self.spill_elev_ft),
                    "deadpool_block": bool(deadpool_block),
                    "elev_ft": float(new_elev_ft),
                    "sj_at_farmington_cfs": float(sj_at_farm_cfs) if sj_at_farm_cfs is not None else float(sj_main_flow_cfs),
                    "sj_at_farmington_lag2_cfs": None
                    if sj_at_farm_lag2_cfs is None
                    else float(sj_at_farm_lag2_cfs),
                    "animas_farmington_q_cfs": float(animas_cfs),
                    "max_storage_af": float(self.max_storage_af),
                    "raw_forcings": raw_forcings,
                    "hydropower_mwh": float(hydropower_mwh) if hydropower_mwh is not None else 0.0,
                    "release_cap_penalty": float(cap_penalty),
                    "release_phys_penalty": float(phys_penalty),
                    "spring_wy": wy,
                    "spring_oi": float(oi_val) if oi_val is not None else np.nan,
                    "spring_go": go_val,
                }
            )

        # Advance
        self.storage_af = float(new_storage_af)
        self.t += 1
        self.episode_step_count += 1

        global_idx_next = self._current_global_idx()
        done = bool(global_idx_next > int(self._episode_end_idx) or global_idx_next >= self.n_steps)

        # Gymnasium semantics: we treat end-of-data as "terminated"; time limits and
        # segment boundaries as "truncated".
        terminated = bool(global_idx_next >= self.n_steps)
        truncated = bool(done and not terminated)

        if not done:
            next_obs = self._build_obs()
            self._last_obs = next_obs
        else:
            next_obs = np.zeros_like(obs, dtype=np.float32)
            self._last_obs = None

        # Reward
        ctx = RewardContext(
            t=global_idx,
            date=date,
            obs=obs,
            action=action,
            next_obs=next_obs,
            info=info,
        )
        total_reward, breakdown = self.reward_fn(ctx)
        self.last_reward_breakdown = breakdown

        # Episode bookkeeping
        self._episode_total_reward += float(total_reward)
        for k, v in breakdown.items():
            self._episode_reward_sums[k] += float(v)

        info["reward_components_step"] = breakdown
        info["reward_components_episode"] = dict(self._episode_reward_sums)
        info["episode_total_reward"] = float(self._episode_total_reward)
        info["reward_components"] = breakdown

        return next_obs, float(total_reward), terminated, truncated, info