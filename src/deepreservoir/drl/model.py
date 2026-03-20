"""
High-level training and evaluation utilities for DeepReservoir DRL.

- loads data via NavajoData
- splits into train / test
- builds reward (from registry)
- builds Gymnasium envs
- trains & evaluates an SB3 agent
- provides helpers to roll out on the test period and compute metrics

Colab equivalence goals:
- Training uses random-window episodes (episode_length_train=3600 by default)
- Testing rolls out the full test period deterministically
- Multi-action PPO: one agent, two continuous actions (handled inside environs.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import json
from datetime import datetime

import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from deepreservoir.data import loader
from deepreservoir.drl import helpers
from deepreservoir.drl import rewards as drl_rewards
from deepreservoir.drl import plotting as drl_plotting
from deepreservoir.drl import metrics as drl_metrics
from deepreservoir.drl.environs import NavajoReservoirEnv
from deepreservoir.define_env.hydropower_model import navajo_power_generation_scalar


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def load_all_model_data() -> Dict[str, pd.DataFrame]:
    """Load the full model dataframe (raw + normalized) and global norm stats."""
    nav_data = loader.NavajoData()
    alldata = nav_data.load_all(include_cont_streamflow=False, model_data=True)

    return {
        "raw": alldata["model_data"],
        "norm": alldata["model_data_norm"],
        "norm_stats": alldata["model_norm_stats"],
    }


def _default_train_eval_split(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    *,
    n_years_test: int,
) -> Dict[str, pd.DataFrame]:
    """Backstop behavior: last N water years = eval/test; everything before = train."""
    train_raw, eval_raw = helpers.split_train_test_by_water_year(data_raw, n_years_test=n_years_test)
    train_norm = data_norm.loc[train_raw.index]
    eval_norm = data_norm.loc[eval_raw.index]
    return {
        "train_raw": train_raw,
        "train_norm": train_norm,
        "eval_raw": eval_raw,
        "eval_norm": eval_norm,
    }


def _slice_train_val_windows(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    *,
    use_full_record: bool,
    n_years_train: int | None,
    train_start: Optional[str],
    train_end: Optional[str],
    exclude_start: Optional[str],
    exclude_end: Optional[str],
    val_start: Optional[str],
    val_end: Optional[str],
) -> tuple[Dict[str, pd.DataFrame | None], dict]:
    """Slice raw+norm dataframes into training + optional validation windows.

    Training window is chosen by exactly one of:
      - use_full_record=True
      - n_years_train (last N water years)
      - train_start/train_end tokens

    A single exclusion "hole" (exclude_start/exclude_end) may be removed from
    training. The base training dataframe still spans the full training window,
    but the environment only samples episode starts from allowed segments and
    truncates episodes at segment boundaries.
    """

    if not isinstance(data_raw.index, pd.DatetimeIndex):
        raise TypeError("Expected DatetimeIndex on model_data.")

    meta: dict = {"train": {}, "exclude": {}, "val": {}}

    # -----------------------------------------------------------------
    # Resolve base training window
    # -----------------------------------------------------------------
    if use_full_record:
        train_raw, w = helpers.slice_by_window(data_raw, start_token=None, end_token=None, label="train_window")
        train_norm = data_norm.loc[train_raw.index]
        meta["train"] = {
            "use_full_record": True,
            "n_years_train": None,
            "train_start": None,
            "train_end": None,
            "resolved": {
                "start": str(w.start.date()),
                "end": str(w.end.date()),
                "start_token": w.start_token,
                "end_token": w.end_token,
                "n_days": int(w.n_days),
            },
        }
    elif n_years_train is not None:
        n = int(n_years_train)
        if n <= 0:
            raise ValueError(f"n_years_train must be positive, got {n}")

        rng = helpers.available_date_range(data_raw)
        last_wy = int(rng["max_water_year"])
        first_wy = int(last_wy - n + 1)
        start_date, _ = helpers.water_year_start_end(first_wy)
        _, end_date = helpers.water_year_start_end(last_wy)

        # Clamp to available range (data join/clipping can truncate endpoints)
        start_date = max(start_date, rng["start"])
        end_date = min(end_date, rng["end"])

        train_raw = data_raw.loc[start_date:end_date]
        if train_raw.empty:
            raise ValueError("Training slice is empty for the requested n_years_train.")
        train_norm = data_norm.loc[train_raw.index]
        meta["train"] = {
            "use_full_record": False,
            "n_years_train": int(n),
            "train_start": str(first_wy),
            "train_end": str(last_wy),
            "resolved": {
                "start": str(pd.Timestamp(train_raw.index.min().date()).date()),
                "end": str(pd.Timestamp(train_raw.index.max().date()).date()),
                "start_token": str(first_wy),
                "end_token": str(last_wy),
                "n_days": int((train_raw.index.max() - train_raw.index.min()).days) + 1,
            },
        }
    else:
        train_raw, w = helpers.slice_by_window(
            data_raw,
            start_token=train_start,
            end_token=train_end,
            label="train_window",
        )
        train_norm = data_norm.loc[train_raw.index]
        meta["train"] = {
            "use_full_record": False,
            "n_years_train": None,
            "train_start": train_start,
            "train_end": train_end,
            "resolved": {
                "start": str(w.start.date()),
                "end": str(w.end.date()),
                "start_token": w.start_token,
                "end_token": w.end_token,
                "n_days": int(w.n_days),
            },
        }

    # -----------------------------------------------------------------
    # Resolve exclusion hole (optional)
    # -----------------------------------------------------------------
    train_allowed_segments: list[tuple[int, int]] | None = None
    if (exclude_start is not None) or (exclude_end is not None):
        if exclude_start is None or exclude_end is None:
            raise ValueError("exclude_start and exclude_end must both be provided.")

        ex_w = helpers.resolve_window_in_df(
            train_raw,
            start_token=exclude_start,
            end_token=exclude_end,
            label="exclude_window",
        )
        i0, i1 = helpers.window_to_index_range(train_raw, ex_w, label="exclude_window")
        keep = helpers.complement_index_ranges((0, len(train_raw) - 1), [(i0, i1)])
        keep = [r for r in keep if (r[1] - r[0] + 1) >= 2]
        if not keep:
            raise ValueError("Exclusion removes all training data (no remaining segment with >=2 days).")
        train_allowed_segments = keep

        meta["exclude"] = {
            "exclude_start": str(exclude_start),
            "exclude_end": str(exclude_end),
            "resolved": {
                "start": str(ex_w.start.date()),
                "end": str(ex_w.end.date()),
                "n_days": int(ex_w.n_days),
            },
            "segments": [
                {
                    "start": str(pd.Timestamp(train_raw.index[a].date()).date()),
                    "end": str(pd.Timestamp(train_raw.index[b].date()).date()),
                    "n_days": int(b - a + 1),
                }
                for a, b in keep
            ],
        }

    # -----------------------------------------------------------------
    # Resolve optional validation window (used only for periodic validation)
    # -----------------------------------------------------------------
    val_raw: pd.DataFrame | None = None
    val_norm: pd.DataFrame | None = None
    if (val_start is not None) or (val_end is not None):
        if val_start is None or val_end is None:
            raise ValueError("val_start and val_end must both be provided.")

        val_raw, v = helpers.slice_by_window(
            data_raw,
            start_token=val_start,
            end_token=val_end,
            label="val_window",
        )
        val_norm = data_norm.loc[val_raw.index]
        meta["val"] = {
            "val_start": str(val_start),
            "val_end": str(val_end),
            "resolved": {
                "start": str(v.start.date()),
                "end": str(v.end.date()),
                "start_token": v.start_token,
                "end_token": v.end_token,
                "n_days": int(v.n_days),
            },
        }

    return {
        "train_raw": train_raw,
        "train_norm": train_norm,
        "train_allowed_segments": train_allowed_segments,
        "val_raw": val_raw,
        "val_norm": val_norm,
    }, meta


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def infer_run_dir_from_model_path(model_path: Path) -> Path:
    """Return a best-effort "run directory" for a model path.

    Heuristic:
      - If model is in a directory containing run_manifest.json -> that directory
      - Else, check parent directory
      - Else, return model_path.parent
    """
    model_path = Path(model_path)
    cand = model_path.parent
    if (cand / "run_manifest.json").exists():
        return cand
    if (cand.parent / "run_manifest.json").exists():
        return cand.parent
    return cand


# ---------------------------------------------------------------------
# Agent / reward / env builders
# ---------------------------------------------------------------------
def build_agent(
    env: gym.Env,
    algo: str = "ppo",
    seed: int | None = None,
    *,
    device: str = "auto",
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    gamma: float | None = None,
) -> PPO:
    algo = algo.lower()
    if algo != "ppo":
        raise ValueError(f"Unsupported algo: {algo}")

    ppo_kwargs: dict = {
        "verbose": 1,
        "seed": seed,
        "device": device,
    }
    if n_steps is not None:
        ppo_kwargs["n_steps"] = n_steps
    if batch_size is not None:
        ppo_kwargs["batch_size"] = batch_size
    if n_epochs is not None:
        ppo_kwargs["n_epochs"] = n_epochs
    if gamma is not None:
        ppo_kwargs["gamma"] = gamma

    return PPO("MlpPolicy", env, **ppo_kwargs)


def build_reward(reward_spec_str: str):
    spec = drl_rewards.parse_objective_spec(reward_spec_str)
    composite = drl_rewards.build_composite_reward(spec, weights=None)
    return composite


def make_env(
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    *,
    episode_length: int | None,
    allowed_segments: list[tuple[int, int]] | None = None,
    is_eval: bool = False,
) -> gym.Env:
    """
    Build a single NavajoReservoirEnv.

    - Training: pass episode_length (e.g., 3600) -> random-window episodes inside env.reset
    - Eval: pass episode_length=None -> full series
    """
    reward_fn = build_reward(reward_spec_str)

    env = NavajoReservoirEnv(
        data_raw=data_raw,
        data_norm=data_norm,
        norm_stats=norm_stats,
        reward_fn=reward_fn,
        episode_length=episode_length,
        allowed_segments=allowed_segments,
        is_eval=is_eval,
    )
    return env


def make_vec_env(
    *,
    data_raw: pd.DataFrame,
    data_norm: pd.DataFrame,
    norm_stats: pd.DataFrame,
    reward_spec_str: str,
    n_envs: int,
    episode_length: int | None,
    allowed_segments: list[tuple[int, int]] | None = None,
    is_eval: bool = False,
    use_subproc: bool = False,
):
    """
    Build a VecEnv of NavajoReservoirEnv instances.
    """
    vec_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    def _make_single_env():
        return make_env(
            data_raw=data_raw,
            data_norm=data_norm,
            norm_stats=norm_stats,
            reward_spec_str=reward_spec_str,
            episode_length=episode_length,
            allowed_segments=allowed_segments,
            is_eval=is_eval,
        )

    env_fns = [_make_single_env for _ in range(n_envs)]
    return vec_cls(env_fns)


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------
def run_rollout_window(
    *,
    model_path: Path | str,
    reward_spec: str,
    window_start: str | None = None,
    window_end: str | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    """Deterministic rollout over an arbitrary time window.

    Parameters
    ----------
    model_path:
        Path to the SB3 .zip file.
    reward_spec:
        Objective specification string used to build the env reward.
    window_start, window_end:
        Either 4-digit water year tokens (e.g. '2002') or 'YYYY-MM-DD'.
        If omitted, uses the full available model_data period.

    Returns
    -------
    DataFrame indexed by date.
    """
    model_path = Path(model_path)

    all_data = load_all_model_data()
    raw_all = all_data["raw"]
    norm_all = all_data["norm"]
    norm_stats = all_data["norm_stats"]

    raw_slice, win = helpers.slice_by_window(
        raw_all,
        start_token=window_start,
        end_token=window_end,
        label="eval_window",
    )
    norm_slice = norm_all.loc[raw_slice.index]

    eval_env = make_env(
        data_raw=raw_slice,
        data_norm=norm_slice,
        norm_stats=norm_stats,
        reward_spec_str=reward_spec,
        episode_length=None,
        is_eval=True,
    )

    agent = PPO.load(model_path, device=device)

    obs, _ = eval_env.reset()
    step = 0
    records: list[dict] = []

    def _to_float_or_nan(val: object) -> float:
        return np.nan if val is None else float(val)

    while True:
        # Date
        if hasattr(eval_env, "_current_date") and callable(eval_env._current_date):  # type: ignore[attr-defined]
            date = eval_env._current_date()  # type: ignore[attr-defined]
        else:
            t = getattr(eval_env, "t", step)
            start_idx = getattr(eval_env, "start_idx", 0)
            global_idx = start_idx + t
            date = eval_env.data_raw.index[global_idx]

        # Agent internal state before step
        storage_agent_af = float(eval_env.storage_af)
        elev_agent_ft = float(eval_env._capacity_to_elev_scalar(storage_agent_af)) if hasattr(eval_env, "_capacity_to_elev_scalar") else float(eval_env.capacity_to_elev(storage_agent_af))

        # Step
        action, _ = agent.predict(obs, deterministic=True)
        action_arr = np.asarray(action, dtype=float).reshape(-1)
        next_obs, reward, terminated, truncated, info_step = eval_env.step(action)
        done = bool(terminated or truncated)

        rec: dict[str, float | int | pd.Timestamp] = {
            "step": step,
            "date": pd.to_datetime(date),
            "reward": float(reward),
            "storage_agent_af": storage_agent_af,
            "elev_agent_ft": elev_agent_ft,
        }
        for i, val in enumerate(action_arr.tolist()):
            rec[f"action_{i}"] = float(val)

        # Reward components
        comps = info_step.get("reward_components", {})
        for k, v in comps.items():
            rec[f"rc_{k}"] = float(v)

        # State after step / operational diagnostics
        if "storage_af" in info_step:
            rec["storage_agent_af_end"] = _to_float_or_nan(info_step["storage_af"])
        if "elev_ft" in info_step:
            rec["elev_agent_ft_end"] = _to_float_or_nan(info_step["elev_ft"])
        elif "storage_agent_af_end" in rec:
            rec["elev_agent_ft_end"] = (
                float(eval_env._capacity_to_elev_scalar(rec["storage_agent_af_end"]))
                if hasattr(eval_env, "_capacity_to_elev_scalar")
                else float(eval_env.capacity_to_elev(rec["storage_agent_af_end"]))
            )

        # Component releases and operational fields (use env's native keys where possible)
        passthrough_keys = [
            "release_sj_main_cfs",
            "release_niip_cfs",
            "sj_main_flow_cfs",
            "sj_at_farmington_cfs",
            "sj_at_farmington_lag2_cfs",
            "spill_cfs",
            "spill_af",
            "deadpool_block",
            "release_cap_penalty",
            "release_phys_penalty",
            "available_af",
            "total_release_af",
            "total_controlled_release_af",
            "prev_elev_ft",
            "deadpool_elev_ft",
            "spill_elev_ft",
        ]
        for key in passthrough_keys:
            if key in info_step:
                val = info_step[key]
                if key == "deadpool_block":
                    rec[key] = int(bool(val))
                else:
                    rec[key] = _to_float_or_nan(val)

        if "deadpool_storage_af" in info_step:
            rec["deadpool_storage_af"] = _to_float_or_nan(info_step["deadpool_storage_af"])
            rec.setdefault("min_storage_af", _to_float_or_nan(info_step["deadpool_storage_af"]))
        if "max_storage_af" in info_step:
            rec["max_storage_af"] = _to_float_or_nan(info_step["max_storage_af"])

        if "total_release_cfs" in info_step:
            rec["release_agent_cfs"] = _to_float_or_nan(info_step["total_release_cfs"])
        if "total_controlled_release_cfs" in info_step:
            rec["release_agent_controlled_cfs"] = _to_float_or_nan(info_step["total_controlled_release_cfs"])

        req_sj = info_step.get("requested_release_sj_main_cfs")
        req_niip = info_step.get("requested_release_niip_cfs")
        if req_sj is not None:
            rec["requested_release_sj_main_cfs"] = _to_float_or_nan(req_sj)
        if req_niip is not None:
            rec["requested_release_niip_cfs"] = _to_float_or_nan(req_niip)
        if req_sj is not None or req_niip is not None:
            rec["requested_total_release_cfs"] = float((req_sj or 0.0) + (req_niip or 0.0))

        # Historic row
        date_row = eval_env.data_raw.loc[rec["date"]]

        if "storage_af" in date_row.index:
            rec["storage_hist_af"] = float(date_row["storage_af"])

        for col in [
            "release_cfs",
            "inflow_cfs",
            "evap_af",
            "sj_farmington_q_cfs",
            "animas_farmington_q_cfs",
            "sj_bluff_q_cfs",
        ]:
            if col in date_row.index:
                rec[col] = float(date_row[col])

        # Convenience conversion for plotting: evap_af (acre-feet/day) -> evap_cfs
        # 1 AF = 43,560 ft^3 ; 1 day = 86,400 s
        if "evap_af" in rec and "evap_cfs" not in rec:
            rec["evap_cfs"] = float(rec["evap_af"]) * (43560.0 / 86400.0)

        # Hydropower: prefer the env-computed value (controlled SJ release + end-of-step elevation)
        if "hydropower_mwh" in info_step:
            rec["hydro_agent_mwh"] = _to_float_or_nan(info_step["hydropower_mwh"])
        elif "release_sj_main_cfs" in rec:
            elev_for_hp = float(rec.get("elev_agent_ft_end", elev_agent_ft))
            hp_agent = navajo_power_generation_scalar(
                cfs_value=float(rec["release_sj_main_cfs"]),
                elevation_ft=elev_for_hp,
            )
            rec["hydro_agent_mwh"] = float(hp_agent)

        # Historic hydropower: use historic total release + elevation from historic storage
        if "storage_hist_af" in rec and "release_cfs" in rec:
            elev_hist_ft = float(eval_env._capacity_to_elev_scalar(rec["storage_hist_af"])) if hasattr(eval_env, "_capacity_to_elev_scalar") else float(eval_env.capacity_to_elev(rec["storage_hist_af"]))
            hp_hist = navajo_power_generation_scalar(
                cfs_value=float(rec["release_cfs"]),
                elevation_ft=elev_hist_ft,
            )
            rec["elev_hist_ft"] = elev_hist_ft
            rec["hydro_hist_mwh"] = float(hp_hist)

        records.append(rec)

        obs = next_obs
        step += 1
        if done:
            break

    df = pd.DataFrame.from_records(records).set_index("date").sort_index()
    return df


def run_test_rollout(
    run_dir: Path | str,
    reward_spec: str,
    *,
    n_years_test: int = 8,
    model_name: str = "last_model",
    device: str = "auto",
) -> pd.DataFrame:
    """Backward-compatible convenience wrapper: evaluate on the last N water years."""
    run_dir = Path(run_dir)
    model_path = run_dir / model_name
    if model_path.suffix == "":
        model_path = model_path.with_suffix(".zip")

    # Resolve to a concrete window (last N water years) using a default split.
    all_data = load_all_model_data()
    split = _default_train_eval_split(all_data["raw"], all_data["norm"], n_years_test=n_years_test)
    w_start = split["eval_raw"].index.min().strftime("%Y-%m-%d")
    w_end = split["eval_raw"].index.max().strftime("%Y-%m-%d")
    return run_rollout_window(
        model_path=model_path,
        reward_spec=reward_spec,
        window_start=w_start,
        window_end=w_end,
        device=device,
    )


def evaluate_model_window(
    *,
    model_path: Path | str,
    reward_spec: str,
    window_start: str | None,
    window_end: str | None,
    outdir: Path | str,
    device: str = "auto",
    which_metrics: str = "core",
    save_rollout: bool = True,
    save_plots: bool = True,
    save_metrics: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate a model on a specified window and write outputs to outdir."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_rollout_window(
        model_path=model_path,
        reward_spec=reward_spec,
        window_start=window_start,
        window_end=window_end,
        device=device,
    )

    if save_rollout:
        stem = "eval_rollout"
        out_path = outdir / f"{stem}.parquet"
        df.to_parquet(out_path)
        df.to_csv(out_path.with_suffix(".csv"), index=True)

    df_train_updates = None
    if save_plots:
        # Best-effort: load per-PPO-update mean reward components so reward plots
        # can be exported alongside eval plots. This file is produced during
        # training when track_reward_components=True.
        try:
            run_dir = infer_run_dir_from_model_path(Path(model_path))
            upd_path = run_dir / "train_update_metrics.parquet"
            if upd_path.exists():
                df_train_updates = pd.read_parquet(upd_path)
        except Exception:
            df_train_updates = None

        plot_dir = outdir / "plots"
        drl_plotting.save_plots(
            df_test=df,
            outdir=plot_dir,
            df_train_updates=df_train_updates,
            which="all",
        )

    metrics_df = drl_metrics.compute_metrics(df, which=which_metrics, validate=True)
    if save_metrics:
        drl_metrics.save_metrics(
            df_test=df,
            outdir=outdir,
            which=which_metrics,
            stem="eval_metrics",
            validate=True,
        )
    return df, metrics_df

# ---------------------------------------------------------------------
# DRLModel class: single training entry point
# ---------------------------------------------------------------------
class DRLModel:
    """
    Single entry point to train/eval, Colab-equivalent intent.

    Key Colab behavior:
    - episode_length_train=3600
    - n_episodes=350
    - total_timesteps = n_episodes * episode_length_train
    """

    def __init__(
        self,
        reward_spec: str,
        *,
        # Training window selection (exactly one should be used by CLI):
        #   - use_full_record
        #   - n_years_train (last N water years)
        #   - train_start/train_end tokens
        n_years_train: int | None = None,
        use_full_record: bool = False,
        train_start: str | None = None,
        train_end: str | None = None,
        # Optional single "hole" exclusion
        exclude_start: str | None = None,
        exclude_end: str | None = None,
        # Optional periodic validation window (during training)
        val_start: str | None = None,
        val_end: str | None = None,
        # Convenience default used by evaluate_test()
        n_years_test: int = 10,
        algo: str = "ppo",
        logdir: str | Path = "runs/debug",
        seed: int | None = None,
        device: str = "auto",
        gamma: float | None = None,
        n_envs: int = 1,
        use_subproc_vec: bool = False,
        episode_length_train: int = 3600,
    ) -> None:

        self.n_years_test = int(n_years_test)
        self.reward_spec = str(reward_spec)
        self.algo = algo.lower()
        self.logdir = Path(logdir)
        self.seed = seed
        self.device = device
        self.n_envs = int(n_envs)
        self.use_subproc_vec = bool(use_subproc_vec)
        self.gamma = gamma
        self.episode_length_train = int(episode_length_train)

        self.n_years_train = None if n_years_train is None else int(n_years_train)
        self.use_full_record = bool(use_full_record)
        self.train_start = train_start
        self.train_end = train_end
        self.exclude_start = exclude_start
        self.exclude_end = exclude_end
        self.val_start = val_start
        self.val_end = val_end

        self._train_updates_cb: TrainUpdateRewardComponentsCallback | None = None
        self.train_update_metrics_: pd.DataFrame | None = None

        # Best-effort load previous metrics
        try:
            self.load_train_update_metrics()
        except Exception:
            pass

        # Load full data once, then slice into train/eval windows.
        self._all_data = load_all_model_data()
        all_raw = self._all_data["raw"]
        all_norm = self._all_data["norm"]
        self.norm_stats = self._all_data["norm_stats"]

        self._data_range = helpers.available_date_range(all_raw)

        self.datasets, self._window_meta = _slice_train_val_windows(
            all_raw,
            all_norm,
            use_full_record=self.use_full_record,
            n_years_train=self.n_years_train,
            train_start=self.train_start,
            train_end=self.train_end,
            exclude_start=self.exclude_start,
            exclude_end=self.exclude_end,
            val_start=self.val_start,
            val_end=self.val_end,
        )

        # TRAIN ENV (random 3600-step episodes)
        if self.n_envs == 1:
            self.train_env = make_env(
                data_raw=self.datasets["train_raw"],
                data_norm=self.datasets["train_norm"],
                norm_stats=self.norm_stats,
                reward_spec_str=self.reward_spec,
                episode_length=self.episode_length_train,
                allowed_segments=self.datasets.get("train_allowed_segments", None),
                is_eval=False,
            )
            self.train_env = Monitor(self.train_env)
        else:
            self.train_env = make_vec_env(
                data_raw=self.datasets["train_raw"],
                data_norm=self.datasets["train_norm"],
                norm_stats=self.norm_stats,
                reward_spec_str=self.reward_spec,
                n_envs=self.n_envs,
                episode_length=self.episode_length_train,
                allowed_segments=self.datasets.get("train_allowed_segments", None),
                is_eval=False,
                use_subproc=self.use_subproc_vec,
            )

        # Optional periodic validation env (full-series over val window)
        self.val_env: gym.Env | None = None
        if self.datasets.get("val_raw") is not None:
            self.val_env = make_env(
                data_raw=self.datasets["val_raw"],
                data_norm=self.datasets["val_norm"],
                norm_stats=self.norm_stats,
                reward_spec_str=self.reward_spec,
                episode_length=None,
                is_eval=True,
            )
            self.val_env = Monitor(self.val_env)

        self.agent: PPO | None = None

    def train(
        self,
        *,
        n_episodes: int = 350,
        total_timesteps: int | None = None,
        val_freq: int | None = None,
        device: str | None = None,
        n_steps: int | None = None,
        batch_size: int | None = None,
        n_epochs: int | None = None,
        gamma: float | None = None,
        track_reward_components: bool = True,
        resume: bool = False,
    ) -> None:
        """
        Train PPO.

        Original code equivalence:
          total_timesteps = n_episodes * episode_length_train

        Note: This matches the original code style most closely when n_envs=1.
        If n_envs>1, SB3 will collect more samples per wall-clock step.
        """
        device_eff = device or self.device
        total_timesteps_eff = int(total_timesteps) if total_timesteps is not None else int(n_episodes * self.episode_length_train)

        # -----------------------------------------------------------------
        # Run manifest (written at train start; updated at train end)
        # -----------------------------------------------------------------
        manifest_path = self.logdir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = _read_json(manifest_path)
            except Exception:
                manifest = {}
        else:
            manifest = {}

        manifest.setdefault("project", "DeepReservoir")
        manifest.setdefault("reservoir", "Navajo")
        manifest.setdefault("logdir", str(self.logdir))
        manifest.setdefault("created_at", manifest.get("created_at", _now_iso()))
        manifest["last_updated_at"] = _now_iso()

        # static-ish config
        manifest["config"] = {
            "algo": self.algo,
            "reward_spec": self.reward_spec,
            "seed": self.seed,
            "device": device_eff,
            "gamma": self.gamma if self.gamma is not None else gamma,
            "n_envs": self.n_envs,
            "use_subproc_vec": self.use_subproc_vec,
            "episode_length_train": self.episode_length_train,
            "train": self._window_meta.get("train", None),
            "exclude": self._window_meta.get("exclude", None),
            "val": self._window_meta.get("val", None),
            "data_range": {
                "start": str(self._data_range["start"].date()),
                "end": str(self._data_range["end"].date()),
                "n_days": int(self._data_range["n_days"]),
                "min_water_year": int(self._data_range["min_water_year"]),
                "max_water_year": int(self._data_range["max_water_year"]),
            },
        }

        inv = {
            "started_at": _now_iso(),
            "resume": bool(resume),
            "requested_total_timesteps": int(total_timesteps_eff),
            "requested_n_episodes": int(n_episodes),
            "val_freq": (None if val_freq is None else int(val_freq)),
            "track_reward_components": bool(track_reward_components),
            "ppo_args": {
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": n_epochs,
                "gamma": gamma,
            },
        }
        manifest.setdefault("train_invocations", [])
        manifest["train_invocations"].append(inv)
        _write_json(manifest_path, manifest)

        # -----------------------------------------------------------------
        # Agent creation / resume
        # -----------------------------------------------------------------
        if self.agent is None or not resume:
            self.agent = build_agent(
                self.train_env,
                algo=self.algo,
                seed=self.seed,
                device=device_eff,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
            )
        else:
            # Resume training: keep the loaded agent hyperparams/optimizer.
            try:
                self.agent.set_env(self.train_env)
            except Exception:
                pass

        best_dir = self.logdir / "best"
        val_log_dir = self.logdir / "periodic_val"
        best_dir.mkdir(parents=True, exist_ok=True)
        val_log_dir.mkdir(parents=True, exist_ok=True)

        cb_list: list[BaseCallback] = []
        if self.val_env is not None and val_freq is not None:
            cb_list.append(
                EvalCallback(
                    self.val_env,
                    best_model_save_path=str(best_dir),
                    log_path=str(val_log_dir),
                    eval_freq=int(val_freq),
                    deterministic=True,
                    render=False,
                )
            )
        self._train_updates_cb = None

        # If resuming and tracking reward components, append to existing file.
        prev_updates: pd.DataFrame | None = None
        update_idx_offset = 0
        if resume and track_reward_components:
            try:
                prev_updates = self.load_train_update_metrics()
                if prev_updates is not None and "update_idx" in prev_updates.columns:
                    update_idx_offset = int(prev_updates["update_idx"].max()) + 1
            except Exception:
                prev_updates = None
                update_idx_offset = 0

        if track_reward_components:
            self._train_updates_cb = TrainUpdateRewardComponentsCallback(start_update_idx=update_idx_offset)
            cb_list.append(self._train_updates_cb)

        callback: BaseCallback | None
        if not cb_list:
            callback = None
        else:
            callback = cb_list[0] if len(cb_list) == 1 else CallbackList(cb_list)

        try:
            self.agent.learn(
                total_timesteps=int(total_timesteps_eff),
                callback=callback,
                reset_num_timesteps=(not resume),
            )
            self.save_model("last_model")
        finally:
            if self._train_updates_cb is not None:
                df_upd = pd.DataFrame(self._train_updates_cb.update_history)
                if prev_updates is not None and not prev_updates.empty:
                    df_upd = pd.concat([prev_updates, df_upd], ignore_index=True)
                self.train_update_metrics_ = df_upd
                out_path = self.logdir / "train_update_metrics.parquet"
                df_upd.to_parquet(out_path, index=False)
                df_upd.to_csv(out_path.with_suffix(".csv"), index=False)

            # Update manifest with end-of-run info (best-effort)
            try:
                manifest = _read_json(manifest_path) if manifest_path.exists() else {}
                manifest["last_updated_at"] = _now_iso()
                if manifest.get("train_invocations"):
                    manifest["train_invocations"][-1]["ended_at"] = _now_iso()
                    if self.agent is not None:
                        manifest["train_invocations"][-1]["agent_num_timesteps"] = int(getattr(self.agent, "num_timesteps", 0))
                # model paths
                manifest["artifacts"] = {
                    "last_model": str((self.logdir / "last_model.zip").resolve()),
                    "best_model_dir": str(best_dir.resolve()),
                    "val_log_dir": str(val_log_dir.resolve()),
                }
                _write_json(manifest_path, manifest)
            except Exception:
                pass

    def save_model(self, name: str = "last_model") -> Path:
        if self.agent is None:
            raise RuntimeError("No agent to save (train or load a model first).")
        path = self.logdir / name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path))
        return path

    def load_model(self, path_or_name: str = "last_model") -> PPO:
        """Load a saved SB3 model.

        Parameters
        ----------
        path_or_name:
            Either a path to a .zip model file, or a name resolved within self.logdir.
        """
        cand = Path(path_or_name)
        if cand.exists():
            path = cand
        else:
            path = self.logdir / path_or_name
        if path.suffix == "":
            path = path.with_suffix(".zip")
        self.agent = PPO.load(path, env=self.train_env, device=self.device)
        return self.agent

    def load_train_update_metrics(self) -> pd.DataFrame | None:
        path = self.logdir / "train_update_metrics.parquet"
        if not path.exists():
            return None
        df_upd = pd.read_parquet(path)
        self.train_update_metrics_ = df_upd
        return df_upd

    def evaluate_test(
        self,
        model_name: str = "last_model",
        *,
        save_rollout: bool = True,
        save_plots: bool = True,
        save_metrics: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = run_test_rollout(
            run_dir=self.logdir,
            reward_spec=self.reward_spec,
            n_years_test=self.n_years_test,
            model_name=model_name,
            device=self.device,
        )

        if save_rollout:
            out_path = self.logdir / "eval_test_rollout.parquet"
            df.to_parquet(out_path)
            df.to_csv(out_path.with_suffix(".csv"), index=True)

        if save_plots:
            df_upd = getattr(self, "train_update_metrics_", None)
            if df_upd is None:
                df_upd = self.load_train_update_metrics()

            outdir = self.logdir / "eval_plots"
            _ = drl_plotting.save_plots(
                df_test=df,
                outdir=outdir,
                df_train_updates=df_upd,
                which="all",
            )

        metrics_df = drl_metrics.compute_metrics(df, which="core", validate=True)

        if save_metrics:
            drl_metrics.save_metrics(
                df_test=df,
                outdir=self.logdir,
                which="core",
                stem="eval_metrics",
                validate=True,
            )

        return df, metrics_df



class TrainUpdateRewardComponentsCallback(BaseCallback):
    """
    Per-rollout (per PPO update) mean reward component tracker.
    Expects env to expose:
      info["reward_components_step"] OR info["reward_components"].
    """

    def __init__(self, *, start_update_idx: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.rollout_sums: dict[str, float] = {}
        self.rollout_reward_sum: float = 0.0
        self.rollout_count: int = 0
        self.update_history: list[dict] = []
        self._update_idx: int = 0
        self._start_update_idx: int = int(start_update_idx)

    def _on_training_start(self) -> None:
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
        self.update_history = []
        self._update_idx = int(self._start_update_idx)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)

        if rewards is not None:
            try:
                self.rollout_reward_sum += float(np.sum(rewards))
            except Exception:
                self.rollout_reward_sum += float(rewards)

        if infos:
            for info in infos:
                comps = info.get("reward_components_step", info.get("reward_components", {}))
                if not comps:
                    continue
                for k, v in comps.items():
                    try:
                        fv = float(v)
                    except Exception:
                        continue
                    self.rollout_sums[k] = self.rollout_sums.get(k, 0.0) + fv

            self.rollout_count += len(infos)
        else:
            self.rollout_count += 1

        return True

    def _on_rollout_end(self) -> None:
        count = max(int(self.rollout_count), 1)

        rec: dict[str, float | int] = {
            "update_idx": int(self._update_idx),
            "timesteps": int(self.num_timesteps),
            "rollout_steps": int(self.rollout_count),
            "mean_total_reward": float(self.rollout_reward_sum / count),
        }
        for k, total in self.rollout_sums.items():
            rec[f"mean_{k}"] = float(total / count)

        self.update_history.append(rec)
        self._update_idx += 1

        # reset for next rollout
        self.rollout_sums = {}
        self.rollout_reward_sum = 0.0
        self.rollout_count = 0
