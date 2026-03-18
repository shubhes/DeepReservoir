# deepreservoir/drl/rewards.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Tuple, Optional, Mapping, Union

import numpy as np
import pandas as pd

from deepreservoir.define_env.niip.niip_demand import niip_daily_demand
from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve


# ---------------------------------------------------------------------
# Types & context
# ---------------------------------------------------------------------

@dataclass
class RewardContext:
    """
    Information needed to compute reward for a single step.

    Notes on alpha
    --------------
    Each reward component can be assigned an ``alpha`` via the reward spec.
    The CompositeReward sets ``ctx.alpha`` (and ctx.objective/ctx.variant)
    before calling each component function.

    Convention (for now): reward functions should return an *unscaled* value
    and CompositeReward applies the alpha multiplier.
    """
    t: int                       # step index (global idx)
    date: pd.Timestamp           # current date
    obs: np.ndarray              # current observation (normalized)
    action: np.ndarray           # current action (raw [-1,1])
    next_obs: np.ndarray         # next observation (normalized)
    info: Dict[str, Any]         # extra per-step info (raw series, flags, etc.)

    # Set by CompositeReward before calling each component reward function.
    objective: str | None = None
    variant: str | None = None
    alpha: float = 1.0


RewardFn = Callable[[RewardContext], float]


def _animas_farmington_cfs(ctx: RewardContext) -> float:
    """Return the Animas-at-Farmington flow from the lean scalar field when
    present, falling back to the legacy raw_forcings container.
    """
    if "animas_farmington_q_cfs" in ctx.info:
        return float(ctx.info.get("animas_farmington_q_cfs", 0.0))

    row = ctx.info.get("raw_forcings", None)
    if row is None:
        return 0.0

    getter = getattr(row, "get", None)
    if getter is None:
        return 0.0

    try:
        return float(getter("animas_farmington_q_cfs", 0.0))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------
# Registry plumbing
# ---------------------------------------------------------------------

OBJECTIVES = [
    "dam_safety",
    "esa_min_flow",
    "esa_spring_peak_release",
    "flooding",
    "hydropower",
    "niip",
    # "physics",
]

REWARD_REGISTRY: Dict[str, Dict[str, RewardFn]] = {obj: {} for obj in OBJECTIVES}


def register_reward(objective: str, variant: str) -> Callable[[RewardFn], RewardFn]:
    """
    Decorator to register a reward function under an objective + variant name.
    """
    if objective not in REWARD_REGISTRY:
        raise KeyError(f"Unknown objective {objective!r}. Expected one of {OBJECTIVES}.")

    def decorator(fn: RewardFn) -> RewardFn:
        if variant in REWARD_REGISTRY[objective]:
            raise ValueError(f"Reward already registered for {objective}:{variant}")
        REWARD_REGISTRY[objective][variant] = fn
        return fn

    return decorator


@dataclass
class RewardComponent:
    objective: str
    variant: str
    alpha: float
    weight: float
    fn: RewardFn

    @property
    def key(self) -> str:
        return f"{self.objective}.{self.variant}"


class CompositeReward:
    """
    Combines multiple RewardComponents into a single scalar reward.

    Call with RewardContext; returns (total_reward, component_breakdown_dict).
    """

    def __init__(self, components: List[RewardComponent]):
        self.components = components

    def __call__(self, ctx: RewardContext) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        breakdown: Dict[str, float] = {}

        for comp in self.components:
            # Provide component context to the reward function.
            ctx.objective = comp.objective
            ctx.variant = comp.variant
            ctx.alpha = float(comp.alpha)

            base = float(comp.fn(ctx))
            # For now, alpha is a simple multiplier applied outside the reward
            # function (reward functions may *read* ctx.alpha in the future).
            r = float(comp.alpha) * base
            weighted = comp.weight * r
            breakdown[comp.key] = float(weighted)
            total += weighted

        return float(total), breakdown


def build_composite_reward(
    spec: Mapping[str, Union[str, "ObjectiveSpec"]],
    weights: Optional[Dict[str, float]] = None,
) -> CompositeReward:
    """
    spec: mapping objective -> variant
    weights: optional mapping objective -> scalar weight
    """
    components: List[RewardComponent] = []
    weights = weights or {}

    for objective, sel in spec.items():
        if isinstance(sel, ObjectiveSpec):
            variant = sel.variant
            alpha = float(sel.alpha)
        else:
            variant = str(sel)
            alpha = 1.0

        if objective not in REWARD_REGISTRY:
            raise KeyError(
                f"Unknown objective {objective!r}. Known: {list(REWARD_REGISTRY.keys())}"
            )
        variants = REWARD_REGISTRY[objective]
        if variant not in variants:
            raise KeyError(
                f"Unknown variant {variant!r} for objective {objective!r}. "
                f"Known: {list(variants.keys())}"
            )

        fn = variants[variant]
        w = float(weights.get(objective, 1.0))
        components.append(RewardComponent(objective, variant, alpha, w, fn))

    return CompositeReward(components)


@dataclass(frozen=True)
class ObjectiveSpec:
    """Parsed objective selection from reward_spec.

    Attributes
    ----------
    variant
        The registered reward variant name, e.g. "baseline".
    alpha
        A per-objective parameter (currently used as a multiplier). Defaults to 1.
    """

    variant: str
    alpha: float = 1.0


def parse_objective_spec(spec_str: str) -> Dict[str, ObjectiveSpec]:
    """
    Format
    ------
    Comma-separated tokens. Each token may be:

      - "objective"                      -> variant="baseline", alpha=1
      - "objective:variant"              -> alpha=1
      - "objective:variant@alpha"        -> alpha parsed as float

    Examples
    --------
      "dam_safety:storage_band@0.5,esa_min_flow:baseline,hydropower"
    """
    spec: Dict[str, ObjectiveSpec] = {}
    if not spec_str:
        return spec

    pairs = [s.strip() for s in spec_str.split(",") if s.strip()]
    for token in pairs:
        # Allow "objective" (implies baseline)
        if ":" not in token:
            obj = token.strip()
            if not obj:
                continue
            spec[obj] = ObjectiveSpec("baseline", 1.0)
            continue

        obj, rest = [p.strip() for p in token.split(":", 1)]
        if not obj:
            continue

        alpha = 1.0
        variant = rest

        # Variant may include @alpha (preferred)
        if "@" in rest:
            variant, alpha_str = [p.strip() for p in rest.split("@", 1)]
            try:
                alpha = float(alpha_str)
            except Exception:
                alpha = 1.0
        else:
            # If '@' is not present, alpha defaults to 1.0.
            # For robustness, tolerate stray extra ':' segments by taking the first as the variant.
            variant = rest.split(":", 1)[0].strip()

        if not variant:
            variant = "baseline"

        spec[obj] = ObjectiveSpec(variant, alpha)
    return spec


# ---------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------

@register_reward("dam_safety", "baseline")
def dam_safety_baseline(ctx: RewardContext) -> float:
    """
    +1 if elevation < 6100 ft, else 0.
    """
    elev_ft = float(ctx.info["elev_ft"])
    return 1.0 if elev_ft < 6100.0 else 0.0


@register_reward("dam_safety", "storage_band")
def dam_safety_storage_band(ctx: RewardContext) -> float:
    """
    Original reward shaping around a target storage band.
    """
    storage = float(ctx.info["storage_af"])
    s_min = 500_000.0
    s_max = 1_731_750.0
    target = 0.5 * (s_min + s_max)

    span = (s_max - s_min) / 2.0
    if span <= 0:
        return 0.0

    x = (storage - target) / span
    r = 1.0 - x**2
    r = float(np.clip(r, -1.0, 1.0))
    return 3*r

@register_reward("dam_safety", "storage_band_shaped")
def dam_safety_storage_band(ctx: RewardContext) -> float:
    """
    Original reward shaping around a target storage band.
    """
    storage = float(ctx.info["storage_af"])
    s_min = 500_000.0
    s_max = 1_731_750.0
    target = 0.5 * (s_min + s_max)

    span = (s_max - s_min) / 2.0
    if span <= 0:
        return 0.0

    x = (storage - target) / span
    r = 1.0 - x**2
    return r


@register_reward("dam_safety", "storage_band_env")
def dam_safety_storage_band_env(ctx: RewardContext) -> float:
    """Storage-band shaping using environment-provided bounds.

    Uses deadpool_storage_af and max_storage_af from ctx.info (set by the env).
    This keeps the reward consistent with elevation-based spill/deadpool changes.
    """
    storage = float(ctx.info["storage_af"])
    s_min = float(ctx.info.get("deadpool_storage_af", 0.0))
    s_max = float(ctx.info.get("max_storage_af", 1.0))
    # Defensive clamps (interpolators may extrapolate slightly negative).
    if s_min < 0.0:
        s_min = 0.0
    if s_max <= s_min:
        return 0.0

    target = 0.5 * (s_min + s_max)
    span = (s_max - s_min) / 2.0
    x = (storage - target) / span
    r = 1.0 - x**2
    return float(np.clip(r, -1.0, 1.0))


@register_reward("dam_safety", "storage_fraction")
def dam_safety_storage_fraction(ctx: RewardContext) -> float:
    """Monotonic storage reward in [0, 1] using env bounds.

    0 at/below deadpool_storage_af, 1 at/above max_storage_af.
    Useful when you simply want the agent to retain water.
    """
    storage = float(ctx.info["storage_af"])
    s_min = float(ctx.info.get("deadpool_storage_af", 0.0))
    s_max = float(ctx.info.get("max_storage_af", 1.0))
    if s_min < 0.0:
        s_min = 0.0
    if s_max <= s_min:
        return 0.0

    frac = (storage - s_min) / (s_max - s_min)
    return float(np.clip(frac, 0.0, 1.0))


@register_reward("esa_min_flow", "baseline")
def esa_min_flow_baseline(ctx: RewardContext) -> float:
    """
    +1 if (Animas at Farmington + San Juan mainstem release) >= 500 cfs, else 0.
    Uses release_sj_main_cfs as component #1 (regular/mainstem release).
    """
    animas_cfs = _animas_farmington_cfs(ctx)
    release_sj_main_cfs = float(ctx.info["release_sj_main_cfs"])
    total_flow_cfs = animas_cfs + release_sj_main_cfs
    return 1.0 if total_flow_cfs >= 500.0 else 0.0


@register_reward("flooding", "baseline")
def flooding_baseline(ctx: RewardContext) -> float:
    """
    Two-part flood safety:
      +1 if same-day Farmington proxy < 5000 cfs
      +1 if lag-2 proxy < 12000 cfs (or safe if not available early)
    Return average in {0, 0.5, 1}.
    """
    q0 = float(ctx.info.get("sj_at_farmington_cfs", 0.0))
    qlag2 = ctx.info.get("sj_at_farmington_lag2_cfs", None)

    c1 = 1.0 if q0 < 5000.0 else 0.0

    if qlag2 is None:
        c2 = 1.0
    else:
        c2 = 1.0 if float(qlag2) < 12000.0 else 0.0

    return 0.5 * (c1 + c2)


@register_reward("hydropower", "baseline")
def hydropower_baseline(ctx: RewardContext) -> float:
    """
    Hydropower reward based on daily generation [MWh/day].

    Mirrors original reward shaping:
      - if < min_mwh -> -0.5
      - else -> linear 0..1 between [min_mwh, max_mwh]
    """
    hydropower_mwh = float(ctx.info["hydropower_mwh"])
    min_mwh = 288.0
    max_mwh = 768.0

    if hydropower_mwh < min_mwh:
        return -0.5
    return float((hydropower_mwh - min_mwh) / (max_mwh - min_mwh))


# ---------------- NIIP ----------------

def _niip_total_season_demand_cfs_sum(doy_start: int = 50, doy_end: int = 300) -> float:
    """
    Precompute a season "total demand" scalar for normalization.
    Since timestep is daily, sum(cfs) is proportional to volume over the season,
    so it works as a stable normalizer.
    """
    doys = np.arange(doy_start, doy_end + 1)
    vals = niip_daily_demand(doys)  
    vals = np.asarray(vals, dtype=float)
    vals = np.clip(vals, 0.0, None)
    total = float(np.sum(vals))
    return total


# cache the scalar so it’s deterministic and cheap
_NIIP_TOTAL_CFS_SUM = _niip_total_season_demand_cfs_sum(50, 300)


@register_reward("niip", "baseline")
def niip_colab_like(ctx: RewardContext) -> float:
    """
    NIIP reward conceptually like teh original code.

    Active DOY 50–300.

    Let:
      D = demand_cfs(doy)
      R = regular release component (we use ctx.info["release_sj_main_cfs"])
      delta = D - R
      reward = 1 - |delta| / TOTAL_SEASON_DEMAND

    Notes:
      - This allows BOTH surplus and shortage to be penalized (same as your Colab).
      - Outside season -> 0.
    """
    doy = int(pd.to_datetime(ctx.date).timetuple().tm_yday)
    if not (50 <= doy <= 300):
        return 0.0

    demand_cfs = float(niip_daily_demand(doy))
    if demand_cfs <= 0.0 or _NIIP_TOTAL_CFS_SUM <= 0.0:
        return 0.0

    
    niip_release_cfs = float(ctx.info["release_niip_cfs"])
    delta = demand_cfs - niip_release_cfs
    r = 1.0 - abs(delta) / (_NIIP_TOTAL_CFS_SUM + 1e-9)

    
    return float(np.clip(r, -1.0, 1.0))

@register_reward("niip", "colab_band")
def niip_colab_band(ctx: RewardContext) -> float:
    """
    NIIP reward matching Colab logic (banded, +/-5%).

    Colab logic:
      if 50 <= doy <= 300:
        daily_demand_af = niip_daily_demand(doy) * 1.98211
        daily_release_af = niip_release_cfs * 1.98211

        if daily_release_af > 1.05*daily_demand_af: r = -0.5
        elif 0.95*demand <= release <= 1.05*demand: r = +1.5
        else: r = -0.5
      else: r = 0
    """
    CFS_TO_AF_PER_DAY = 1.98211

    doy = int(pd.to_datetime(ctx.date).timetuple().tm_yday)
    if not (50 <= doy <= 300):
        return 0.0

    # Demand: niip_daily_demand returns cfs (as used in your Colab),
    # then convert to AF/day
    demand_cfs = float(niip_daily_demand(doy))
    daily_demand_af = demand_cfs * CFS_TO_AF_PER_DAY

    if daily_demand_af <= 0.0:
        return 0.0

    # NIIP release: use NIIP component (NOT sanjuan_release)
    niip_release_cfs = float(ctx.info.get("release_niip_cfs", 0.0))
    daily_release_af = niip_release_cfs * CFS_TO_AF_PER_DAY

    if daily_release_af > (1.05 * daily_demand_af):
        return -2.5
    elif (0.95 * daily_demand_af) <= daily_release_af <= (1.05 * daily_demand_af):
        return 10.5
    else:
        return -2.5

# ---------------- SPR rewards ----------------

_SPRING_PEAK_CURVE = SpringPeakReleaseCurve()


@register_reward("esa_spring_peak_release", "oi")
def esa_spring_peak_oi(ctx: RewardContext) -> float:
    """
    The FIRST SPR reward (OI-based)

    Reads OI from env info:
      ctx.info["spring_oi"] in [0,1] (or NaN)

    Map OI to [-1,1]:
      r = 2*OI - 1
    """
    oi = ctx.info.get("spring_oi", np.nan)
    if oi is None or not np.isfinite(oi):
        return 0.0
    return float(2.0 * float(oi) - 1.0)


@register_reward("esa_spring_peak_release", "curve")
def esa_spring_peak_curve(ctx: RewardContext) -> float:
    """
    SPR curve-matching reward.

    Compares the *regular/mainstem release* (release_sj_main_cfs)
    against the target curve (from SpringPeakReleaseCurve).

    Inside window -> reward in [-1,1]
    Outside window -> 0
    """
    date = pd.to_datetime(ctx.date)
    target = float(_SPRING_PEAK_CURVE.target_cfs_from_date(date))

    if target <= 0.0:
        return 0.0

    actual = float(ctx.info["release_sj_main_cfs"])

    tolerance_cfs = 500.0  # tune (250, 500, 1000)
    err = abs(actual - target)

    r = 1.0 - (err / (tolerance_cfs + 1e-9))
    return float(np.clip(r, -1.0, 1.0))

# @register_reward("esa_spring_peak_release", "farmington_10k")
# def esa_spring_peak_farmington_10k(ctx: RewardContext) -> float:
#     """
#     SPR bonus: during SPR season only, reward if (Animas + San Juan mainstem release) at Farmington >= 10,000 cfs.

#     Uses:
#       - ctx.info["sj_at_farmington_cfs"]  (already computed in env)
#       - SpringPeakReleaseCurve to determine whether we are inside the SPR window

#     Returns:
#       - 0.0 outside SPR window
#       - +1.0 if threshold met during SPR window, else 0.0
#     """
#     date = pd.to_datetime(ctx.date)

#     # Use the SPR curve as the "season gate" (active only when curve is active)
#     target = _SPRING_PEAK_CURVE.target_cfs_from_date(date)
#     if target <= 0.0:
#         return 0.0  # outside SPR window

#     q_farm = float(ctx.info.get("sj_at_farmington_cfs", 0.0))

#     return 1.0 if q_farm >= 10_000.0 else 0.0

@register_reward("esa_spring_peak_release", "farmington_10k")
def esa_spring_peak_farmington_10k(ctx: RewardContext) -> float:
    """
    SPR window reward: encourage (Animas + San Juan mainstem release) >= 10,000 cfs at Farmington,
    and favor "simultaneous-ish" peaking (both contribute meaningfully).

    Uses:
      - ctx.date
      - ctx.info["raw_forcings"]["animas_farmington_q_cfs"]
      - ctx.info["release_sj_main_cfs"]

    Returns:
      - 0 outside SPR window
      - [0, 1] inside window
    """
    date = pd.to_datetime(ctx.date)

    # Use the SPR curve to define the "SPR window"
    target = _SPRING_PEAK_CURVE.target_cfs_from_date(date)
    if target <= 0.0:
        return 0.0

    animas = _animas_farmington_cfs(ctx)
    sanjuan = float(ctx.info.get("release_sj_main_cfs", 0.0))

    total = animas + sanjuan

    # Gate: only reward if we reach the threshold
    threshold = 10_000.0
    if total < threshold:
        return 0.0

    # How far above threshold? (soft ramp up to 1.0)
    # e.g., +0 at 10k, +1 at 12k
    ramp = 2_000.0
    magnitude = (total - threshold) / ramp
    magnitude = float(np.clip(magnitude, 0.0, 1.0))

    # "Simultaneous peak" encouragement:
    # balance ~ 1 if animas ≈ sanjuan, ~0 if one dominates the sum.
    balance = 1.0 - (abs(animas - sanjuan) / (total + 1e-6))
    balance = float(np.clip(balance, 0.0, 1.0))

    return 30 * magnitude * balance

@register_reward("esa_spring_peak_release", "farmington_10k_shaped")
def spr_farmington_10k_shaped(ctx: RewardContext) -> float:
    """
    Reward Farmington discharge approaching/exceeding 10k during SPR window.
    Farmington approx = Animas(gauge) + San Juan mainstem release (agent)
    Uses spring_oi as the SPR window mask (OI>0 means in window).
    """
    oi = float(ctx.info.get("spring_oi", np.nan))
    if not np.isfinite(oi) or oi <= 0.0:
        return 0.0  # outside SPR window

    animas = _animas_farmington_cfs(ctx)
    sanjuan = float(ctx.info.get("release_sj_main_cfs", 0.0))
    farm = animas + sanjuan

    thr = 10_000.0

    # Shaping: value in [0,1] as you approach 10k
    # Example: 0 at 6k, ~1 at 10k (tune low_ref)
    low_ref = 6_000.0
    shaped = (farm - low_ref) / (thr - low_ref + 1e-6)
    shaped = float(np.clip(shaped, 0.0, 1.0))

    # Bonus if you actually exceed threshold
    bonus = 1.0 if farm >= thr else 0.0

    # Combine; keep magnitude modest (weights will do the heavy lifting)
    return 0.7 * shaped + 0.3 * bonus
