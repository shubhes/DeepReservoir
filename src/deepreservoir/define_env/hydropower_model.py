# navajo_model.py
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Union, Sequence
from scipy.interpolate import interp1d
from deepreservoir.data.metadata import project_metadata

m = project_metadata()
path_model_params: Path = m.path("hydropower_eta")

# -------------------------------------------------------------------
# Load single-eta parameter from pickle
# -------------------------------------------------------------------
def _load_eta_from_pickle(pkl_path: Path) -> float:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "eta_eff" in obj:
        return float(obj["eta_eff"])
    raise ValueError("Unsupported parameters.pkl format (expected dict with 'eta_eff').")


_eta_loaded: Optional[float] = None
if path_model_params.exists():
    try:
        _eta_loaded = _load_eta_from_pickle(path_model_params)
    except Exception:
        _eta_loaded = None

# -------------------------------------------------------------------
# Internal tailwater model
# -------------------------------------------------------------------
_TAILWATER_Q_CFS = np.array([0.0, 2000.0, 4000.0, 8000.0, 16000.0, 24000.0, 32000.0, 40000.0], dtype=float)
_TAILWATER_ELEV_FT = np.array([5711.7, 5712.3, 5712.9, 5714.0, 5716.0, 5718.1, 5720.3, 5721.5], dtype=float)


def _create_tailwater_model():
    return interp1d(_TAILWATER_Q_CFS, _TAILWATER_ELEV_FT, kind="linear", fill_value="extrapolate")


_tailwater_model = _create_tailwater_model()


def _tailwater_ft_scalar(q_cfs: float) -> float:
    """Fast scalar tailwater interpolation for the env hot path."""
    q = float(q_cfs)
    return float(np.interp(q, _TAILWATER_Q_CFS, _TAILWATER_ELEV_FT))


# Constants
_RHO = 1000.0
_G = 9.81
_CFS_TO_CMS = 0.0283168
_FT_TO_M = 0.3048
_TURBINE_LIMIT_CFS = 1300.0
_PLANT_CAPACITY_MW = 32.0
_ENERGY_COEFF_BASE = _RHO * _G * _CFS_TO_CMS * _FT_TO_M * 24.0 / 1e6


def _resolve_eta_and_coeff(eta_eff: Optional[float]) -> tuple[float, float]:
    if eta_eff is None:
        if _eta_loaded is None:
            raise RuntimeError("No eta provided and could not load from parameters.pkl.")
        eta = float(_eta_loaded)
    else:
        eta = float(eta_eff)
    return eta, eta * _ENERGY_COEFF_BASE


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def navajo_power_generation_scalar(
    cfs_value: float,
    elevation_ft: float,
    eta_eff: Optional[float] = None,
) -> float:
    """Fast scalar daily energy production (MWh) for the env hot path."""
    _, energy_coeff = _resolve_eta_and_coeff(eta_eff)

    q_cfs = float(cfs_value)
    if q_cfs <= 0.0:
        return 0.0

    tw_ft = _tailwater_ft_scalar(q_cfs)
    head_ft = float(elevation_ft) - tw_ft
    if head_ft <= 0.0:
        return 0.0

    q_cfs_capped = min(q_cfs, _TURBINE_LIMIT_CFS)
    energy_mwh = energy_coeff * q_cfs_capped * head_ft
    plant_capacity_mwh_day = _PLANT_CAPACITY_MW * 24.0
    if energy_mwh > plant_capacity_mwh_day:
        return plant_capacity_mwh_day
    return float(energy_mwh)


def navajo_power_generation_model(
    cfs_values: Union[float, Sequence[float]],
    elevation_ft: Union[float, Sequence[float]],
    eta_eff: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Predict daily energy production (MWh) given releases (cfs) and reservoir
    elevations (feet), using a single global efficiency eta.

    Parameters
    ----------
    cfs_values : float or array-like
        Releases in cubic feet per second.
    elevation_ft : float or array-like
        Reservoir elevation in feet.
    eta_eff : float, optional
        Efficiency to use. If None, loads from parameters.pkl.

    Returns
    -------
    energy_MWh : float or np.ndarray
        Daily energy production in MWh.
    """
    if np.isscalar(cfs_values) and np.isscalar(elevation_ft):
        return navajo_power_generation_scalar(
            cfs_value=float(cfs_values),
            elevation_ft=float(elevation_ft),
            eta_eff=eta_eff,
        )

    eta_eff, _ = _resolve_eta_and_coeff(eta_eff)

    q_cfs = np.asarray(cfs_values, dtype=float)
    elev_ft = np.asarray(elevation_ft, dtype=float)

    # Tailwater and head
    tw_ft = _tailwater_model(q_cfs)
    head_m = (elev_ft - tw_ft) * _FT_TO_M
    head_m = np.clip(head_m, 0, None)

    # Flow to m³/s with turbine limit
    q_cfs_capped = np.clip(q_cfs, 0, _TURBINE_LIMIT_CFS)
    q_cms = q_cfs_capped * _CFS_TO_CMS

    power_MW = eta_eff * _RHO * _G * q_cms * head_m / 1e6
    power_MW = np.minimum(power_MW, _PLANT_CAPACITY_MW)
    energy_MWh = power_MW * 24.0

    # Return scalar if scalar input
    return energy_MWh if energy_MWh.ndim > 0 and energy_MWh.size > 1 else float(energy_MWh)


# import matplotlib.pyplot as plt
# df = pd.read_csv(r"X:\Research\DeepReservoir\Code\DeepReservoir\data\Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv")
# energies = navajo_power_generation_model(df['Total Release (cfs)'], df['Elevation (feet)'])    
# plt.plot(energies)
# plt.show()

# # --- Load & compute daily energy ---
# df = pd.read_csv(r"X:\Research\DeepReservoir\Code\DeepReservoir\data\Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv")
# start_yr, end_yr = 2000, 2010   # <-- adjust as needed

# date_col = "Date"  # update if needed
# df[date_col] = pd.to_datetime(df[date_col])
# df = df.sort_values(date_col)

# energy_MWh = navajo_power_generation_model(df['Total Release (cfs)'], df['Elevation (feet)'])
# s = pd.Series(energy_MWh, index=df[date_col], name="energy_MWh")

# # --- Year filter ---
# mask = (s.index.year >= start_yr) & (s.index.year <= end_yr)
# s = s[mask]

# # --- Remove Feb 29 so DOY aligns to 365 ---
# is_feb29 = (s.index.month == 2) & (s.index.day == 29)
# s = s[~is_feb29]

# # --- Climatology stats ---
# doy = s.index.dayofyear
# grouped = s.groupby(doy)

# mean_vals = grouped.mean()
# q25_vals = grouped.quantile(0.25)
# q75_vals = grouped.quantile(0.75)

# clim = pd.DataFrame({
#     "mean": mean_vals,
#     "q25": q25_vals,
#     "q75": q75_vals
# })

# # --- Plot ---
# plt.figure(figsize=(8, 5))
# plt.plot(clim.index, clim['mean'], linewidth=2, label='Mean')
# plt.fill_between(clim.index, clim['q25'], clim['q75'], alpha=0.2, label='25–75%')
# plt.xlabel('Day of Year')
# plt.ylabel('Energy (MWh/day)')
# plt.title(f'Historical Energy Production, Simulated ({start_yr} - {end_yr})')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()


# # Look at how elevation impacts power generation
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Choose an elevation range ---
# # Option A: use your CSV to bound the realistic elevation range
# df = pd.read_csv(r"X:\Research\DeepReservoir\Code\DeepReservoir\data\Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv")
# emin = float(np.nanmin(df['Elevation (feet)']))
# emax = float(np.nanmax(df['Elevation (feet)']))

# # Option B (fallback): set an explicit range if you prefer
# emin, emax = 6030, 6151

# elev_ft = np.linspace(emin, emax, 300)

# # --- Fix flow at turbine max ---
# q_cfs = 1300.0
# q_arr = np.full_like(elev_ft, q_cfs, dtype=float)

# # --- Run your model (returns MWh/day) and convert to MW ---
# energy_MWh = navajo_power_generation_model(q_arr, elev_ft)
# power_MW = energy_MWh / 24.0

# # --- Plot ---
# plt.figure(figsize=(7.5, 5))
# plt.plot(elev_ft, power_MW, linewidth=2)
# plt.axhline(_PLANT_CAPACITY_MW, linestyle='--', linewidth=1, label=f'Plant capacity ({_PLANT_CAPACITY_MW:.0f} MW)')
# plt.title('Hydropower vs. Elevation at 1300 cfs')
# plt.xlabel('Reservoir Elevation (ft)')
# plt.ylabel('Power (MW)')
# plt.grid(True, alpha=0.3)
# plt.legend()
# plt.tight_layout()
# plt.show()

