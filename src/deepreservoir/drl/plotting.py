"""
Plotting utilities for DeepReservoir evaluation and training.

Design goals (theme):
- Modern, slightly darker "scientific" look (non-default Matplotlib feel)
- Consistent role-based colors across *all* plots (Agent vs Historic)
- Subtle grid + softened spines
- Units use parentheses, not brackets: e.g., "Flow (cfs)"
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from deepreservoir.define_env.spring_peak_release_curve import SpringPeakReleaseCurve


# -----------------------------------------------------------------------------
# Theme + palette
# -----------------------------------------------------------------------------

# Prefer modern sans fonts if available; fall back to DejaVu Sans (bundled).
# We vendor fonts under <repo_root>/assets/fonts/**. This avoids system installs.
_PREFERRED_SANS = ["Inter", "Source Sans 3", "IBM Plex Sans", "DejaVu Sans"]
_FONTS_INITIALIZED = False


def _find_repo_root(start: Path) -> Path | None:
    """Walk upward until we find a directory that contains both 'src' and 'assets'."""
    p = start
    for _ in range(10):
        if (p / "src").is_dir() and (p / "assets").is_dir():
            return p
        if p.parent == p:
            break
        p = p.parent
    return None


def _register_repo_fonts(repo_root: Path) -> int:
    """Register all .ttf fonts under assets/fonts into Matplotlib's in-process font manager."""
    font_dir = repo_root / "assets" / "fonts"
    if not font_dir.is_dir():
        return 0
    count = 0
    for ttf in font_dir.rglob("*.ttf"):
        try:
            fm.fontManager.addfont(str(ttf))
            count += 1
        except Exception:
            # Ignore bad/corrupt font files; we will fall back safely.
            pass
    return count


def _available_sans_fonts(preferred: Sequence[str]) -> list[str]:
    """Return preferred fonts that Matplotlib can resolve (without fallback), plus DejaVu Sans."""
    avail: list[str] = []
    for name in preferred:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            avail.append(name)
        except Exception:
            pass
    if "DejaVu Sans" not in avail:
        avail.append("DejaVu Sans")
    return avail


def _init_fonts(preferred: Sequence[str] = _PREFERRED_SANS) -> None:
    """One-time font registration + rcParams update."""
    global _FONTS_INITIALIZED
    if not _FONTS_INITIALIZED:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is not None:
            _register_repo_fonts(repo_root)

        # Clear Matplotlib's internal findfont cache so newly-added fonts are discoverable.
        try:
            fm._findfont_cached.cache_clear()  # type: ignore[attr-defined]
        except Exception:
            pass

        _FONTS_INITIALIZED = True

    # Always (re-)apply preferred order so callers don't have to manually intervene.
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = _available_sans_fonts(preferred)


_init_fonts()

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Figure/axes styling (Nature-ish base)
FIG_FACE = "#FFFFFF"
AX_FACE = "#FFFFFF"  # white (Nature-ish)
SPINE = "#94A3B8"  # slate 400
GRID = "#E2E8F0"  # slate 200
TXT = "#111827"  # gray 900
MUTED = "#6B7280"  # gray 500

# Subtitle placement (axes coords)
SUBTITLE_X = 0.02  # shift right slightly (0=left edge)
SUBTITLE_Y = 0.98

mpl.rcParams["figure.facecolor"] = FIG_FACE
mpl.rcParams["axes.facecolor"] = AX_FACE
mpl.rcParams["axes.edgecolor"] = SPINE
mpl.rcParams["axes.labelcolor"] = TXT
mpl.rcParams["xtick.color"] = TXT
mpl.rcParams["ytick.color"] = TXT
mpl.rcParams["text.color"] = TXT
mpl.rcParams["axes.titlecolor"] = TXT

mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = GRID
mpl.rcParams["grid.linestyle"] = ":"
mpl.rcParams["grid.linewidth"] = 0.7
mpl.rcParams["grid.alpha"] = 0.65

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.titleweight"] = "regular"
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10

# Nicer dotted/dashed rendering in PNG/PDF
mpl.rcParams["lines.dash_capstyle"] = "round"
mpl.rcParams["lines.solid_capstyle"] = "round"
mpl.rcParams["lines.dash_joinstyle"] = "round"

# Role-based colors (keep these consistent across plots!)
# Goal: agent stands out vs gray historic, without looking cartoony.
AGENT_COLOR = "#2B6CB0"  # muted denim
AGENT_FILL = "#BBD7F0"  # pale denim fill

HIST_COLOR = "#8B95A5"  # lighter slate (baseline)
HIST_FILL = "#E5E7EB"  # pale slate fill

# Historic should be visually distinct from Agent via *style* as well as color.
# Use a dotted line (not chunky dashes).
HIST_LINESTYLE = (0, (1.0, 1.0))
# HIST_LINESTYLE = ':'

# Asymmetric opacity (but keep both fairly opaque)
ALPHA_AGENT_LINE = 0.98
ALPHA_HIST_LINE = 0.90
ALPHA_TRACE_AGENT = 0.14
ALPHA_TRACE_HIST = 0.12

INFLOW_COLOR = "#0F766E"  # teal 700
EVAP_COLOR = "#B45309"  # amber 700
THRESH_COLOR = "#C2410C"  # orange 700
TOTAL_COLOR = "#111827"  # gray 900
ELEV_COLOR = "#A16207"  # amber 800-ish (for elevation axis)

BAND_COLOR = "#94A3B8"  # slate 400 (deadpool/max lines)

# Objective colors (training reward components) – modern, distinct, not default cycle
OBJECTIVE_COLOR_MAP = {
    "dam_safety": "#7C3AED",  # violet 600
    "esa_min_flow": "#0EA5E9",  # sky 500
    "flooding": "#F97316",  # orange 500
    "niip": "#10B981",  # emerald 500
    "physics": "#EF4444",  # red 500
}
FALLBACK_COLORS = [
    "#14B8A6",  # teal 500
    "#F59E0B",  # amber 500
    "#6366F1",  # indigo 500
    "#EC4899",  # pink 500
    "#22C55E",  # green 500
]

# Visual hierarchy
LW_PRIMARY = 1.5
LW_SECONDARY = 1.25
LW_TERTIARY = 1.0

ALPHA_RANGE = 0.06
ALPHA_IQR = 0.18
ALPHA_TRACE = 0.10
ALPHA_AREA = 0.16


# -----------------------------------------------------------------------------
# Public driver
# -----------------------------------------------------------------------------

def save_plots(
    *,
    df_test: pd.DataFrame,
    outdir: Path | str,
    df_train_updates: pd.DataFrame | None = None,
    which: str | Sequence[str] | None = "all",
    plot_kwargs: Mapping[str, Mapping[str, object]] | None = None,
    dpi: int = 300,
) -> dict[str, Path]:
    """
    Generate and save one or more standard plots.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Ensure vendored fonts are registered and preferred font order is active.
    _init_fonts()
    if plot_kwargs is None:
        plot_kwargs = {}

    selected = _resolve_plot_keys(which)
    saved: dict[str, Path] = {}

    for name in selected:
        spec = PLOT_REGISTRY[name]
        func = spec["func"]  # type: ignore[assignment]
        requires = spec["requires"]  # type: ignore[assignment]
        filename = spec["filename"]  # type: ignore[assignment]

        args: list[object] = []
        if "df_test" in requires:
            args.append(df_test)
        if "df_train_updates" in requires:
            if df_train_updates is None:
                import warnings

                warnings.warn(
                    f"Plot {name!r} requires df_train_updates but none was provided; skipping."
                )
                continue
            args.append(df_train_updates)

        kw = dict(plot_kwargs.get(name, {}))
        res = func(*args, **kw)  # type: ignore[misc]
        fig = res[0]

        path = outdir / filename
        save(fig, path, dpi=dpi)
        saved[name] = path

    return saved


def save(
    fig: plt.Figure,
    path: Path | str,
    dpi: int = 300,
    close: bool = True,
    transparent: bool = False,
) -> None:
    """Generic figure saver."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    if close:
        plt.close(fig)


# -----------------------------------------------------------------------------
# Styling helpers
# -----------------------------------------------------------------------------

def _apply_axes_style(ax: plt.Axes, *, time_axis: bool = False) -> None:
    """Apply consistent axes styling."""
    ax.set_facecolor(AX_FACE)

    # Grid (major only by default)
    ax.grid(True, which="major")
    ax.grid(False, which="minor")

    # Spines: keep left/bottom, hide top/right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Ticks
    ax.tick_params(axis="both", which="major", length=5, width=1.0, color=SPINE)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8, color=SPINE)

    if time_axis:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

        # Ensure annual tick marks are present on multi-year time axes.
        # We use *minor* ticks so labels can remain sparse (ConciseDateFormatter on majors).
        ax.xaxis.set_minor_locator(mdates.YearLocator(base=1))

    # Thousands separators on y by default.
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # If a plot later overrides the formatter to ScalarFormatter, keep it plain.
    yfmt = ax.yaxis.get_major_formatter()
    if hasattr(yfmt, "set_scientific"):
        try:
            yfmt.set_scientific(False)
        except Exception:
            pass
    if hasattr(yfmt, "set_useOffset"):
        try:
            yfmt.set_useOffset(False)
        except Exception:
            pass


def _legend(ax: plt.Axes, *, outside: bool = True, ncol: int = 1) -> None:
    """Consistent legend styling."""
    if outside:
        leg = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            ncol=ncol,
        )
    else:
        leg = ax.legend(loc="best", frameon=True, ncol=ncol)

    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_alpha(0.92)
    frame.set_facecolor("#ffffff")
    frame.set_edgecolor("#E5E7EB")
    frame.set_linewidth(0.8)


def _title(
    ax: plt.Axes,
    title: str,
    subtitle: str | None = None,
    *,
    subtitle_x: float = SUBTITLE_X,
    subtitle_y: float = SUBTITLE_Y,
) -> None:
    """Left-aligned title.

    Subtitles have been intentionally disabled (they were redundant and hard to place
    cleanly across different plot types). We keep the parameters for compatibility.
    """
    ax.set_title(title, loc="left", pad=6)


# -----------------------------------------------------------------------------
# Storage
# -----------------------------------------------------------------------------

def plot_storage_timeseries(
    df: pd.DataFrame,
    *,
    storage_agent_col: str = "storage_agent_af",
    storage_hist_col: str = "storage_hist_af",
    elev_col: str = "elev_agent_ft",
    show_elevation: bool = False,
    figsize: Tuple[float, float] = (10, 4.2),
) -> tuple[plt.Figure, plt.Axes, plt.Axes | None]:
    """
    Agent vs historic storage time series.

    - Plots storage in million acre-feet (MAF) for readability.
    - Elevation axis is optional (show_elevation=False by default).
    """
    # Hard-coded storage band for Navajo (AF)
    S_MIN_AF = 500_000.0  # dead pool / safe min
    S_MAX_AF = 1_731_750.0  # max storage

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")

    x = df.index
    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=True)

    def _to_maf(s: pd.Series) -> pd.Series:
        return s.astype(float) / 1e6

    ax.set_xlabel("Date")
    ax.set_ylabel("Storage (MAF)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))

    handles: list[object] = []
    labels: list[str] = []

    if storage_hist_col in df.columns:
        h = ax.plot(
            x,
            _to_maf(df[storage_hist_col].reindex(x)),
            label="Historic",
            linewidth=LW_PRIMARY,
            color=HIST_COLOR,
            linestyle=HIST_LINESTYLE,
            alpha=ALPHA_HIST_LINE,
            zorder=2,
        )[0]
        handles.append(h)
        labels.append("Historic")

    if storage_agent_col in df.columns:
        h = ax.plot(
            x,
            _to_maf(df[storage_agent_col].reindex(x)),
            label="Agent",
            linewidth=LW_PRIMARY,
            color=AGENT_COLOR,
            alpha=ALPHA_AGENT_LINE,
            zorder=3,
        )[0]
        handles.append(h)
        labels.append("Agent")

    # Operating bounds (MAF)
    h1 = ax.axhline(S_MIN_AF / 1e6, linestyle=(0, (4, 3)), linewidth=1.3, color=TOTAL_COLOR, alpha=0.85, label="Deadpool", zorder=1)
    h2 = ax.axhline(S_MAX_AF / 1e6, linestyle=(0, (4, 3)), linewidth=1.3, color=TOTAL_COLOR, alpha=0.85, label="Max storage", zorder=1)
    handles.extend([h1, h2])
    labels.extend(["Deadpool", "Max storage"])

    ax2: plt.Axes | None = None
    if show_elevation:
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(SPINE)
        ax2.spines["right"].set_linewidth(1.0)
        ax2.tick_params(axis="y", which="major", length=5, width=1.0, color=SPINE)
        ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
        ax2.tick_params(axis="y", labelsize=mpl.rcParams["ytick.labelsize"], colors=ELEV_COLOR)

        if elev_col in df.columns:
            elev = df[elev_col].reindex(x).astype(float)
            if elev.notna().any():
                e_min = float(np.nanmin(elev.values))
                e_max = float(np.nanmax(elev.values))
                pad = max(1.0, 0.03 * (e_max - e_min))
                ax2.set_ylim(e_min - pad, e_max + pad)

                ax2.fill_between(
                    x,
                    ax2.get_ylim()[0],
                    elev.values,
                    color=ELEV_COLOR,
                    alpha=0.06,
                    linewidth=0,
                    zorder=0,
                )
                h = ax2.plot(
                    x,
                    elev.values,
                    color=ELEV_COLOR,
                    linewidth=LW_SECONDARY,
                    linestyle=(0, (4, 3)),
                    alpha=0.9,
                    label="Elevation",
                )[0]
                handles.append(h)
                labels.append("Elevation")

        ax2.set_ylabel("Reservoir elevation (ft)", color=ELEV_COLOR, rotation=270, labelpad=20)

    _title(ax, "Storage", "Test period", subtitle_y=1.02)
    if handles:
        leg = ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )
        frame = leg.get_frame()
        frame.set_alpha(0.92)
        frame.set_facecolor("#ffffff")
        frame.set_edgecolor("#E5E7EB")
        frame.set_linewidth(0.8)

    fig.tight_layout()
    return fig, ax, ax2


def plot_storage_doy(
    df: pd.DataFrame,
    *,
    year_start_month: int = 10,
    year_start_day: int = 1,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Storage climatology by day-of-water-year (DOWY), agent vs historic.

    Default uses water years starting Oct 1, so if your evaluation spans Oct–Sep
    you’ll see all evaluation years (not calendar-year-trimmed).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOWY plot.")

    if "storage_hist_af" not in df.columns or "storage_agent_af" not in df.columns:
        raise ValueError("df must contain 'storage_hist_af' and 'storage_agent_af'.")

    hist_full = _select_full_water_years(df["storage_hist_af"].dropna(), year_start_month, year_start_day)
    agent_full = _select_full_water_years(df["storage_agent_af"].dropna(), year_start_month, year_start_day)

    hist_stats = _dowy_stats(hist_full, year_start_month, year_start_day)
    agent_stats = _dowy_stats(agent_full, year_start_month, year_start_day)

    dowy = hist_stats.index.values

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=False)

    ax.set_xlabel("Day of water year")
    ax.set_ylabel("Storage (MAF)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))

    for col in ["min", "q25", "median", "q75", "max"]:
        hist_stats[col] = hist_stats[col] / 1e6
        agent_stats[col] = agent_stats[col] / 1e6

    ax.fill_between(dowy, hist_stats["q25"], hist_stats["q75"], color=HIST_FILL, alpha=ALPHA_IQR, linewidth=0, label="Historic IQR")
    ax.plot(dowy, hist_stats["median"], color=HIST_COLOR, linestyle=HIST_LINESTYLE, alpha=ALPHA_HIST_LINE, linewidth=LW_PRIMARY, label="Historic median", zorder=3)

    ax.fill_between(dowy, agent_stats["q25"], agent_stats["q75"], color=AGENT_FILL, alpha=ALPHA_IQR, linewidth=0, label="Agent IQR")
    ax.plot(dowy, agent_stats["median"], color=AGENT_COLOR, alpha=ALPHA_AGENT_LINE, linewidth=LW_PRIMARY, label="Agent median", zorder=4)

    _title(ax, "Storage climatology (DOWY)", "Median and interquartile range (complete water years)")
    _legend(ax, outside=True, ncol=1)
    fig.tight_layout()
    return fig, ax


def plot_storage_doy_traces(
    df: pd.DataFrame,
    *,
    year_start_month: int = 10,
    year_start_day: int = 1,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Per-water-year storage trajectories (complete water years), with medians overlaid."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOWY plot.")

    if "storage_hist_af" not in df.columns or "storage_agent_af" not in df.columns:
        raise ValueError("df must contain 'storage_hist_af' and 'storage_agent_af'.")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=False)

    ax.set_xlabel("Day of water year")
    ax.set_ylabel("Storage (MAF)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))

    first_hist = True
    for _, x_dowy, vals in _iter_full_water_year_traces(df["storage_hist_af"], year_start_month, year_start_day):
        ax.plot(
            x_dowy,
            vals / 1e6,
            color=HIST_COLOR,
            alpha=ALPHA_TRACE_HIST,
            linewidth=0.9,
            label="Historic (per year)" if first_hist else "_nolegend_",
        )
        first_hist = False

    first_agent = True
    for _, x_dowy, vals in _iter_full_water_year_traces(df["storage_agent_af"], year_start_month, year_start_day):
        ax.plot(
            x_dowy,
            vals / 1e6,
            color=AGENT_COLOR,
            alpha=ALPHA_TRACE_AGENT,
            linewidth=0.9,
            label="Agent (per year)" if first_agent else "_nolegend_",
        )
        first_agent = False

    hist_stats = _dowy_stats(
        _select_full_water_years(df["storage_hist_af"].dropna(), year_start_month, year_start_day),
        year_start_month,
        year_start_day,
    )
    agent_stats = _dowy_stats(
        _select_full_water_years(df["storage_agent_af"].dropna(), year_start_month, year_start_day),
        year_start_month,
        year_start_day,
    )

    dowy = hist_stats.index.values
    ax.plot(dowy, hist_stats["median"] / 1e6, color=HIST_COLOR, linestyle=HIST_LINESTYLE, alpha=ALPHA_HIST_LINE, linewidth=LW_PRIMARY, label="Historic median", zorder=3)
    ax.plot(dowy, agent_stats["median"] / 1e6, color=AGENT_COLOR, alpha=ALPHA_AGENT_LINE, linewidth=LW_PRIMARY, label="Agent median", zorder=4)

    _title(ax, "Storage trajectories (DOWY)", "Per-year traces with median overlay (complete water years)")
    _legend(ax, outside=True, ncol=1)
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Hydropower
# -----------------------------------------------------------------------------

def plot_hydropower_timeseries(
    df: pd.DataFrame,
    *,
    hydro_agent_col: str = "hydro_agent_mwh",
    hydro_hist_col: str = "hydro_hist_mwh",
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Hydropower generation time series (agent vs historic)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")
    if hydro_agent_col not in df.columns:
        raise ValueError(f"df must contain {hydro_agent_col!r}.")

    x = df.index
    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=True)

    ax.set_xlabel("Date")
    ax.set_ylabel("Hydropower (MWh/day)")

    if hydro_hist_col in df.columns:
        ax.plot(
            x,
            df[hydro_hist_col],
            label="Historic",
            linewidth=LW_PRIMARY,
            color=HIST_COLOR,
            linestyle=HIST_LINESTYLE,
            alpha=ALPHA_HIST_LINE,
            zorder=2,
        )

    ax.plot(
        x,
        df[hydro_agent_col],
        label="Agent",
        linewidth=LW_PRIMARY,
        color=AGENT_COLOR,
        alpha=ALPHA_AGENT_LINE,
        zorder=3,
    )

    _title(ax, "Hydropower generation", "Test period")
    _legend(ax, outside=True)
    fig.tight_layout()
    return fig, ax


def plot_hydropower_doy(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Day-of-year hydropower stats (agent vs historic). (No-leap DOY axis.)"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")
    if "hydro_hist_mwh" not in df.columns or "hydro_agent_mwh" not in df.columns:
        raise ValueError("df must contain 'hydro_hist_mwh' and 'hydro_agent_mwh'.")

    hist_stats = _doy_stats(df["hydro_hist_mwh"].dropna())
    agent_stats = _doy_stats(df["hydro_agent_mwh"].dropna())
    doy = hist_stats.index.values

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=False)

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hydropower (MWh/day)")

    ax.fill_between(doy, hist_stats["q25"], hist_stats["q75"], color=HIST_FILL, alpha=ALPHA_IQR, linewidth=0, label="Historic IQR")
    ax.plot(doy, hist_stats["median"], color=HIST_COLOR, linestyle=HIST_LINESTYLE, alpha=ALPHA_HIST_LINE, linewidth=LW_PRIMARY, label="Historic median", zorder=3)

    ax.fill_between(doy, agent_stats["q25"], agent_stats["q75"], color=AGENT_FILL, alpha=ALPHA_IQR, linewidth=0, label="Agent IQR")
    ax.plot(doy, agent_stats["median"], color=AGENT_COLOR, alpha=ALPHA_AGENT_LINE, linewidth=LW_PRIMARY, label="Agent median", zorder=4)

    _title(ax, "Hydropower climatology (DOY)", "Median and interquartile range")
    _legend(ax, outside=True)
    fig.tight_layout()
    return fig, ax


def plot_hydropower_doy_traces(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Per-year DOY hydropower trajectories with medians overlaid (no-leap DOY axis)."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for DOY plot.")
    if "hydro_hist_mwh" not in df.columns or "hydro_agent_mwh" not in df.columns:
        raise ValueError("df must contain 'hydro_hist_mwh' and 'hydro_agent_mwh'.")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=False)

    ax.set_xlabel("Day of year")
    ax.set_ylabel("Hydropower (MWh/day)")

    first_hist = True
    for _, x_doy, vals in _iter_full_year_traces(df["hydro_hist_mwh"]):
        ax.plot(
            x_doy,
            vals,
            color=HIST_COLOR,
            linestyle=HIST_LINESTYLE,
            alpha=ALPHA_TRACE_HIST,
            linewidth=0.9,
            label="Historic (per year)" if first_hist else "_nolegend_",
        )
        first_hist = False

    first_agent = True
    for _, x_doy, vals in _iter_full_year_traces(df["hydro_agent_mwh"]):
        ax.plot(
            x_doy,
            vals,
            color=AGENT_COLOR,
            alpha=ALPHA_TRACE_AGENT,
            linewidth=0.9,
            label="Agent (per year)" if first_agent else "_nolegend_",
        )
        first_agent = False

    hist_stats = _doy_stats(df["hydro_hist_mwh"].dropna())
    agent_stats = _doy_stats(df["hydro_agent_mwh"].dropna())
    doy = hist_stats.index.values
    ax.plot(doy, hist_stats["median"], color=HIST_COLOR, linestyle=HIST_LINESTYLE, alpha=ALPHA_HIST_LINE, linewidth=LW_PRIMARY, label="Historic median", zorder=3)
    ax.plot(doy, agent_stats["median"], color=AGENT_COLOR, alpha=ALPHA_AGENT_LINE, linewidth=LW_PRIMARY, label="Agent median", zorder=4)

    _title(ax, "Hydropower trajectories (DOY)", "Per-year traces with median overlay")
    _legend(ax, outside=True)
    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Releases + forcings
# -----------------------------------------------------------------------------

def plot_release_timeseries(
    df: pd.DataFrame,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Mass balance view.

    - Releases + evaporation on primary axis (left)
    - Inflow as *shaded area* on a secondary axis (right) with 0 at the TOP
      (so the inflow "hangs down" and doesn't visually compete with releases)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")

    x = df.index
    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=True)

    ax.set_xlabel("Date")
    ax.set_ylabel("Flow (cfs)")

    # Secondary axis for inflow shading (0 at top)
    ax_in = ax.twinx()
    ax_in.set_facecolor("none")
    ax_in.grid(False)
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_color(SPINE)
    ax_in.spines["right"].set_linewidth(1.0)
    ax_in.tick_params(axis="y", which="major", length=5, width=1.0, color=SPINE, labelcolor=MUTED)
    ax_in.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax_in.set_ylabel("Inflow (cfs)", color=MUTED, rotation=270, labelpad=18)

    inflow_patch = None
    if "inflow_cfs" in df.columns:
        inflow = df["inflow_cfs"].astype(float).reindex(x)
        max_in = float(np.nanmax(inflow.values)) if np.isfinite(np.nanmax(inflow.values)) else 0.0
        max_in = max(max_in, 1.0)
        # 0 at the top
        ax_in.set_ylim(max_in * 1.05, 0.0)

        ax_in.fill_between(
            x,
            0.0,
            inflow.values,
            color=INFLOW_COLOR,
            alpha=ALPHA_AREA,
            linewidth=0,
            zorder=0,
        )
        inflow_patch = Patch(facecolor=INFLOW_COLOR, edgecolor="none", alpha=ALPHA_AREA, label="Inflow")

    # Historic release (dotted)
    if "release_cfs" in df.columns:
        ax.plot(
            x,
            df["release_cfs"],
            label="Historic release",
            linewidth=LW_PRIMARY,
            color=HIST_COLOR,
            linestyle=HIST_LINESTYLE,
            alpha=ALPHA_HIST_LINE,
            zorder=2,
        )

    # Agent release (solid, on top)
    if "release_agent_cfs" in df.columns:
        ax.plot(
            x,
            df["release_agent_cfs"],
            label="Agent release",
            linewidth=LW_SECONDARY,
            color=AGENT_COLOR,
            alpha=ALPHA_AGENT_LINE,
            zorder=3,
        )

    # Evaporation (kept on primary axis; dashed)
    if "evap_cfs" in df.columns:
        ax.plot(
            x,
            df["evap_cfs"],
            label="Evaporation (eq. cfs)",
            linewidth=LW_SECONDARY,
            color=EVAP_COLOR,
            linestyle=(0, (4, 3)),
            alpha=0.92,
            zorder=2,
        )

    _title(ax, "Mass balance", "Test period")

    # Legend (include inflow patch if present)
    handles, labels = ax.get_legend_handles_labels()
    if inflow_patch is not None:
        handles = [inflow_patch] + handles
        labels = ["Inflow"] + labels

    leg = ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )
    frame = leg.get_frame()
    frame.set_alpha(0.92)
    frame.set_facecolor("#ffffff")
    frame.set_edgecolor("#E5E7EB")
    frame.set_linewidth(0.8)

    fig.tight_layout()
    return fig, ax

# -----------------------------------------------------------------------------
# SPR plots
# -----------------------------------------------------------------------------

_SPRING_PEAK_CURVE = SpringPeakReleaseCurve()


def plot_spr_farmington_components_and_demand_timeseries(
    df: pd.DataFrame,
    *,
    threshold_cfs: float = 10_000.0,
    figsize: tuple[float, float] = (10, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Farmington discharge components + SPR demand curve.

    Curves:
      - Total Farmington = San Juan(agent) + Animas(gauge)
      - San Juan(agent)
      - Animas(gauge)
      - SPR demand curve (repeated yearly)
      - Threshold
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex for time series plots.")
    required = ["release_sj_main_cfs", "animas_farmington_q_cfs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    x = df.index
    sanjuan = df["release_sj_main_cfs"].astype(float)
    animas = df["animas_farmington_q_cfs"].astype(float)
    farmington = sanjuan + animas
    exceed = farmington >= float(threshold_cfs)

    spr_demand = pd.Series(
        [_SPRING_PEAK_CURVE.target_cfs_from_date(pd.to_datetime(d)) for d in x],
        index=x,
        dtype=float,
        name="spr_demand_cfs",
    )

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=True)

    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge at Farmington (cfs)")

    ax.plot(
        x,
        farmington.values,
        linewidth=LW_PRIMARY,
        color=TOTAL_COLOR,
        label="Total (San Juan + Animas)",
        zorder=4,
    )
    ax.plot(
        x,
        sanjuan.values,
        linewidth=LW_SECONDARY,
        color=AGENT_COLOR,
        alpha=0.9,
        label="San Juan (agent)",
        zorder=3,
    )
    ax.plot(
        x,
        animas.values,
        linewidth=LW_SECONDARY,
        color=INFLOW_COLOR,
        alpha=0.85,
        label="Animas (gauge)",
        zorder=2,
    )
    ax.plot(
        x,
        spr_demand.values,
        linewidth=LW_SECONDARY,
        linestyle=(0, (4, 3)),
        color=MUTED,
        alpha=0.9,
        label="SPR demand",
        zorder=1,
    )

    ax.axhline(
        float(threshold_cfs),
        linestyle=(0, (4, 3)),
        linewidth=LW_TERTIARY + 0.4,
        color=THRESH_COLOR,
        alpha=0.9,
        label=f"Threshold ({float(threshold_cfs):,.0f} cfs)",
        zorder=1,
    )

    if exceed.any():
        ax.scatter(
            x[exceed],
            farmington.loc[exceed].values,
            s=26,
            color=THRESH_COLOR,
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label=f"Exceedance (n={int(exceed.sum())})",
        )

    _title(ax, "SPR: Components and demand curve", "Farmington discharge", subtitle_y=1.02)
    _legend(ax, outside=True)
    fig.tight_layout()
    return fig, ax

def plot_niip_doy_percentiles_with_demand(
    df: pd.DataFrame,
    *,
    sanjuan_col: str = "sanjuan_release_cfs",
    niip_col: str = "niip_release_cfs",
    doy_start: int = 50,
    doy_end: int = 300,
    figsize: tuple[float, float] = (10, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    NIIP plot (Colab-style DOY percentiles) with NIIP demand curve:

      - San Juan release (agent component #1): median + IQR by DOY
      - NIIP release (agent component #2):     median + IQR by DOY
      - NIIP demand curve (deterministic):     line by DOY

    All units are cfs.

    Required:
      - df.index is DatetimeIndex
      - df contains columns: sanjuan_col and niip_col
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    missing = [c for c in [sanjuan_col, niip_col] if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing required columns for NIIP plot: {missing}")

    # Compute DOY and restrict to NIIP season
    doy = df.index.dayofyear.astype(int)
    mask = (doy >= int(doy_start)) & (doy <= int(doy_end))
    if not mask.any():
        raise ValueError(f"No rows fall within DOY window {doy_start}-{doy_end}.")

    df_season = df.loc[mask]
    doy_season = doy[mask]

    # Tidy frame for groupby quantiles
    tidy = pd.DataFrame({
        "DOY": doy_season.values,
        "SanJuan": df_season[sanjuan_col].astype(float).values,
        "NIIP": df_season[niip_col].astype(float).values,
    })

    def _doy_stats(values: pd.Series) -> pd.DataFrame:
        g = values.groupby(tidy["DOY"])
        return pd.DataFrame({
            "median": g.median(),
            "q25": g.quantile(0.25),
            "q75": g.quantile(0.75),
        })

    sj_stats = _doy_stats(pd.Series(tidy["SanJuan"]))
    niip_stats = _doy_stats(pd.Series(tidy["NIIP"]))

    # Demand curve: deterministic by DOY (so "repeats every year" naturally)
    doys = sj_stats.index.values
    demand = np.array([float(niip_daily_demand(int(d))) for d in doys], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)

    # Style consistent with your file
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    for spine in ("top",):
        ax.spines[spine].set_visible(False)

    ax.tick_params(labelsize=11)
    ax.set_xlabel("Day of year", fontsize=13)
    ax.set_ylabel("Flow [cfs]", fontsize=13)

    # San Juan: median + IQR
    ax.plot(
        sj_stats.index.values,
        sj_stats["median"].values,
        linewidth=1.8,
        label="San Juan release (median)",
    )
    ax.fill_between(
        sj_stats.index.values,
        sj_stats["q25"].values,
        sj_stats["q75"].values,
        alpha=0.25,
        label="San Juan IQR",
    )

    # NIIP: median + IQR
    ax.plot(
        niip_stats.index.values,
        niip_stats["median"].values,
        linewidth=1.8,
        linestyle="--",
        label="NIIP release (median)",
    )
    ax.fill_between(
        niip_stats.index.values,
        niip_stats["q25"].values,
        niip_stats["q75"].values,
        alpha=0.20,
        label="NIIP IQR",
    )

    # Demand curve (line only)
    ax.plot(
        doys,
        demand,
        linewidth=1.8,
        linestyle=":",
        label="NIIP demand curve",
    )

    ax.set_title(f"NIIP seasonal distribution by DOY (DOY {doy_start}–{doy_end})", fontsize=14)

    leg = ax.legend(loc="best", frameon=True)
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_facecolor("white")

    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# Training rewards
# -----------------------------------------------------------------------------

def plot_train_update_mean_rewards(
    df_train_updates: pd.DataFrame,
    *,
    x_axis: str = "timesteps",
    figsize: tuple[float, float] = (9, 4.0),
) -> tuple[plt.Figure, plt.Axes]:
    """Per-update mean reward for each objective (line plot)."""
    if x_axis in df_train_updates.columns:
        x = df_train_updates[x_axis].values
        xlabel = "Timesteps" if x_axis == "timesteps" else x_axis
    elif "update_idx" in df_train_updates.columns:
        x = df_train_updates["update_idx"].values
        xlabel = "Update"
    else:
        x = df_train_updates.index.values
        xlabel = "Update"

    mean_cols = [c for c in df_train_updates.columns if c.startswith("mean_") and c != "mean_total_reward"]
    if not mean_cols and "mean_total_reward" not in df_train_updates.columns:
        raise ValueError("No 'mean_*' columns found in df_train_updates.")

    fig, ax = plt.subplots(figsize=figsize)
    _apply_axes_style(ax, time_axis=False)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Mean reward per step")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.3f}"))
    
    def _color_for_objective(obj_key: str, idx: int) -> str:
        base = obj_key.split(".", 1)[0]
        if base in OBJECTIVE_COLOR_MAP:
            return OBJECTIVE_COLOR_MAP[base]
        return FALLBACK_COLORS[idx % len(FALLBACK_COLORS)]

    for i, col in enumerate(mean_cols):
        full_key = col[len("mean_") :]
        ax.plot(
            x,
            df_train_updates[col].values,
            linewidth=LW_TERTIARY,
            color=_color_for_objective(full_key, i),
            alpha=0.95,
            label=full_key,
        )

    if "mean_total_reward" in df_train_updates.columns:
        total_mean = df_train_updates["mean_total_reward"].values
    elif mean_cols:
        total_mean = df_train_updates[mean_cols].sum(axis=1).values
    else:
        total_mean = df_train_updates.index.to_numpy(dtype=float) * 0.0

    ax.plot(
        x,
        total_mean,
        linewidth=LW_SECONDARY,
        color=TOTAL_COLOR,
        label="total",
        zorder=5,
    )

    _title(ax, "Training: mean reward by objective", "Per-policy update")
    _legend(ax, outside=True, ncol=1)
    fig.tight_layout()
    return fig, ax


def plot_train_update_reward_heatmap(
    df_train_updates: pd.DataFrame,
    *,
    x_axis: str = "timesteps",
    figsize: tuple[float, float] = (9, 4.6),
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of per-update mean reward components."""
    if x_axis in df_train_updates.columns:
        x = df_train_updates[x_axis].values
        xlabel = "Timesteps" if x_axis == "timesteps" else x_axis
    elif "update_idx" in df_train_updates.columns:
        x = df_train_updates["update_idx"].values
        xlabel = "Update"
    else:
        x = df_train_updates.index.values
        xlabel = "Update"

    mean_cols = [c for c in df_train_updates.columns if c.startswith("mean_") and c != "mean_total_reward"]
    if not mean_cols:
        raise ValueError("No 'mean_*' objective columns found for heatmap.")

    labels = [c[len("mean_") :] for c in mean_cols]
    M = df_train_updates[mean_cols].to_numpy(dtype=float).T

    v = np.nanmax(np.abs(M))
    v = float(v) if np.isfinite(v) and v > 0 else 1.0
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(AX_FACE)
    ax.grid(False)

    # Map image x-axis to actual timesteps/updates rather than 0..N columns
    if len(x) >= 2 and np.all(np.isfinite(x)):
        x0, x1 = float(x[0]), float(x[-1])
        extent = (x0, x1, len(labels) - 0.5, -0.5)
        im = ax.imshow(
            M,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm_r",  # inverted: positive -> blue
            norm=norm,
            extent=extent,
        )
        ax.set_xlim(x0, x1)
    else:
        im = ax.imshow(
            M,
            aspect="auto",
            interpolation="nearest",
            cmap="coolwarm_r",  # inverted: positive -> blue
            norm=norm,
        )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title("Training: reward components (heatmap)", loc="left", pad=6)

    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)

    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # Ensure 0 appears as a labeled tick when the axis spans 0.
    try:
        ticks = [t for t in ax.get_xticks() if np.isfinite(t)]
        if len(x) >= 2 and np.all(np.isfinite(x)):
            x0, x1 = float(x[0]), float(x[-1])
            if x0 <= 0.0 <= x1 and 0.0 not in ticks:
                ticks.append(0.0)
            ticks = [t for t in ticks if min(x0, x1) - 1e-9 <= t <= max(x0, x1) + 1e-9]
        ticks = sorted(set(float(t) for t in ticks))
        if ticks:
            ax.set_xticks(ticks)
    except Exception:
        pass
    ax.tick_params(axis="both", which="major", length=4, width=1.0, color=SPINE)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.outline.set_edgecolor("#E5E7EB")
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(color=SPINE)

    fig.tight_layout()
    return fig, ax


# -----------------------------------------------------------------------------
# DOY utilities
# -----------------------------------------------------------------------------

def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def _doy_noleap(dti: pd.DatetimeIndex) -> np.ndarray:
    """
    Day-of-year on a 365-day 'no-leap' calendar.

    - Feb 29 is excluded.
    - For leap years, days after Feb 28 are shifted down by 1 so March 1 is always day 60.

    Notes:
    - Pandas attribute accessors like `.is_leap_year` can return either an Index-like
      object or a raw ndarray depending on pandas version. Use `np.asarray(...)`
      to keep this robust across environments.
    """
    d = pd.DatetimeIndex(dti).normalize()
    doy = np.asarray(d.dayofyear, dtype=int)
    is_leap = np.asarray(d.is_leap_year, dtype=bool)
    after_feb = (np.asarray(d.month, dtype=int) > 2)
    doy = doy.copy()
    doy[is_leap & after_feb] -= 1
    return doy


def _select_full_years(series: pd.Series) -> pd.Series:
    """Keep only complete calendar years (Jan 1–Dec 31) with no missing days."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    s = series.dropna().copy()
    s.index = s.index.normalize()

    years = np.unique(s.index.year)
    mask_keep = np.zeros(len(s), dtype=bool)

    for y in years:
        mask_y = (s.index.year == y)
        s_y = s[mask_y]
        if s_y.empty:
            continue
        first = s_y.index.min()
        last = s_y.index.max()
        expected_days = (last - first).days + 1
        if first == pd.Timestamp(y, 1, 1) and last == pd.Timestamp(y, 12, 31) and len(s_y) == expected_days:
            mask_keep |= mask_y

    return s[mask_keep]


def _doy_stats(series: pd.Series) -> pd.DataFrame:
    """Compute DOY stats on a 365-day no-leap axis (avoids leap-year discontinuities)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    mask = ~((series.index.month == 2) & (series.index.day == 29))
    s = series[mask]
    doy = _doy_noleap(s.index)
    grouped = s.groupby(doy)
    return pd.DataFrame({
        "min": grouped.min(),
        "q25": grouped.quantile(0.25),
        "median": grouped.median(),
        "q75": grouped.quantile(0.75),
        "max": grouped.max(),
    })


def _iter_full_year_traces(series: pd.Series):
    """Yield (year, x_doy, values) for each complete calendar year (no-leap DOY axis)."""
    full = _select_full_years(series.dropna())
    if full.empty:
        return

    for y in sorted(np.unique(full.index.year)):
        s_y = full[full.index.year == y]
        if s_y.empty:
            continue
        mask = ~((s_y.index.month == 2) & (s_y.index.day == 29))
        s_y = s_y[mask]
        x = _doy_noleap(s_y.index)
        yield y, x, s_y.values


def _water_year(dti: pd.DatetimeIndex, year_start_month: int = 10, year_start_day: int = 1) -> np.ndarray:
    """Compute water-year labels for an arbitrary start month/day."""
    d = pd.DatetimeIndex(dti).normalize()
    y = np.asarray(d.year, dtype=int)
    m = np.asarray(d.month, dtype=int)
    day = np.asarray(d.day, dtype=int)
    starts_after = (m > year_start_month) | ((m == year_start_month) & (day >= year_start_day))
    return y + starts_after.astype(int)


def _dowy_noleap(dti: pd.DatetimeIndex, year_start_month: int = 10, year_start_day: int = 1) -> np.ndarray:
    """
    Day-of-water-year on a 365-day 'no-leap' axis.

    - Feb 29 is excluded by upstream filters (and safe here).
    - For leap years *in the water-year end year*, days after Feb 28 are shifted down by 1.
    """
    # Coerce to DatetimeIndex and normalize to midnight to avoid timezone/time-of-day issues.
    d = pd.DatetimeIndex(dti).normalize()

    wy = _water_year(d, year_start_month, year_start_day)
    start_year = wy - 1

    # IMPORTANT: ensure starts is a DatetimeIndex (not a Series), otherwise broadcasting/subtraction
    # can yield pandas objects with immutable Index semantics.
    starts = pd.DatetimeIndex(
        pd.to_datetime(
            {
                "year": start_year,
                "month": np.full_like(start_year, year_start_month),
                "day": np.full_like(start_year, year_start_day),
            }
        )
    )

    # Compute DOWY as a *mutable* numpy array (days since water-year start + 1).
    # Using .values avoids pandas Index objects and makes mutation safe.
    dowy = ((d.values - starts.values) / np.timedelta64(1, "D")).astype(int) + 1

    # No-leap adjustment: for leap years, shift days after Feb 28 in the *end year* down by 1.
    wy_is_leap = np.array([_is_leap_year(int(x)) for x in wy], dtype=bool)
    in_end_year = (np.asarray(d.year, dtype=int) == wy)
    after_feb = (np.asarray(d.month, dtype=int) > 2)

    mask = wy_is_leap & in_end_year & after_feb
    if mask.any():
        dowy = dowy.copy()
        dowy[mask] -= 1

    return dowy


def _select_full_water_years(
    series: pd.Series,
    year_start_month: int = 10,
    year_start_day: int = 1,
) -> pd.Series:
    """Keep only complete water years (no-leap; always 365 days after dropping Feb 29)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    s = series.dropna().copy()
    s.index = s.index.normalize()
    s = s[~((s.index.month == 2) & (s.index.day == 29))]

    idx = s.index
    wy = _water_year(idx, year_start_month, year_start_day)
    unique_wy = np.unique(wy)
    mask_keep = np.zeros(len(s), dtype=bool)

    for this_wy in unique_wy:
        mask_y = (wy == this_wy)
        s_y = s[mask_y]
        if s_y.empty:
            continue

        start = pd.Timestamp(int(this_wy) - 1, year_start_month, year_start_day)
        next_start = pd.Timestamp(int(this_wy), year_start_month, year_start_day)
        end = next_start - pd.Timedelta(days=1)

        if s_y.index.min() == start and s_y.index.max() == end and len(s_y) == 365:
            mask_keep |= mask_y

    return s[mask_keep]


def _dowy_stats(series: pd.Series, year_start_month: int = 10, year_start_day: int = 1) -> pd.DataFrame:
    """Compute DOWY stats (no-leap axis)."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("Series index must be a DatetimeIndex.")

    s = series.copy()
    s = s[~((s.index.month == 2) & (s.index.day == 29))]
    dowy = _dowy_noleap(s.index, year_start_month, year_start_day)
    grouped = s.groupby(dowy)
    return pd.DataFrame({
        "min": grouped.min(),
        "q25": grouped.quantile(0.25),
        "median": grouped.median(),
        "q75": grouped.quantile(0.75),
        "max": grouped.max(),
    })


def _iter_full_water_year_traces(series: pd.Series, year_start_month: int = 10, year_start_day: int = 1):
    """Yield (water_year, x_dowy, values) for each complete water year (no-leap axis)."""
    full = _select_full_water_years(series.dropna(), year_start_month, year_start_day)
    if full.empty:
        return

    idx = full.index.normalize()
    wy = _water_year(idx, year_start_month, year_start_day)

    for this_wy in sorted(np.unique(wy)):
        s_y = full[wy == this_wy]
        if s_y.empty:
            continue
        x = _dowy_noleap(s_y.index, year_start_month, year_start_day)
        yield int(this_wy), x, s_y.values


# -----------------------------------------------------------------------------
# Plot registry + groups
# -----------------------------------------------------------------------------

PLOT_REGISTRY: dict[str, Mapping[str, object]] = {
    "storage_timeseries": {
        "func": plot_storage_timeseries,
        "requires": ("df_test",),
        "filename": "storage_timeseries.png",
    },
    "release_timeseries": {
        "func": plot_release_timeseries,
        "requires": ("df_test",),
        "filename": "release_timeseries.png",
    },
    "hydropower_timeseries": {
        "func": plot_hydropower_timeseries,
        "requires": ("df_test",),
        "filename": "hydropower_timeseries.png",
    },
    "storage_doy": {
        "func": plot_storage_doy,
        "requires": ("df_test",),
        "filename": "storage_doy.png",
    },
    "storage_doy_traces": {
        "func": plot_storage_doy_traces,
        "requires": ("df_test",),
        "filename": "storage_doy_traces.png",
    },
    "hydropower_doy": {
        "func": plot_hydropower_doy,
        "requires": ("df_test",),
        "filename": "hydropower_doy.png",
    },
    "hydropower_doy_traces": {
        "func": plot_hydropower_doy_traces,
        "requires": ("df_test",),
        "filename": "hydropower_doy_traces.png",
    },
    # NOTE: spr_farmington_10k_timeseries intentionally not exported per request.
    "spr_farmington_components_and_demand_timeseries": {
        "func": plot_spr_farmington_components_and_demand_timeseries,
        "requires": ("df_test",),
        "filename": "spr_farmington_components_and_demand_timeseries.png",
    },
    "train_update_mean_rewards": {
        "func": plot_train_update_mean_rewards,
        "requires": ("df_train_updates",),
        "filename": "train_update_mean_rewards.png",
    },
    "train_update_reward_heatmap": {
        "func": plot_train_update_reward_heatmap,
        "requires": ("df_train_updates",),
        "filename": "train_update_reward_heatmap.png",
    },
    "niip_doy_percentiles_with_demand": {
       "func": plot_niip_doy_percentiles_with_demand,
       "requires": ("df_test",),
       "filename": "niip_doy_percentiles_with_demand.png",
    },
}

PLOT_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "storage_timeseries",
        "release_timeseries",
        "hydropower_timeseries",
        "train_update_mean_rewards",
        "train_update_reward_heatmap",
    ),
    "storage": (
        "storage_timeseries",
        "storage_doy",
        "storage_doy_traces",
    ),
    "hydropower": (
        "hydropower_timeseries",
        "hydropower_doy",
        "hydropower_doy_traces",
    ),
    "rewards": (
        "train_update_mean_rewards",
        "train_update_reward_heatmap",
    ),
    "doy": (
        "storage_doy",
        "storage_doy_traces",
        "hydropower_doy",
        "hydropower_doy_traces",
    ),
    "spr": (
        "spr_farmington_components_and_demand_timeseries",
    ),
    "timeseries": (
        "storage_timeseries",
        "release_timeseries",
        "hydropower_timeseries",
        "spr_farmington_components_and_demand_timeseries",
    ),
     "niip": (
        "niip_doy_percentiles_with_demand",
    ),
}


def _resolve_plot_keys(which: str | Sequence[str] | None) -> list[str]:
    """
    Expand 'which' into a concrete list of plot keys.
    """
    if which is None or which == "all":
        return list(PLOT_REGISTRY.keys())

    keys: list[str] = []

    def add_name(name: str) -> None:
        if name in PLOT_GROUPS:
            for k in PLOT_GROUPS[name]:
                if k not in keys:
                    keys.append(k)
        elif name in PLOT_REGISTRY:
            if name not in keys:
                keys.append(name)
        else:
            raise ValueError(f"Unknown plot or group name: {name!r}")

    if isinstance(which, str):
        parts = [p.strip() for p in which.split(",") if p.strip()]
        for p in parts:
            add_name(p)
    else:
        for item in which:
            add_name(str(item))

    return keys
