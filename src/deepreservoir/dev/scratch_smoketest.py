"""VS Code-friendly smoke test runner for DeepReservoir branch-to-branch timing.

Edit ONLY:
    RUN_LABEL

Then run the file/cells again for each repo state you want to compare.

This script:
- trains a fixed smoke suite (storage, hydropower, niip, esa_min_flow, spr_curve)
- times training and eval separately
- saves rollout / plots / metrics
- writes per-run timing JSON
- appends a summary table to:
    runs/<RUN_LABEL>/smoke_summary.csv
    runs/<RUN_LABEL>/smoke_summary.json

It uses a manual eval path so plotting/metrics still get exercised, while avoiding
fragility around model wrapper eval helpers.
"""

from __future__ import annotations

import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from deepreservoir.drl.model import (
    DRLModel,
    infer_run_dir_from_model_path,
    run_rollout_window,
)
from deepreservoir.drl import metrics as drl_metrics
from deepreservoir.drl import plotting as drl_plotting


# %%
# -----------------------------------------------------------------------------
# CHANGE ONLY THIS
# -----------------------------------------------------------------------------

RUN_LABEL = "compare_me"

# -----------------------------------------------------------------------------
# FIXED SMOKE CONFIG
# -----------------------------------------------------------------------------

RUNS_ROOT = Path("runs") / RUN_LABEL
OVERWRITE_EXISTING = True
TIMESTEPS_TRAIN = 50_000

USE_FULL_RECORD = False
N_YEARS_TRAIN: int | None = None
TRAIN_START: str | None = None
TRAIN_END: str | None = "2010"
EXCLUDE_START: str | None = None
EXCLUDE_END: str | None = None
VAL_START: str | None = None
VAL_END: str | None = None
VAL_FREQ: int = 50_000

EVAL_START: str | None = "2011"
EVAL_END: str | None = None
WHICH_METRICS = "core"

SAVE_ROLLOUT = True
SAVE_PLOTS = True
SAVE_METRICS = True

EPISODE_LENGTH_TRAIN = 3600
N_ENVS = 1
N_STEPS = 2048
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.999
SEED: int | None = 123
DEVICE = "cpu"

SMOKE_CASES: dict[str, str] = {
    "storage": "dam_safety:storage_band_shaped@2.0",
    "hydropower": "hydropower:baseline@1.0",
    "niip": "niip:baseline@1.0",
    "esa_min_flow": "esa_min_flow:baseline",
    "spr_curve": "esa_spring_peak_release:curve@1.0",
}

SUITE_CASES = [
    "storage",
    "hydropower",
    "niip",
    "esa_min_flow",
    "spr_curve",
]


# %%
# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_case_dir(case_name: str) -> Path:
    return RUNS_ROOT / case_name


def _prepare_run_dir(run_dir: Path, overwrite: bool = False) -> None:
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "last_model.zip"
    if model_path.exists() and not overwrite:
        raise RuntimeError(
            f"{model_path} already exists. Set OVERWRITE_EXISTING=True, "
            "delete the run dir, or change RUN_LABEL."
        )


def _summary_paths() -> tuple[Path, Path]:
    return RUNS_ROOT / "smoke_summary.csv", RUNS_ROOT / "smoke_summary.json"


def _append_summary(row: dict) -> pd.DataFrame:
    csv_path, json_path = _summary_paths()
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(csv_path, index=False)
    json_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def _load_train_update_metrics(model_path: Path) -> pd.DataFrame | None:
    try:
        run_dir = infer_run_dir_from_model_path(model_path)
        upd_path = run_dir / "train_update_metrics.parquet"
        if upd_path.exists():
            return pd.read_parquet(upd_path)
    except Exception:
        return None
    return None


def _save_eval_outputs_manual(
    *,
    model_path: Path,
    reward_spec: str,
    outdir: Path,
    device: str,
    which_metrics: str,
    save_rollout: bool,
    save_plots: bool,
    save_metrics: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, str | None]:
    """Manual eval path with graceful plot failure handling."""
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_rollout_window(
        model_path=model_path,
        reward_spec=reward_spec,
        window_start=EVAL_START,
        window_end=EVAL_END,
        device=device,
    )

    if save_rollout:
        stem = "eval_rollout"
        out_path = outdir / f"{stem}.parquet"
        df.to_parquet(out_path)
        df.to_csv(out_path.with_suffix(".csv"), index=True)

    plot_success = True
    plot_error = None

    if save_plots:
        try:
            df_train_updates = _load_train_update_metrics(model_path)
            drl_plotting.save_plots(
                df_test=df,
                outdir=outdir / "plots",
                df_train_updates=df_train_updates,
                which="all",
            )
        except Exception as e:
            plot_success = False
            plot_error = "".join(
                traceback.format_exception_only(type(e), e)
            ).strip()
            (outdir / "plot_error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(f"[warning] Plot export failed, but continuing: {plot_error}")

    metrics_df = drl_metrics.compute_metrics(df, which=which_metrics, validate=True)
    if save_metrics:
        drl_metrics.save_metrics(
            df_test=df,
            outdir=outdir,
            which=which_metrics,
            stem="eval_metrics",
            validate=True,
        )

    return df, metrics_df, plot_success, plot_error


def _print_timing_summary(summary: dict) -> None:
    print("\n" + "=" * 92)
    print(f"case                : {summary['case_name']}")
    print(f"reward_spec         : {summary['reward_spec']}")
    print(f"run_dir             : {summary['run_dir']}")
    print(f"train_seconds       : {summary['train_seconds']:.2f}")
    print(f"train_timesteps/sec : {summary['train_timesteps_per_sec']:.2f}")
    print(f"eval_seconds        : {summary['eval_seconds']:.2f}")
    print(f"total_seconds       : {summary['total_seconds']:.2f}")
    print(f"plot_success        : {summary['plot_success']}")
    if summary.get("plot_error"):
        print(f"plot_error          : {summary['plot_error']}")
    print("=" * 92 + "\n")


def run_smoke_case(
    case_name: str,
    reward_spec: str,
    *,
    overwrite: bool = False,
    save_eval_outputs: bool = True,
) -> dict:
    run_dir = _safe_case_dir(case_name)
    _prepare_run_dir(run_dir, overwrite=overwrite)

    print(f"[{_timestamp()}] Starting smoke case: {case_name}")
    print(f"[{_timestamp()}] RUN_DIR = {run_dir.resolve()}")
    print(f"[{_timestamp()}] reward_spec = {reward_spec}")

    model = DRLModel(
        reward_spec=reward_spec,
        use_full_record=USE_FULL_RECORD,
        n_years_train=N_YEARS_TRAIN,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        exclude_start=EXCLUDE_START,
        exclude_end=EXCLUDE_END,
        val_start=VAL_START,
        val_end=VAL_END,
        logdir=run_dir,
        seed=SEED,
        device=DEVICE,
        n_envs=N_ENVS,
        use_subproc_vec=False,
        episode_length_train=EPISODE_LENGTH_TRAIN,
    )

    t0 = time.perf_counter()
    model.train(
        total_timesteps=int(TIMESTEPS_TRAIN),
        val_freq=(VAL_FREQ if (VAL_START is not None and VAL_END is not None) else None),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        track_reward_components=True,
        resume=False,
    )
    t1 = time.perf_counter()
    train_seconds = float(t1 - t0)
    train_timesteps_per_sec = (
        float(TIMESTEPS_TRAIN / train_seconds) if train_seconds > 0 else float("nan")
    )

    end_tag = EVAL_END if EVAL_END is not None else "end"
    outdir = run_dir / f"eval_{EVAL_START}_{end_tag}"

    t2 = time.perf_counter()
    _, df_metrics, plot_success, plot_error = _save_eval_outputs_manual(
        model_path=run_dir / "last_model.zip",
        reward_spec=reward_spec,
        outdir=outdir,
        device=DEVICE,
        which_metrics=WHICH_METRICS,
        save_rollout=save_eval_outputs and SAVE_ROLLOUT,
        save_plots=save_eval_outputs and SAVE_PLOTS,
        save_metrics=save_eval_outputs and SAVE_METRICS,
    )
    t3 = time.perf_counter()
    eval_seconds = float(t3 - t2)

    metrics_csv = outdir / "eval_metrics.csv"
    plots_dir = outdir / "plots"
    summary = {
        "timestamp": _timestamp(),
        "run_label": RUN_LABEL,
        "case_name": case_name,
        "reward_spec": reward_spec,
        "timesteps_train": int(TIMESTEPS_TRAIN),
        "train_seconds": train_seconds,
        "train_timesteps_per_sec": train_timesteps_per_sec,
        "eval_seconds": eval_seconds,
        "total_seconds": train_seconds + eval_seconds,
        "plot_success": bool(plot_success),
        "plot_error": plot_error,
        "device": DEVICE,
        "seed": SEED,
        "episode_length_train": int(EPISODE_LENGTH_TRAIN),
        "n_envs": int(N_ENVS),
        "n_steps": int(N_STEPS),
        "batch_size": int(BATCH_SIZE),
        "n_epochs": int(N_EPOCHS),
        "gamma": float(GAMMA),
        "train_start": TRAIN_START,
        "train_end": TRAIN_END,
        "eval_start": EVAL_START,
        "eval_end": EVAL_END,
        "run_dir": str(run_dir.resolve()),
        "metrics_csv": str(metrics_csv.resolve()),
        "plots_dir": str(plots_dir.resolve()),
    }

    (run_dir / "smoke_timing_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    df_summary = _append_summary(summary)
    _print_timing_summary(summary)

    print("[metrics]\n" + df_metrics.to_string(index=False))
    print(f"\n[summary csv] {(_summary_paths()[0]).resolve()}")
    print(f"[summary rows] {len(df_summary)}")
    return summary


def run_smoke_suite(
    case_names: Iterable[str] | None = None,
    *,
    overwrite: bool = False,
    save_eval_outputs: bool = True,
) -> pd.DataFrame:
    if case_names is None:
        case_names = SUITE_CASES

    rows: list[dict] = []
    for case_name in case_names:
        reward_spec = SMOKE_CASES[case_name]
        row = run_smoke_case(
            case_name,
            reward_spec,
            overwrite=overwrite,
            save_eval_outputs=save_eval_outputs,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = [
        "case_name",
        "timesteps_train",
        "train_seconds",
        "train_timesteps_per_sec",
        "eval_seconds",
        "total_seconds",
        "plot_success",
        "run_dir",
    ]
    print("\n[suite summary]\n" + df[cols].to_string(index=False))
    return df


# %%
# -----------------------------------------------------------------------------
# QUICK CHECKS
# -----------------------------------------------------------------------------

print(f"RUN_LABEL         = {RUN_LABEL}")
print(f"RUNS_ROOT         = {RUNS_ROOT.resolve()}")
print(f"TIMESTEPS_TRAIN   = {TIMESTEPS_TRAIN:,}")
print(f"DEVICE            = {DEVICE}")
print(f"TRAIN WINDOW      = start-of-record -> {TRAIN_END}")
print(f"EVAL WINDOW       = {EVAL_START} -> {EVAL_END if EVAL_END is not None else 'end'}")
print("\nSmoke cases:")
for k in SUITE_CASES:
    print(f"  - {k:12s} : {SMOKE_CASES[k]}")


# %%
# -----------------------------------------------------------------------------
# RUN FIXED SUITE
# -----------------------------------------------------------------------------

suite_df = run_smoke_suite(
    SUITE_CASES,
    overwrite=OVERWRITE_EXISTING,
    save_eval_outputs=True,
)


# %%
# -----------------------------------------------------------------------------
# LOAD / VIEW EXISTING SUMMARY TABLE
# -----------------------------------------------------------------------------

summary_csv, summary_json = _summary_paths()
if summary_csv.exists():
    df_summary_existing = pd.read_csv(summary_csv)
    display_cols = [
        "timestamp",
        "run_label",
        "case_name",
        "timesteps_train",
        "train_seconds",
        "train_timesteps_per_sec",
        "eval_seconds",
        "total_seconds",
        "plot_success",
    ]
    print(df_summary_existing[display_cols].to_string(index=False))
else:
    print(f"No summary CSV found yet at: {summary_csv.resolve()}")
