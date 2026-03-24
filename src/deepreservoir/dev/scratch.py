"""VS Code-friendly scratch runner (cell-by-cell).

This scratch is pre-configured to demonstrate a common workflow:

  1) Train from start-of-record through WY 2010 for 100_000 timesteps
  2) Evaluate from WY 2011 through end-of-record
  3) Resume training for an additional 500_000 timesteps

Window token rules
------------------
- 'YYYY'        -> water year (Oct 1 of YYYY-1 .. Sep 30 of YYYY)
- 'YYYY-MM-DD'  -> exact day

Note on "end of record"
-----------------------
In the python API, you can use ``window_end=None`` to mean "use the last
available date".

Tip: run ``python -m deepreservoir.drl.cli info`` to see the available
(date-clipped) model_data range.
"""

from __future__ import annotations

from pathlib import Path

from deepreservoir.drl.model import DRLModel, evaluate_model_window


# %%
# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------

RUN_DIR = Path("runs/fixed_deadpool_alpha_2_250000")
START_FRESH = True  # set False if you want to manually resume instead

REWARD_SPEC = (
    "dam_safety:storage_fraction@2.0,"
    # "esa_min_flow:baseline,"
    # "flooding:baseline@1.0,"
    # "hydropower:baseline@1.0,"
    # "niip:baseline@1.0,"
)

# --- Training window: start-of-record -> WY 2010 ---
USE_FULL_RECORD = False
N_YEARS_TRAIN: int | None = None
TRAIN_START: str | None = None
TRAIN_END: str | None = "2010"  # WY token

# Optional single exclusion hole (unused in this demo)
EXCLUDE_START: str | None = None
EXCLUDE_END: str | None = None

# Optional periodic validation during training (unused in this demo)
VAL_START: str | None = None
VAL_END: str | None = None
VAL_FREQ: int = 50_000

# --- Standalone evaluation: WY 2011 -> end-of-record ---
EVAL_START: str | None = "2011"
EVAL_END: str | None = None  # None = end-of-record

# Timesteps
TIMESTEPS_TRAIN_1 = 250_000
TIMESTEPS_TRAIN_2 = 500_000

# PPO knobs (safe single-env defaults)
EPISODE_LENGTH_TRAIN = 3600
N_ENVS = 1
N_STEPS = 2048
BATCH_SIZE = 256  # must be <= N_STEPS * N_ENVS
N_EPOCHS = 10
GAMMA = 0.999

SEED: int | None = 123
DEVICE = "cpu"  # or "auto" / "cuda"


# %%
# -----------------------------------------------------------------------------
# 1) Create run dir + instantiate training model
# -----------------------------------------------------------------------------

RUN_DIR.mkdir(parents=True, exist_ok=True)
model_path = RUN_DIR / "last_model.zip"

if START_FRESH and model_path.exists():
    raise RuntimeError(
        f"START_FRESH=True but {model_path} already exists. "
        "Delete the run dir, pick a new RUN_DIR, or set START_FRESH=False."
    )

m = DRLModel(
    reward_spec=REWARD_SPEC,
    use_full_record=USE_FULL_RECORD,
    n_years_train=N_YEARS_TRAIN,
    train_start=TRAIN_START,
    train_end=TRAIN_END,
    exclude_start=EXCLUDE_START,
    exclude_end=EXCLUDE_END,
    val_start=VAL_START,
    val_end=VAL_END,
    logdir=RUN_DIR,
    seed=SEED,
    device=DEVICE,
    n_envs=N_ENVS,
    use_subproc_vec=False,
    episode_length_train=EPISODE_LENGTH_TRAIN,
)

print(f"[scratch] RUN_DIR = {RUN_DIR.resolve()}")
print(f"[scratch] Train window: start-of-record -> {TRAIN_END} (WY)")

# -----------------------------------------------------------------------------
# 2) Train #1 (fresh)
# -----------------------------------------------------------------------------

print(f"[scratch] Training #1 for {TIMESTEPS_TRAIN_1:,} timesteps")

m.train(
    total_timesteps=int(TIMESTEPS_TRAIN_1),
    val_freq=(VAL_FREQ if (VAL_START is not None and VAL_END is not None) else None),
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    n_epochs=N_EPOCHS,
    gamma=GAMMA,
    track_reward_components=True,
    resume=False,
)

print(f"[scratch] Done training #1. Model saved to: {model_path}")


# %%
# -----------------------------------------------------------------------------
# 3) Eval: WY 2011 -> end-of-record
# -----------------------------------------------------------------------------

if EVAL_START is None:
    raise RuntimeError("EVAL_START is None; set it or skip this cell.")

end_tag = (EVAL_END if EVAL_END is not None else "end")
outdir = RUN_DIR / f"eval_{EVAL_START}_{end_tag}"

print(f"[scratch] Evaluating on window: {EVAL_START} -> {end_tag}")

_, df_metrics = evaluate_model_window(
    model_path=model_path,
    reward_spec=REWARD_SPEC,
    window_start=EVAL_START,
    window_end=EVAL_END,
    outdir=outdir,
    device=DEVICE,
    which_metrics="core",
    save_rollout=True,
    save_plots=True,
    save_metrics=True,
)

print("\n[scratch] eval metrics:\n" + df_metrics.to_string(index=False))


# # %%
# # -----------------------------------------------------------------------------
# # 4) Train #2 (resume): +500_000 timesteps
# # -----------------------------------------------------------------------------

# print(f"[scratch] Training #2 (resume) for +{TIMESTEPS_TRAIN_2:,} timesteps")

# # New DRLModel instance to demonstrate the intended resume workflow.
# # (Same config, then load weights+optimizer state.)
# m2 = DRLModel(
#     reward_spec=REWARD_SPEC,
#     use_full_record=USE_FULL_RECORD,
#     n_years_train=N_YEARS_TRAIN,
#     train_start=TRAIN_START,
#     train_end=TRAIN_END,
#     exclude_start=EXCLUDE_START,
#     exclude_end=EXCLUDE_END,
#     val_start=VAL_START,
#     val_end=VAL_END,
#     logdir=RUN_DIR,
#     seed=SEED,
#     device=DEVICE,
#     n_envs=N_ENVS,
#     use_subproc_vec=False,
#     episode_length_train=EPISODE_LENGTH_TRAIN,
# )

# m2.load_model(str(model_path))

# m2.train(
#     total_timesteps=int(TIMESTEPS_TRAIN_2),
#     val_freq=(VAL_FREQ if (VAL_START is not None and VAL_END is not None) else None),
#     n_steps=N_STEPS,
#     batch_size=BATCH_SIZE,
#     n_epochs=N_EPOCHS,
#     gamma=GAMMA,
#     track_reward_components=True,
#     resume=True,
# )

# print(f"[scratch] Done training #2. Updated model saved to: {model_path}")
