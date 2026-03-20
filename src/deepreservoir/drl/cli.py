"""deepreservoir.drl.cli

CLI for DeepReservoir DRL.

Subcommands
-----------
info
  Print the available model_data date range.

train
  Train a model (fresh) or resume training an existing model.
  Training data selection is explicit and uses *window tokens*:
    - 'YYYY'        -> water year (Oct 1 of YYYY-1 .. Sep 30 of YYYY)
    - 'YYYY-MM-DD'  -> exact day

  Training window can be specified as:
    - --use-full-record
    - --n-years-train N   (last N water years)
    - --train-start/--train-end tokens

  Optionally exclude a single "hole" from training via --exclude-start/--exclude-end.
  Episodes are truncated at segment boundaries (no cross-gap trajectories).

  Optionally run *periodic validation during training* (SB3 EvalCallback) via
  --val-start/--val-end/--val-freq.

eval
  Evaluate a saved model over a specified window. Metrics + plots are exported by default.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import os

from deepreservoir.drl import model
from deepreservoir.drl import helpers
from deepreservoir.drl import sweeps


_DEFAULT_RUNS_ROOT = Path(__file__).resolve().parents[3] / "runs"


def _sanitize_token(s: str) -> str:
    return s.replace(":", "-").replace("/", "-").replace("\\", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepReservoir DRL CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------
    p_info = sub.add_parser("info", help="Print available model_data date range")
    p_info.add_argument(
        "--quiet",
        action="store_true",
        help="Only print start/end dates (no extra text).",
    )

    # ------------------------------------------------------------------
    # train
    # ------------------------------------------------------------------
    p_train = sub.add_parser("train", help="Train a DRL agent (fresh) or resume training")

    # training window selection (one of these must be provided)
    p_train.add_argument(
        "--use-full-record",
        action="store_true",
        help="Use the full available dataset range for training (explicit).",
    )
    p_train.add_argument(
        "--n-years-train",
        type=int,
        default=None,
        help="Train on the last N water years of available data.",
    )
    p_train.add_argument(
        "--train-start",
        type=str,
        default=None,
        help="Train window start token ('YYYY' water year or 'YYYY-MM-DD').",
    )
    p_train.add_argument(
        "--train-end",
        type=str,
        default=None,
        help="Train window end token ('YYYY' water year or 'YYYY-MM-DD').",
    )

    # single exclusion hole
    p_train.add_argument(
        "--exclude-start",
        type=str,
        default=None,
        help="Exclude window start token (single hole removed from training).",
    )
    p_train.add_argument(
        "--exclude-end",
        type=str,
        default=None,
        help="Exclude window end token (single hole removed from training).",
    )

    # optional periodic validation during training (EvalCallback)
    p_train.add_argument(
        "--val-start",
        type=str,
        default=None,
        help="Validation window start token for periodic evaluation during training.",
    )
    p_train.add_argument(
        "--val-end",
        type=str,
        default=None,
        help="Validation window end token for periodic evaluation during training.",
    )
    p_train.add_argument(
        "--val-freq",
        type=int,
        default=50_000,
        help="Validation frequency in timesteps (only used if --val-start/--val-end are provided).",
    )

    # training length
    p_train.add_argument(
        "--episode-length-train",
        type=int,
        default=3600,
        help="Training episode length (timesteps) used to map n_episodes -> total_timesteps.",
    )
    p_train.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help=(
            "Total timesteps to train for this invocation. If --n-episodes is not provided, "
            "n_episodes = ceil(total_timesteps / episode_length_train)."
        ),
    )
    p_train.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help=(
            "Training-length knob used by DRLModel.train(). If provided, overrides --total-timesteps. "
            "Effective total_timesteps = n_episodes * episode_length_train."
        ),
    )

    # resume
    p_train.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from <logdir>/last_model.zip.",
    )
    p_train.add_argument(
        "--resume-model",
        type=str,
        default=None,
        help="Path to an existing SB3 .zip model to resume training from (overrides --resume).",
    )
    p_train.add_argument(
        "--addtl-timesteps",
        type=int,
        default=None,
        help="Additional timesteps to train when resuming. If omitted, uses --total-timesteps.",
    )
    p_train.add_argument(
        "--allow-window-change",
        action="store_true",
        help="When resuming, allow changing training/validation windows (otherwise changes are ignored).",
    )

    # core config
    p_train.add_argument("--seed", type=int, default=0, help="Random seed passed to SB3.")
    p_train.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=["ppo"],
        help="RL algorithm to use (only PPO wired up for now).",
    )
    p_train.add_argument("--device", type=str, default="auto", help="SB3 device (cpu, cuda, auto).")
    p_train.add_argument("--gamma", type=float, default=0.999, help="Discount factor (PPO gamma).")
    p_train.add_argument("--n-envs", type=int, default=1, help="Number of parallel envs.")
    p_train.add_argument(
        "--use-subproc-vec",
        action="store_true",
        help="Use SubprocVecEnv for parallelism (recommended only for script usage).",
    )

    # PPO knobs (only used when NOT resuming)
    p_train.add_argument("--n-steps", type=int, default=2048, help="PPO n_steps (ignored when resuming).")
    p_train.add_argument("--batch-size", type=int, default=4096, help="PPO batch_size (ignored when resuming).")
    p_train.add_argument("--n-epochs", type=int, default=10, help="PPO n_epochs (ignored when resuming).")
    # (val-freq is above; keep eval-freq out of the user-facing API)
    p_train.add_argument(
        "--no-track-reward-components",
        action="store_true",
        help="Disable tracking mean reward components per PPO update.",
    )

    p_train.add_argument(
        "--reward-spec",
        type=str,
        default="dam_safety:storage_band",
        help=(
            "Comma-separated objective specs. Examples: "
            "'dam_safety:storage_band,esa_min_flow:baseline,hydropower:baseline' or with per-objective alpha "
            "multipliers: 'dam_safety:storage_band@8.0,esa_min_flow:baseline@3.0,physics:baseline'"
        ),
    )
    p_train.add_argument(
        "--logdir",
        type=str,
        default="runs/navajo",
        help="Directory to store models and logs (run directory).",
    )

    # ------------------------------------------------------------------
    # eval
    # ------------------------------------------------------------------
    p_eval = sub.add_parser("eval", help="Evaluate a saved model on a specified window")
    p_eval.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model file.")
    p_eval.add_argument("--start", type=str, required=True, help="Window start token ('YYYY' or 'YYYY-MM-DD').")
    p_eval.add_argument("--end", type=str, required=True, help="Window end token ('YYYY' or 'YYYY-MM-DD').")
    p_eval.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory for eval artifacts. Default: <run_dir>/eval_<start>_<end>",
    )
    p_eval.add_argument(
        "--reward-spec",
        type=str,
        default=None,
        help="Override reward spec. If omitted, will try to read run_manifest.json next to the model.",
    )
    p_eval.add_argument("--device", type=str, default="auto", help="SB3 device (cpu, cuda, auto).")
    p_eval.add_argument(
        "--which-metrics",
        type=str,
        default="core",
        help="Metric group to compute (e.g., 'core', 'spr', 'all').",
    )
    p_eval.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p_eval.add_argument("--no-rollout", action="store_true", help="Skip saving eval rollout parquet/csv.")
    p_eval.add_argument("--no-metrics", action="store_true", help="Skip saving eval metrics csv/json.")

    # ------------------------------------------------------------------
    # test (alias for eval)
    # ------------------------------------------------------------------
    p_test = sub.add_parser("test", help="Alias for eval (evaluate a saved model on a specified window)")
    p_test.add_argument("--model", type=str, required=True, help="Path to SB3 .zip model file.")
    p_test.add_argument("--start", type=str, required=True, help="Window start token ('YYYY' or 'YYYY-MM-DD').")
    p_test.add_argument("--end", type=str, required=True, help="Window end token ('YYYY' or 'YYYY-MM-DD').")
    p_test.add_argument("--outdir", type=str, default=None, help="Output directory for eval artifacts. Default: <run_dir>/eval_<start>_<end>")
    p_test.add_argument("--reward-spec", type=str, default=None, help="Override reward spec. If omitted, will try to read run_manifest.json next to the model.")
    p_test.add_argument("--device", type=str, default="auto", help="SB3 device (cpu, cuda, auto).")
    p_test.add_argument("--which-metrics", type=str, default="core", help="Metric group to compute (e.g., 'core', 'spr', 'all').")
    p_test.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p_test.add_argument("--no-rollout", action="store_true", help="Skip saving eval rollout parquet/csv.")
    p_test.add_argument("--no-metrics", action="store_true", help="Skip saving eval metrics csv/json.")

    # ------------------------------------------------------------------
    # report-metrics
    # ------------------------------------------------------------------
    p_report = sub.add_parser(
        "report-metrics",
        help="Build a presentation-friendly master_metrics.xlsx workbook from eval_metrics.csv files",
    )
    p_report.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help=(
            "Root directory to scan recursively for eval_metrics.csv files. "
            f"Default: {_DEFAULT_RUNS_ROOT}"
        ),
    )
    p_report.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output workbook path. Default: <runs_root>/master_metrics.xlsx",
    )
    p_report.add_argument(
        "--metrics-filename",
        type=str,
        default="eval_metrics.csv",
        help="Metrics filename to scan for recursively.",
    )


    # ------------------------------------------------------------------
    # plan-sweep
    # ------------------------------------------------------------------
    p_plan = sub.add_parser(
        "plan-sweep",
        help="Expand a JSON sweep spec into a task plan plus SLURM helper scripts",
    )
    p_plan.add_argument("--spec", type=str, required=True, help="Path to a JSON sweep spec.")
    p_plan.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Control/output directory for generated plan + scripts. Default: <runs_root>/<sweep_name>/_sweep",
    )

    # ------------------------------------------------------------------
    # submit-sweep
    # ------------------------------------------------------------------
    p_submit = sub.add_parser(
        "submit-sweep",
        help="Materialize a JSON sweep spec and immediately submit the array + dependent report jobs via sbatch",
    )
    p_submit.add_argument("--spec", type=str, required=True, help="Path to a JSON sweep spec.")
    p_submit.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Control/output directory for generated plan + scripts. Default: <runs_root>/<sweep_name>/_sweep",
    )

    # ------------------------------------------------------------------
    # run-sweep-task
    # ------------------------------------------------------------------
    p_task = sub.add_parser(
        "run-sweep-task",
        help="Run one task from a materialized sweep plan (intended for SLURM arrays)",
    )
    p_task.add_argument("--plan", type=str, required=True, help="Path to a plan.json produced by plan-sweep.")
    p_task.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="0-based task id. Defaults to SLURM_ARRAY_TASK_ID if unset.",
    )
    p_task.add_argument(
        "--no-skip-complete",
        action="store_true",
        help="Force rerun even if the model and eval artifacts already exist.",
    )

    return parser




def cmd_plan_sweep(args) -> None:
    result = sweeps.materialize_sweep_plan(spec_path=args.spec, outdir=args.outdir)
    print(
        "[cli] wrote sweep plan: "
        f"{result['plan_path']} (tasks={result['n_tasks']}, sweep_root={result['sweep_root']})"
    )
    print(f"[cli] SLURM helper dir: {result['control_dir']}")


def cmd_submit_sweep(args) -> None:
    plan = sweeps.materialize_sweep_plan(spec_path=args.spec, outdir=args.outdir)
    print(
        "[cli] wrote sweep plan: "
        f"{plan['plan_path']} (tasks={plan['n_tasks']}, sweep_root={plan['sweep_root']})"
    )
    print(f"[cli] SLURM helper dir: {plan['control_dir']}")
    result = sweeps.submit_materialized_sweep(plan["control_dir"])
    print(f"[cli] submitted sweep array job: {result['array_jobid']}")
    print(f"[cli] submitted report job: {result['report_jobid']}")


def cmd_run_sweep_task(args) -> None:
    task_id = args.task_id
    if task_id is None:
        env_val = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_val is None:
            raise ValueError("--task-id was not provided and SLURM_ARRAY_TASK_ID is not set.")
        task_id = int(env_val)

    result = sweeps.run_sweep_task(
        plan_path=args.plan,
        task_id=int(task_id),
        skip_complete=(not args.no_skip_complete),
    )
    print(
        "[cli] completed sweep task: "
        f"task_id={result['task_id']} status={result['status']} logdir={result['logdir']}"
    )

def cmd_info(args) -> None:
    all_data = model.load_all_model_data()
    rng = helpers.available_date_range(all_data["raw"])
    if args.quiet:
        print(f"{rng['start'].date()} {rng['end'].date()}")
        return

    print("[data] available model_data range:")
    print(f"  start: {rng['start'].date()} (WY {rng['min_water_year']})")
    print(f"  end:   {rng['end'].date()} (WY {rng['max_water_year']})")
    print(f"  days:  {rng['n_days']}")


def cmd_train(args) -> None:
    # ------------------------------------------------------------------
    # Validate CLI arguments
    # ------------------------------------------------------------------
    is_resume = bool(args.resume) or (args.resume_model is not None)

    mode_count = 0
    mode_count += int(bool(args.use_full_record))
    mode_count += int(args.n_years_train is not None)
    mode_count += int((args.train_start is not None) or (args.train_end is not None))
    if (not is_resume) and mode_count == 0:
        raise ValueError(
            "Please specify a training window via one of: --use-full-record, --n-years-train N, "
            "or --train-start/--train-end. Use `python -m deepreservoir.drl.cli info` to see the available range."
        )
    if mode_count > 1 and (not is_resume or args.allow_window_change):
        raise ValueError(
            "Training window options are mutually exclusive. Choose only one of: --use-full-record, "
            "--n-years-train, or --train-start/--train-end."
        )

    if ((args.exclude_start is None) ^ (args.exclude_end is None)) and (not is_resume or args.allow_window_change):
        raise ValueError("--exclude-start and --exclude-end must be provided together.")
    if ((args.val_start is None) ^ (args.val_end is None)) and (not is_resume or args.allow_window_change):
        raise ValueError("--val-start and --val-end must be provided together.")

    if (not is_resume) and (args.batch_size > (args.n_steps * args.n_envs)):
        raise ValueError(
            f"batch_size ({args.batch_size}) must be <= n_steps*n_envs ({args.n_steps}*{args.n_envs}={args.n_steps*args.n_envs})."
        )

    # If resuming, use the model's run directory + manifest defaults unless explicitly overridden.
    manifest_cfg = None
    inferred_run_dir = None
    resume_model_path = None
    if is_resume:
        if args.resume_model is not None:
            resume_model_path = Path(args.resume_model)
        else:
            resume_model_path = Path(args.logdir) / "last_model.zip"

        if not resume_model_path.exists():
            raise FileNotFoundError(f"Resume model not found: {resume_model_path}")

        inferred_run_dir = model.infer_run_dir_from_model_path(resume_model_path)
        manifest_path = inferred_run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest_cfg = model._read_json(manifest_path).get("config", None)
            except Exception:
                manifest_cfg = None

    # If user didn't change logdir from the default and we can infer a run_dir, use it.
    logdir = Path(args.logdir)
    if inferred_run_dir is not None and str(args.logdir) == "runs/navajo":
        logdir = inferred_run_dir

    # Fill unset args from manifest (resume workflow)
    if manifest_cfg is not None:
        # reward spec
        if args.reward_spec == "dam_safety:storage_band" and manifest_cfg.get("reward_spec"):
            args.reward_spec = str(manifest_cfg["reward_spec"])

        # window tokens (only if user didn't provide)
        # Train/eval config (new manifest format)
        # Window config: by default, resume uses manifest windows. Ignore
        # attempted window changes unless --allow-window-change is set.
        tr = manifest_cfg.get("train", {}) if isinstance(manifest_cfg.get("train"), dict) else {}
        hole = manifest_cfg.get("exclude", {}) if isinstance(manifest_cfg.get("exclude"), dict) else {}
        val = manifest_cfg.get("val", {}) if isinstance(manifest_cfg.get("val"), dict) else {}

        user_any_window = any(
            [
                bool(args.use_full_record),
                args.n_years_train is not None,
                args.train_start is not None,
                args.train_end is not None,
                args.exclude_start is not None,
                args.exclude_end is not None,
                args.val_start is not None,
                args.val_end is not None,
            ]
        )

        if not args.allow_window_change:
            if user_any_window:
                print("[cli][warn] resume: ignoring provided window arguments (use --allow-window-change to override).")

            args.use_full_record = bool(tr.get("use_full_record", False))
            args.n_years_train = tr.get("n_years_train", None)
            args.train_start = tr.get("train_start", None)
            args.train_end = tr.get("train_end", None)
            args.exclude_start = hole.get("exclude_start", None)
            args.exclude_end = hole.get("exclude_end", None)
            args.val_start = val.get("val_start", None)
            args.val_end = val.get("val_end", None)
            if isinstance(val.get("val_freq"), int):
                args.val_freq = int(val.get("val_freq"))
        else:
            # Fill missing from manifest.
            if not (args.use_full_record or (args.n_years_train is not None) or (args.train_start or args.train_end)):
                args.use_full_record = bool(tr.get("use_full_record", False))
                args.n_years_train = tr.get("n_years_train", None)
                args.train_start = tr.get("train_start", None)
                args.train_end = tr.get("train_end", None)
            if args.exclude_start is None and hole.get("exclude_start") is not None:
                args.exclude_start = hole.get("exclude_start")
            if args.exclude_end is None and hole.get("exclude_end") is not None:
                args.exclude_end = hole.get("exclude_end")
            if args.val_start is None and val.get("val_start") is not None:
                args.val_start = val.get("val_start")
            if args.val_end is None and val.get("val_end") is not None:
                args.val_end = val.get("val_end")

        # core env/training config
        if isinstance(manifest_cfg.get("episode_length_train"), int):
            args.episode_length_train = int(manifest_cfg["episode_length_train"])
        if isinstance(manifest_cfg.get("seed"), int):
            args.seed = int(manifest_cfg["seed"])
        if isinstance(manifest_cfg.get("n_envs"), int):
            args.n_envs = int(manifest_cfg["n_envs"])
        if isinstance(manifest_cfg.get("use_subproc_vec"), bool):
            args.use_subproc_vec = bool(manifest_cfg["use_subproc_vec"])

    # Determine n_episodes from total_timesteps if user didn't provide it.
    if args.n_episodes is None:
        args.n_episodes = int(math.ceil(float(args.total_timesteps) / float(args.episode_length_train)))

    eff_total = int(args.n_episodes * args.episode_length_train)
    print(
        "[cli] training config: "
        f"n_episodes={args.n_episodes} episode_length_train={args.episode_length_train} "
        f"-> total_timesteps≈{eff_total} (requested {args.total_timesteps})"
    )

    logdir.mkdir(parents=True, exist_ok=True)

    drlm = model.DRLModel(
        reward_spec=args.reward_spec,
        n_years_train=args.n_years_train,
        use_full_record=bool(args.use_full_record),
        train_start=args.train_start,
        train_end=args.train_end,
        exclude_start=args.exclude_start,
        exclude_end=args.exclude_end,
        val_start=args.val_start,
        val_end=args.val_end,
        algo=args.algo,
        logdir=logdir,
        seed=args.seed,
        device=args.device,
        gamma=args.gamma,
        n_envs=args.n_envs,
        use_subproc_vec=args.use_subproc_vec,
        episode_length_train=args.episode_length_train,
    )

    resume = False
    addtl_timesteps = None
    if is_resume:
        resume = True
        print(f"[cli] resuming from: {resume_model_path}")
        drlm.load_model(str(resume_model_path))
        addtl_timesteps = int(args.addtl_timesteps) if args.addtl_timesteps is not None else int(args.total_timesteps)

    drlm.train(
        n_episodes=args.n_episodes,
        total_timesteps=(addtl_timesteps if resume else eff_total),
        val_freq=(args.val_freq if (args.val_start is not None) else None),
        n_steps=(None if resume else args.n_steps),
        batch_size=(None if resume else args.batch_size),
        n_epochs=(None if resume else args.n_epochs),
        gamma=(None if resume else args.gamma),
        track_reward_components=(not args.no_track_reward_components),
        resume=resume,
    )


def cmd_report_metrics(args) -> None:
    from deepreservoir.drl import reporting

    runs_root = Path(args.runs_root) if args.runs_root is not None else _DEFAULT_RUNS_ROOT

    result = reporting.build_master_metrics_workbook(
        runs_root=runs_root,
        outpath=args.out,
        metrics_filename=args.metrics_filename,
    )
    print(
        "[cli] wrote metrics dashboard: "
        f"{result['outpath']} (evals={result['n_evals']}, experiments={result['n_experiments']})"
    )


def cmd_eval(args) -> None:
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = model.infer_run_dir_from_model_path(model_path)

    reward_spec = args.reward_spec
    if reward_spec is None:
        manifest = run_dir / "run_manifest.json"
        if manifest.exists():
            try:
                reward_spec = model._read_json(manifest).get("config", {}).get("reward_spec", None)
            except Exception:
                reward_spec = None
    if reward_spec is None:
        raise ValueError(
            "Could not infer reward_spec from run_manifest.json. Please pass --reward-spec explicitly."
        )

    if args.outdir is None:
        stem = f"eval_{_sanitize_token(args.start)}_{_sanitize_token(args.end)}"
        outdir = run_dir / stem
    else:
        outdir = Path(args.outdir)

    _, df_metrics = model.evaluate_model_window(
        model_path=model_path,
        reward_spec=reward_spec,
        window_start=args.start,
        window_end=args.end,
        outdir=outdir,
        device=args.device,
        which_metrics=args.which_metrics,
        save_rollout=(not args.no_rollout),
        save_plots=(not args.no_plots),
        save_metrics=(not args.no_metrics),
    )

    try:
        print("\n[cli] eval metrics:\n" + df_metrics.to_string(index=False))
    except Exception:
        pass


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "info":
        cmd_info(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd in ("eval", "test"):
        cmd_eval(args)
    elif args.cmd == "report-metrics":
        cmd_report_metrics(args)
    elif args.cmd == "plan-sweep":
        cmd_plan_sweep(args)
    elif args.cmd == "submit-sweep":
        cmd_submit_sweep(args)
    elif args.cmd == "run-sweep-task":
        cmd_run_sweep_task(args)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
