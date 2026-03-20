from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from deepreservoir.drl import model


_REPO_ROOT = Path(__file__).resolve().parents[3]


TRAIN_KEYS = {
    "use_full_record",
    "n_years_train",
    "train_start",
    "train_end",
    "exclude_start",
    "exclude_end",
    "val_start",
    "val_end",
    "val_freq",
    "episode_length_train",
    "total_timesteps",
    "n_episodes",
    "resume",
    "resume_model",
    "addtl_timesteps",
    "allow_window_change",
    "seed",
    "algo",
    "device",
    "gamma",
    "n_envs",
    "use_subproc_vec",
    "n_steps",
    "batch_size",
    "n_epochs",
    "track_reward_components",
    "reward_spec",
    "logdir",
}

EVAL_KEYS = {
    "device",
    "which_metrics",
    "save_plots",
    "save_rollout",
    "save_metrics",
    "windows",
}


class SweepSpecError(ValueError):
    pass


def _expand_pathlike(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def _sanitize_name(value: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "item"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SweepSpecError(f"Sweep spec must be a JSON object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def _merge_dict(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    out = deepcopy(base)
    if not override:
        return out
    for k, v in override.items():
        out[k] = deepcopy(v)
    return out


def _validate_train_args(d: dict[str, Any], *, context: str) -> dict[str, Any]:
    extra = sorted(set(d) - TRAIN_KEYS)
    if extra:
        raise SweepSpecError(f"Unknown train key(s) in {context}: {extra}")
    return d


def _validate_eval_args(d: dict[str, Any], *, context: str) -> dict[str, Any]:
    extra = sorted(set(d) - EVAL_KEYS)
    if extra:
        raise SweepSpecError(f"Unknown eval key(s) in {context}: {extra}")
    return d


def _normalize_windows(windows: list[dict[str, Any]] | None, *, context: str) -> list[dict[str, str]]:
    if not windows:
        raise SweepSpecError(f"No eval windows defined in {context}.")
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, w in enumerate(windows):
        if not isinstance(w, dict):
            raise SweepSpecError(f"Eval window #{i} in {context} must be an object.")
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            raise SweepSpecError(f"Eval window #{i} in {context} must include 'start' and 'end'.")
        name = _sanitize_name(w.get("name") or f"{start}_{end}")
        if name in seen:
            raise SweepSpecError(f"Duplicate eval window name in {context}: {name}")
        seen.add(name)
        out.append({"name": name, "start": str(start), "end": str(end)})
    return out


def _normalize_spec(spec_path: Path) -> dict[str, Any]:
    spec = _read_json(spec_path)
    sweep_name = _sanitize_name(spec.get("sweep_name") or spec_path.stem)
    runs_root = _expand_pathlike(spec.get("runs_root"))
    if runs_root is None:
        runs_root = (_REPO_ROOT / "runs").resolve()

    base_train = _validate_train_args(dict(spec.get("base_train") or {}), context="base_train")
    base_eval = _validate_eval_args(dict(spec.get("base_eval") or {}), context="base_eval")
    base_eval_windows = _normalize_windows(base_eval.get("windows"), context="base_eval")
    base_eval["windows"] = base_eval_windows

    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise SweepSpecError("Sweep spec must define a non-empty 'experiments' list.")

    norm_experiments: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    task_id = 0
    sweep_root = (runs_root / sweep_name).resolve()

    for idx, exp in enumerate(experiments):
        if not isinstance(exp, dict):
            raise SweepSpecError(f"Experiment #{idx} must be an object.")
        exp_name = _sanitize_name(exp.get("name") or f"exp_{idx:03d}")
        seeds = exp.get("seeds", spec.get("seeds", [0]))
        if isinstance(seeds, int):
            seeds = [int(seeds)]
        if not isinstance(seeds, list) or not seeds:
            raise SweepSpecError(f"Experiment '{exp_name}' must define one or more seeds.")
        seeds = [int(s) for s in seeds]

        train_overrides = _validate_train_args(dict(exp.get("train_overrides") or {}), context=f"experiment[{exp_name}].train_overrides")
        eval_overrides = _validate_eval_args(dict(exp.get("eval_overrides") or {}), context=f"experiment[{exp_name}].eval_overrides")

        train_args = _merge_dict(base_train, train_overrides)
        if exp.get("reward_spec") is not None:
            train_args["reward_spec"] = str(exp.get("reward_spec"))
        if not train_args.get("reward_spec"):
            raise SweepSpecError(f"Experiment '{exp_name}' has no reward_spec (set base_train.reward_spec or experiment.reward_spec).")

        eval_args = _merge_dict(base_eval, eval_overrides)
        eval_args["windows"] = _normalize_windows(eval_args.get("windows"), context=f"experiment[{exp_name}].eval_overrides or base_eval")
        eval_args.setdefault("which_metrics", "core")
        eval_args.setdefault("device", str(train_args.get("device", "auto")))
        eval_args.setdefault("save_plots", True)
        eval_args.setdefault("save_rollout", True)
        eval_args.setdefault("save_metrics", True)

        norm_experiments.append(
            {
                "name": exp_name,
                "seeds": seeds,
                "reward_spec": train_args["reward_spec"],
                "train_overrides": train_overrides,
                "eval_overrides": eval_overrides,
            }
        )

        for seed in seeds:
            logdir = (sweep_root / exp_name / f"seed_{seed:03d}").resolve()
            task_rows.append(
                {
                    "task_id": task_id,
                    "sweep_name": sweep_name,
                    "experiment_name": exp_name,
                    "seed": seed,
                    "logdir": str(logdir),
                    "train_args": _merge_dict(train_args, {"seed": seed, "logdir": str(logdir)}),
                    "eval_args": deepcopy(eval_args),
                }
            )
            task_id += 1

    normalized = {
        "sweep_name": sweep_name,
        "spec_path": str(spec_path.resolve()),
        "runs_root": str(runs_root),
        "sweep_root": str(sweep_root),
        "repo_root": str(_REPO_ROOT),
        "base_train": base_train,
        "base_eval": base_eval,
        "experiments": norm_experiments,
        "tasks": task_rows,
        "slurm": dict(spec.get("slurm") or {}),
    }
    return normalized


def _task_csv_rows(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        train_args = task["train_args"]
        eval_args = task["eval_args"]
        rows.append(
            {
                "task_id": task["task_id"],
                "experiment_name": task["experiment_name"],
                "seed": task["seed"],
                "logdir": task["logdir"],
                "reward_spec": train_args.get("reward_spec", ""),
                "train_start": train_args.get("train_start", ""),
                "train_end": train_args.get("train_end", ""),
                "n_years_train": train_args.get("n_years_train", ""),
                "total_timesteps": train_args.get("total_timesteps", ""),
                "n_episodes": train_args.get("n_episodes", ""),
                "which_metrics": eval_args.get("which_metrics", "core"),
                "eval_windows": ";".join(f"{w['name']}:{w['start']}->{w['end']}" for w in eval_args.get("windows", [])),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _shell_setup_lines(normalized: dict[str, Any]) -> list[str]:
    slurm = normalized.get("slurm", {}) or {}
    lines = [f'cd "{normalized["repo_root"]}"', f'export PYTHONPATH="{normalized["repo_root"]}/src:${{PYTHONPATH:-}}"']
    extra_pythonpath = slurm.get("pythonpath")
    if extra_pythonpath:
        lines.append(f'export PYTHONPATH="{extra_pythonpath}:${{PYTHONPATH}}"')
    for mod in slurm.get("module_loads", []) or []:
        lines.append(str(mod))
    for cmd in slurm.get("setup_commands", []) or []:
        lines.append(str(cmd))
    activate = slurm.get("activate")
    if activate:
        lines.append(str(activate))
    return lines


def _shell_executable(normalized: dict[str, Any]) -> str:
    slurm = normalized.get("slurm", {}) or {}
    shell = slurm.get("shell")
    if shell:
        return str(shell)
    if slurm.get("login_shell"):
        return "/bin/bash -l"
    return "/bin/bash"


def _python_executable(normalized: dict[str, Any]) -> str:
    slurm = normalized.get("slurm", {}) or {}
    py = slurm.get("python_executable")
    if py:
        return str(py)
    return "python"


def _sbatch_lines(normalized: dict[str, Any], *, n_tasks: int, logs_dir: Path) -> list[str]:
    slurm = normalized.get("slurm", {}) or {}
    job_name = _sanitize_name(slurm.get("job_name") or f"dr_{normalized['sweep_name']}")
    array = f"0-{n_tasks - 1}"
    array_parallelism = slurm.get("array_parallelism")
    if array_parallelism is not None:
        array = f"{array}%{int(array_parallelism)}"
    lines = [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --array={array}",
        f"#SBATCH --output={logs_dir}/%x_%A_%a.out",
        f"#SBATCH --error={logs_dir}/%x_%A_%a.err",
    ]
    mapping = {
        "account": "--account",
        "partition": "--partition",
        "qos": "--qos",
        "constraint": "--constraint",
        "cpus_per_task": "--cpus-per-task",
        "mem": "--mem",
        "time": "--time",
    }
    for key, flag in mapping.items():
        val = slurm.get(key)
        if val is not None and str(val) != "":
            lines.append(f"#SBATCH {flag}={val}")
    for line in slurm.get("extra_sbatch_lines", []) or []:
        line = str(line)
        lines.append(line if line.startswith("#SBATCH") else f"#SBATCH {line}")
    return lines


def _write_script(path: Path, text: str | list[str], shell_executable: str = "/bin/bash") -> None:
    if isinstance(text, list):
        script_text = "\n".join(str(line) for line in text)
    else:
        script_text = str(text)

    script_text = script_text.replace("\r\n", "\n")
    if not script_text.endswith("\n"):
        script_text += "\n"

    shebang = f"#!{shell_executable}"
    lines = script_text.splitlines()
    if lines and lines[0].startswith("#!"):
        lines[0] = shebang
    else:
        lines.insert(0, shebang)
    script_text = "\n".join(lines) + "\n"

    path.write_text(script_text, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)

def materialize_sweep_plan(spec_path: Path | str, outdir: Path | str | None = None) -> dict[str, Any]:
    spec_path = Path(spec_path).resolve()
    normalized = _normalize_spec(spec_path)
    sweep_root = Path(normalized["sweep_root"])
    control_dir = _expand_pathlike(outdir) if outdir is not None else (sweep_root / "_sweep")
    assert control_dir is not None
    control_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = control_dir / "slurm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = deepcopy(normalized)
    plan["control_dir"] = str(control_dir)
    plan_path = control_dir / "plan.json"
    _write_json(plan_path, plan)
    _write_json(control_dir / "normalized_spec.json", normalized)
    _write_csv(control_dir / "task_matrix.csv", _task_csv_rows(plan["tasks"]))

    readme_lines = [
        f"Sweep name: {plan['sweep_name']}",
        f"Sweep root: {plan['sweep_root']}",
        f"Tasks: {len(plan['tasks'])}",
        "",
        "Primary files:",
        f"  - plan.json           : machine-readable task plan",
        f"  - task_matrix.csv     : human-readable task table",
        f"  - sweep_array.sbatch  : SLURM array job",
        f"  - sweep_report.sbatch : report job (builds master_metrics.xlsx after the array finishes)",
        f"  - submit_sweep.sh     : convenience wrapper that submits both jobs with a dependency",
        "",
        "Recommended usage:",
        "  1) Inspect task_matrix.csv.",
        "  2) Run ./submit_sweep.sh on the HPC login node (or use `submit-sweep` from the CLI).",
        "",
        "Notes:",
        "  - Each task trains into its own run directory and then evaluates all configured windows.",
        "  - Plots are exported by default for each eval window unless save_plots=false in the spec.",
        "  - The master workbook is built once, after the array finishes, to avoid concurrent writes.",
    ]
    (control_dir / "README.txt").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    setup_lines = ["set -euo pipefail"] + _shell_setup_lines(plan)
    shell_executable = _shell_executable(plan)
    python_executable = _python_executable(plan)
    array_script = []
    array_script.extend(_sbatch_lines(plan, n_tasks=len(plan["tasks"]), logs_dir=logs_dir))
    array_script.extend(setup_lines)
    array_script.append('TASK_ID="${SLURM_ARRAY_TASK_ID}"')
    array_script.append(f'{python_executable} -m deepreservoir.drl.cli run-sweep-task --plan "{plan_path}" --task-id "${{TASK_ID}}"')
    _write_script(control_dir / "sweep_array.sbatch", array_script, shell_executable=shell_executable)

    report_script = []
    report_script.extend([
        f"#SBATCH --job-name={_sanitize_name('report_' + plan['sweep_name'])}",
        f"#SBATCH --output={logs_dir}/%x_%j.out",
        f"#SBATCH --error={logs_dir}/%x_%j.err",
    ])
    slurm = plan.get("slurm", {}) or {}
    for key, flag in {"account": "--account", "partition": "--partition", "qos": "--qos", "constraint": "--constraint", "cpus_per_task": "--cpus-per-task", "mem": "--mem", "time": "--time"}.items():
        val = slurm.get(key)
        if val is not None and str(val) != "":
            report_script.append(f"#SBATCH {flag}={val}")
    report_script.extend(setup_lines)
    report_script.append(f'{python_executable} -m deepreservoir.drl.cli report-metrics --runs-root "{sweep_root}"')
    _write_script(control_dir / "sweep_report.sbatch", report_script, shell_executable=shell_executable)

    submit_lines = [
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "$0")" && pwd)"',
        'cd "$HERE"',
        'ARRAY_JOBID=$(sbatch --parsable "$HERE/sweep_array.sbatch")',
        'echo "Submitted sweep array job: ${ARRAY_JOBID}"',
        'REPORT_JOBID=$(sbatch --parsable --dependency=afterok:${ARRAY_JOBID} "$HERE/sweep_report.sbatch")',
        'echo "Submitted report job: ${REPORT_JOBID}"',
    ]
    _write_script(control_dir / "submit_sweep.sh", submit_lines, shell_executable=shell_executable)

    return {
        "plan_path": str(plan_path),
        "control_dir": str(control_dir),
        "sweep_root": str(sweep_root),
        "n_tasks": len(plan["tasks"]),
    }


def submit_materialized_sweep(control_dir: Path | str) -> dict[str, str]:
    control_dir = Path(control_dir).resolve()
    array_script = control_dir / "sweep_array.sbatch"
    report_script = control_dir / "sweep_report.sbatch"
    if not array_script.exists():
        raise FileNotFoundError(f"Missing array script: {array_script}")
    if not report_script.exists():
        raise FileNotFoundError(f"Missing report script: {report_script}")

    array_proc = subprocess.run(
        ["sbatch", "--parsable", str(array_script)],
        check=True,
        capture_output=True,
        text=True,
    )
    array_jobid = array_proc.stdout.strip().split(";")[0]

    report_proc = subprocess.run(
        ["sbatch", "--parsable", f"--dependency=afterok:{array_jobid}", str(report_script)],
        check=True,
        capture_output=True,
        text=True,
    )
    report_jobid = report_proc.stdout.strip().split(";")[0]

    submit_script = control_dir / "submit_sweep.sh"
    return {
        "control_dir": str(control_dir),
        "submit_script": str(submit_script),
        "array_jobid": array_jobid,
        "report_jobid": report_jobid,
    }


def _bool_flag(argv: list[str], *, positive_key: str, value: Any, pos_flag: str, neg_flag: str | None = None) -> None:
    if value is None:
        return
    if bool(value):
        argv.append(pos_flag)
    elif neg_flag is not None:
        argv.append(neg_flag)


def _append_kv(argv: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def _build_train_argv(task: dict[str, Any]) -> list[str]:
    a = task["train_args"]
    argv = [sys.executable, "-m", "deepreservoir.drl.cli", "train"]
    _bool_flag(argv, positive_key="use_full_record", value=a.get("use_full_record"), pos_flag="--use-full-record")
    _append_kv(argv, "--n-years-train", a.get("n_years_train"))
    _append_kv(argv, "--train-start", a.get("train_start"))
    _append_kv(argv, "--train-end", a.get("train_end"))
    _append_kv(argv, "--exclude-start", a.get("exclude_start"))
    _append_kv(argv, "--exclude-end", a.get("exclude_end"))
    _append_kv(argv, "--val-start", a.get("val_start"))
    _append_kv(argv, "--val-end", a.get("val_end"))
    _append_kv(argv, "--val-freq", a.get("val_freq"))
    _append_kv(argv, "--episode-length-train", a.get("episode_length_train"))
    _append_kv(argv, "--total-timesteps", a.get("total_timesteps"))
    _append_kv(argv, "--n-episodes", a.get("n_episodes"))
    _bool_flag(argv, positive_key="resume", value=a.get("resume"), pos_flag="--resume")
    _append_kv(argv, "--resume-model", a.get("resume_model"))
    _append_kv(argv, "--addtl-timesteps", a.get("addtl_timesteps"))
    _bool_flag(argv, positive_key="allow_window_change", value=a.get("allow_window_change"), pos_flag="--allow-window-change")
    _append_kv(argv, "--seed", a.get("seed"))
    _append_kv(argv, "--algo", a.get("algo"))
    _append_kv(argv, "--device", a.get("device"))
    _append_kv(argv, "--gamma", a.get("gamma"))
    _append_kv(argv, "--n-envs", a.get("n_envs"))
    _bool_flag(argv, positive_key="use_subproc_vec", value=a.get("use_subproc_vec"), pos_flag="--use-subproc-vec")
    _append_kv(argv, "--n-steps", a.get("n_steps"))
    _append_kv(argv, "--batch-size", a.get("batch_size"))
    _append_kv(argv, "--n-epochs", a.get("n_epochs"))
    if a.get("track_reward_components") is False:
        argv.append("--no-track-reward-components")
    _append_kv(argv, "--reward-spec", a.get("reward_spec"))
    _append_kv(argv, "--logdir", a.get("logdir"))
    return argv


def _build_eval_argv(task: dict[str, Any], window: dict[str, str]) -> tuple[list[str], Path]:
    train_args = task["train_args"]
    eval_args = task["eval_args"]
    logdir = Path(train_args["logdir"])
    outdir = logdir / f"eval__{window['name']}"
    argv = [
        sys.executable,
        "-m",
        "deepreservoir.drl.cli",
        "eval",
        "--model",
        str(logdir / "last_model.zip"),
        "--start",
        str(window["start"]),
        "--end",
        str(window["end"]),
        "--outdir",
        str(outdir),
        "--device",
        str(eval_args.get("device", train_args.get("device", "auto"))),
        "--which-metrics",
        str(eval_args.get("which_metrics", "core")),
    ]
    if eval_args.get("save_plots") is False:
        argv.append("--no-plots")
    if eval_args.get("save_rollout") is False:
        argv.append("--no-rollout")
    if eval_args.get("save_metrics") is False:
        argv.append("--no-metrics")
    return argv, outdir


def _eval_complete(outdir: Path, *, save_plots: bool, save_rollout: bool, save_metrics: bool) -> bool:
    if save_metrics and not (outdir / "eval_metrics.csv").exists():
        return False
    if save_rollout and not (outdir / "eval_rollout.parquet").exists():
        return False
    if save_plots and not (outdir / "plots").is_dir():
        return False
    return True


def _annotate_manifest(logdir: Path, task: dict[str, Any], plan_path: Path) -> None:
    manifest_path = logdir / "run_manifest.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        try:
            manifest = model._read_json(manifest_path)
        except Exception:
            manifest = {}
    else:
        manifest = {}
    manifest["sweep"] = {
        "sweep_name": task.get("sweep_name"),
        "experiment_name": task.get("experiment_name"),
        "task_id": int(task.get("task_id", -1)),
        "plan_path": str(plan_path),
        "seed": int(task.get("seed", 0)),
    }
    model._write_json(manifest_path, manifest)


def run_sweep_task(plan_path: Path | str, task_id: int, *, skip_complete: bool = True) -> dict[str, Any]:
    plan_path = Path(plan_path).resolve()
    plan = _read_json(plan_path)
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SweepSpecError(f"Plan has no tasks: {plan_path}")
    task_id = int(task_id)
    if task_id < 0 or task_id >= len(tasks):
        raise SweepSpecError(f"task_id out of range: {task_id} (n_tasks={len(tasks)})")

    task = tasks[task_id]
    logdir = Path(task["train_args"]["logdir"]).resolve()
    logdir.mkdir(parents=True, exist_ok=True)
    _annotate_manifest(logdir, task, plan_path)

    model_path = logdir / "last_model.zip"
    eval_args = task["eval_args"]
    save_plots = bool(eval_args.get("save_plots", True))
    save_rollout = bool(eval_args.get("save_rollout", True))
    save_metrics = bool(eval_args.get("save_metrics", True))

    pending_evals: list[tuple[dict[str, str], Path, list[str]]] = []
    for window in eval_args.get("windows", []):
        argv, outdir = _build_eval_argv(task, window)
        if (not skip_complete) or (not _eval_complete(outdir, save_plots=save_plots, save_rollout=save_rollout, save_metrics=save_metrics)):
            pending_evals.append((window, outdir, argv))

    resume_mode = bool(task["train_args"].get("resume")) or bool(task["train_args"].get("resume_model"))
    need_train = (not model_path.exists()) or resume_mode

    if skip_complete and (not need_train) and not pending_evals:
        print(f"[sweep] task {task_id}: already complete -> {logdir}")
        return {"task_id": task_id, "status": "skipped_complete", "logdir": str(logdir)}

    if need_train:
        train_argv = _build_train_argv(task)
        print(f"[sweep] task {task_id}: training {task['experiment_name']} seed={task['seed']}")
        subprocess.run(train_argv, check=True)
        if not model_path.exists():
            raise RuntimeError(f"Training finished but model was not found: {model_path}")
    else:
        print(f"[sweep] task {task_id}: reusing existing model {model_path}")

    _annotate_manifest(logdir, task, plan_path)

    for window, outdir, argv in pending_evals:
        print(f"[sweep] task {task_id}: eval {window['name']} ({window['start']} -> {window['end']})")
        subprocess.run(argv, check=True)
        if save_metrics and not (outdir / "eval_metrics.csv").exists():
            raise RuntimeError(f"Eval did not produce eval_metrics.csv: {outdir}")

    return {
        "task_id": task_id,
        "status": "ok",
        "logdir": str(logdir),
        "n_pending_evals": len(pending_evals),
        "trained": bool(need_train),
    }
