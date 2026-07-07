#!/usr/bin/env python3
"""Multi-node SLURM runner for distributed video QA generation.

Submits a single multi-node SLURM job. Each node processes a shard of the
input data independently using SLURM_PROCID for sharding. After all nodes
finish, rank 0 merges the shard outputs into the final prediction file.

Usage via generation scripts (one command, fully automatic):
    python run_generate_convqa.py \\
        --slurm-nodes 8 \\
        --slurm-partition gpu \\
        --conda-env /path/to/conda/env \\
        --model-type llama4 --llm-model /path/to/model \\
        --video-folder /path/to/videos
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys


def _split_jsonl(input_path: str, num_shards: int, output_dir: str) -> list[str]:
    with open(input_path) as f:
        lines = [line for line in f if line.strip()]

    shard_paths = []
    for i in range(num_shards):
        shard_lines = lines[i::num_shards]
        shard_path = os.path.join(output_dir, f"shard_{i}.jsonl")
        with open(shard_path, "w") as f:
            f.writelines(shard_lines)
        shard_paths.append(shard_path)

    return shard_paths


def _merge_shards(shard_dir: str, output_path: str, num_shards: int) -> int:
    shard_data: dict[int, list[str]] = {}
    missing: list[int] = []
    total = 0
    for i in range(num_shards):
        shard_path = os.path.join(shard_dir, f"pred_shard_{i}.jsonl")
        try:
            with open(shard_path) as f:
                shard_data[i] = [line for line in f if line.strip()]
        except FileNotFoundError:
            missing.append(i)
            shard_data[i] = []
        total += len(shard_data[i])
    if missing:
        raise FileNotFoundError(
            f"{len(missing)}/{num_shards} shard(s) missing: "
            + ", ".join(str(m) for m in missing)
        )

    max_per_shard = max(len(v) for v in shard_data.values()) if shard_data else 0
    with open(output_path, "w") as f:
        for idx in range(max_per_shard):
            for shard_id in range(num_shards):
                if idx < len(shard_data[shard_id]):
                    f.write(shard_data[shard_id][idx])
                    if not shard_data[shard_id][idx].endswith("\n"):
                        f.write("\n")

    return total


def _validate_sbatch_params(partition: str, reservation: str, time_limit: str) -> None:
    """Validate SLURM parameters against shell injection."""
    _sbatch_safe = re.compile(r"^[A-Za-z0-9_:.,/-]+$")
    for name, value in [
        ("partition", partition),
        ("reservation", reservation),
        ("time_limit", time_limit),
    ]:
        if value and not _sbatch_safe.match(value):
            raise ValueError(
                f"Invalid {name}={value!r}: must contain only "
                f"alphanumeric characters, underscores, colons, dots, "
                f"commas, hyphens, and forward slashes."
            )


def _resolve_conda_env(conda_env: str, conda_base: str) -> tuple[str, str, str, str]:
    """Resolve conda environment setup strings.

    Returns:
        (env_setup, conda_exports, srun_env_setup, q_conda) -- shell
        fragments for the head node, export block, srun worker, and
        quoted conda env path.
    """
    if not conda_env:
        return "", "", "", ""

    conda_sh = ""
    if conda_base:
        conda_sh = os.path.join(conda_base, "etc", "profile.d", "conda.sh")
        if not os.path.exists(conda_sh):
            raise ValueError(f"conda.sh not found at {conda_sh}")
    else:
        # Infer conda base from the env path.  Standard layout:
        #   /path/to/miniconda3/envs/foo -> conda base = /path/to/miniconda3
        env_parent = os.path.dirname(conda_env)
        if os.path.basename(env_parent) == "envs":
            inferred_base = os.path.dirname(env_parent)
        else:
            # Non-standard layout: treat the env path itself as the base
            inferred_base = conda_env
        candidate = os.path.join(
            inferred_base,
            "etc",
            "profile.d",
            "conda.sh",
        )
        if os.path.exists(candidate):
            conda_sh = candidate

    q_conda_sh = shlex.quote(conda_sh) if conda_sh else ""
    q_conda = shlex.quote(conda_env)
    conda_source = f"source {q_conda_sh}\n" if conda_sh else ""
    env_setup = f"{conda_source}conda activate {q_conda}\n"

    # Export conda paths as env vars so srun workers reference them via
    # ${_CONDA_SH} / ${_CONDA_ENV} -- avoids single-quote nesting inside
    # the bash -c '...' block.
    conda_exports = f"export _CONDA_ENV={q_conda}\n"
    if conda_sh:
        conda_exports += f"export _CONDA_SH={q_conda_sh}\n"
        srun_env_setup = 'source "${_CONDA_SH}" && conda activate "${_CONDA_ENV}"'
    else:
        srun_env_setup = 'conda activate "${_CONDA_ENV}"'

    return env_setup, conda_exports, srun_env_setup, q_conda


def _build_extra_sbatch(partition: str, reservation: str) -> str:
    """Build extra #SBATCH directive lines."""
    sbatch_lines = []
    if partition:
        sbatch_lines.append(f"#SBATCH --partition={partition}")
    if reservation:
        sbatch_lines.append(f"#SBATCH --reservation={reservation}")
    return "\n".join(sbatch_lines)


def submit(
    script: str,
    input_path: str,
    output_path: str,
    num_nodes: int,
    extra_args: list[str],
    partition: str = "",
    reservation: str = "",
    conda_env: str = "",
    conda_base: str = "",
    gpus_per_node: int = 8,
    time_limit: str = "4:00:00",
    post_merge_commands: list[str] | None = None,
) -> str:
    _validate_sbatch_params(partition, reservation, time_limit)

    work_dir = os.path.dirname(os.path.abspath(script))
    # SBATCH directives don't support quoting, so validate work_dir contains
    # no characters that could cause shell expansion (spaces, $, backticks,
    # %, etc.)
    _sbatch_unsafe = re.compile(r"[\s$`\"'\\!;|&<>(){}%]")
    unsafe_match = _sbatch_unsafe.search(work_dir)
    if unsafe_match:
        raise ValueError(
            f"work_dir contains unsafe character {unsafe_match.group()!r} "
            f"(SBATCH directives cannot quote paths): {work_dir}"
        )
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)

    output_stem = re.sub(
        r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(output_path))[0]
    )
    shard_dir = os.path.join(
        os.path.dirname(os.path.abspath(output_path)), f"_shards_{output_stem}"
    )
    os.makedirs(shard_dir, exist_ok=True)

    print(f"Splitting {input_path} into {num_nodes} shards...")
    shard_inputs = _split_jsonl(input_path, num_nodes, shard_dir)
    for i, p in enumerate(shard_inputs):
        with open(p) as f:
            n = sum(1 for _ in f)
        print(f"  Shard {i}: {n} samples")

    extra_sbatch = _build_extra_sbatch(partition, reservation)
    env_setup, conda_exports, srun_env_setup, _ = _resolve_conda_env(
        conda_env, conda_base
    )

    extra_str = " ".join(shlex.quote(a) for a in extra_args)
    q_extra_str = shlex.quote(extra_str) if extra_str else "''"
    q_work_dir = shlex.quote(work_dir)
    q_script = shlex.quote(script)
    q_shard_dir = shlex.quote(shard_dir)
    q_output_path = shlex.quote(output_path)
    post_merge = "\n".join(post_merge_commands) if post_merge_commands else ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks={num_nodes}
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --job-name=distributed_eval
#SBATCH --output={work_dir}/logs/{output_stem}_%j.log
#SBATCH --error={work_dir}/logs/{output_stem}_%j.log
{extra_sbatch}
set -euo pipefail
export PYTHONUNBUFFERED=1
cd {q_work_dir}
{env_setup}
{conda_exports}export _SHARD_DIR={q_shard_dir}
export _SCRIPT={q_script}
export _OUTPUT_PATH={q_output_path}
export _EXTRA_ARGS={q_extra_str}
srun bash -c '
  {srun_env_setup}
  SHARD_IN="${{_SHARD_DIR}}/shard_${{SLURM_PROCID}}.jsonl"
  SHARD_OUT="${{_SHARD_DIR}}/pred_shard_${{SLURM_PROCID}}.jsonl"
  echo "Node $(hostname): shard ${{SLURM_PROCID}}/{num_nodes}"
  eval python3 "${{_SCRIPT}}" --input "$SHARD_IN" --output "$SHARD_OUT" --no-eval ${{_EXTRA_ARGS}}
'

echo "All {num_nodes} shards done. Merging..."
python3 -c "
import os
from slurm_runner import _merge_shards
total = _merge_shards(os.environ['_SHARD_DIR'], os.environ['_OUTPUT_PATH'], {num_nodes})
print(f'Merged {{total}} predictions into ' + os.environ['_OUTPUT_PATH'])
"
{post_merge}
echo "DONE"
"""
    script_path = os.path.join(work_dir, f"_slurm_{output_stem}.sh")
    with open(script_path, "w") as f:
        f.write(sbatch_script)

    result = subprocess.run(
        ["sbatch", "--parsable", script_path],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        raise RuntimeError("sbatch failed")

    job_id = result.stdout.strip()
    print(f"\nSubmitted SLURM {job_id} ({num_nodes} nodes)")
    print(f"Monitor: sacct -j {job_id} --format=JobID,State,Elapsed")
    print(f"Logs: {work_dir}/logs/{output_stem}_{job_id}.log")

    return job_id


def _split_jsonl_pair(
    golden_path: str,
    preds_path: str,
    num_shards: int,
    shard_root: str,
) -> list[tuple[str, str]]:
    """Split golden + predictions JSONLs in lockstep by line index.

    The judge requires positionally-aligned pairs (`evaluate_convqa` enforces
    `len(golden) == len(preds)`), so both files must have the same number of
    rows. Returns list of `(golden_shard_path, preds_shard_path)`.
    """
    with open(golden_path) as f:
        gold_lines = [line for line in f if line.strip()]
    with open(preds_path) as f:
        pred_lines = [line for line in f if line.strip()]
    if len(gold_lines) != len(pred_lines):
        raise ValueError(
            f"Sharded judge requires equal-length golden/predictions, "
            f"got {len(gold_lines)} vs {len(pred_lines)}."
        )
    if num_shards > len(gold_lines):
        raise ValueError(
            f"num_shards ({num_shards}) exceeds number of conversations "
            f"({len(gold_lines)}); most shards would be empty. "
            f"Pass --llm-judge-nodes <= {len(gold_lines)}."
        )

    shard_paths: list[tuple[str, str]] = []
    for i in range(num_shards):
        sub = os.path.join(shard_root, f"shard_{i}")
        os.makedirs(sub, exist_ok=True)
        g_path = os.path.join(sub, "golden.jsonl")
        p_path = os.path.join(sub, "predictions.jsonl")
        with open(g_path, "w") as gf, open(p_path, "w") as pf:
            gf.writelines(gold_lines[i::num_shards])
            pf.writelines(pred_lines[i::num_shards])
        shard_paths.append((g_path, p_path))
    return shard_paths


def _merge_judge_shards(shard_root: str, output_path: str, num_shards: int) -> None:
    """Merge per-shard `results.json` files into a single results/summary pair.

    Each shard's `results.json` already has `per_row` (with `bleu_per_turn`
    and `llm_judge_per_turn`). We concatenate, then recompute the aggregates
    so the final numbers exactly match a single-shot single-node run.
    """
    import json

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_evaluation import (  # noqa: E402  (local-only import)
        _aggregate_convqa_output,
        _safe_mean,
        write_results,
    )

    merged_per_row: list[dict[str, object]] = []
    all_bleu: list[float] = []
    all_judge: list[float] = []
    cat_bleu: dict[str, list[float]] = {}
    cat_judge: dict[str, list[float]] = {}
    have_bleu = False
    have_judge = False
    missing: list[int] = []

    for i in range(num_shards):
        rpath = os.path.join(shard_root, f"shard_{i}", "results.json")
        if not os.path.exists(rpath):
            missing.append(i)
            continue
        with open(rpath) as f:
            shard = json.load(f)
        for row in shard.get("per_row", []):
            merged_per_row.append(row)
            cat = str(row.get("category", ""))
            if "bleu_per_turn" in row:
                have_bleu = True
                all_bleu.extend(row["bleu_per_turn"])
                cat_bleu.setdefault(cat, []).extend(row["bleu_per_turn"])
            if "llm_judge_per_turn" in row:
                have_judge = True
                all_judge.extend(row["llm_judge_per_turn"])
                cat_judge.setdefault(cat, []).extend(row["llm_judge_per_turn"])

    if missing:
        raise RuntimeError(
            f"Missing per-shard results for shards {missing} in {shard_root}"
        )

    merged_per_row.sort(key=lambda r: (r.get("video_path", ""), r.get("index", 0)))
    for j, row in enumerate(merged_per_row):
        row["index"] = j

    # Re-use the canonical aggregator so the schema matches single-node runs.
    # Use .get() defensively: shard rows may omit gold_answers/bleu_per_turn
    # (e.g., a shard ran without BLEU). The aggregator only inspects len(golden)
    # and the truthiness of bleu_scores/judge_scores, so None entries are safe.
    fake_golden = [{"answers": r.get("gold_answers")} for r in merged_per_row]
    merged = _aggregate_convqa_output(
        fake_golden,
        bleu_scores=[r.get("bleu_per_turn") for r in merged_per_row]
        if have_bleu
        else None,
        judge_scores=[r.get("llm_judge_per_turn") for r in merged_per_row]
        if have_judge
        else None,
        all_bleu=all_bleu,
        all_judge=all_judge,
        cat_bleu=cat_bleu,
        cat_judge=cat_judge,
        per_row=merged_per_row,
    )
    # `_aggregate_convqa_output` only sets total_turns from the flat lists,
    # which is what we want; keep mean check to catch arithmetic regressions.
    # Use explicit `raise` (not `assert`) so the check survives `python -O`.
    if have_bleu:
        expected_bleu = round(_safe_mean(all_bleu), 4)
        if abs(merged["bleu"] - expected_bleu) >= 1e-9:
            raise RuntimeError(
                f"bleu mean mismatch after merge: aggregator returned "
                f"{merged['bleu']!r}, expected {expected_bleu!r}"
            )
    if have_judge:
        expected_judge = round(_safe_mean(all_judge), 4)
        if abs(merged["llm_judge"] - expected_judge) >= 1e-9:
            raise RuntimeError(
                f"llm_judge mean mismatch after merge: aggregator returned "
                f"{merged['llm_judge']!r}, expected {expected_judge!r}"
            )

    summary_path = write_results(output_path, merged)
    print(f"Merged {num_shards} shards: {len(merged_per_row)} rows -> {output_path}")
    print(f"Summary: {summary_path}")


def submit_judge(
    task: str,
    golden_path: str,
    preds_path: str,
    output_path: str,
    num_nodes: int,
    extra_eval_args: list[str],
    partition: str = "",
    reservation: str = "",
    conda_env: str = "",
    conda_base: str = "",
    gpus_per_node: int = 8,
    time_limit: str = "4:00:00",
) -> tuple[str, str]:
    """Submit a sbatch array for sharded `--eval-only --llm-judge` scoring.

    Each array task scores one shard (1 node, 8 GPUs) by invoking
    `run_evaluation.py --task <task> --eval-only --llm-judge
    --llm-judge-nodes 0 ...`. A dependent merge job auto-runs after the
    array completes and writes the final results.json + results_summary.json.

    Returns `(array_job_id, merge_job_id)`.
    """
    if task != "convqa":
        raise ValueError(
            f"--llm-judge-nodes currently supports --task convqa only, got {task!r}."
        )
    _validate_sbatch_params(partition, reservation, time_limit)

    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "run_evaluation.py"
    )
    work_dir = os.path.dirname(os.path.abspath(script))
    _sbatch_unsafe = re.compile(r"[\s$`\"'\\!;|&<>(){}%]")
    unsafe_match = _sbatch_unsafe.search(work_dir)
    if unsafe_match:
        raise ValueError(
            f"work_dir contains unsafe character {unsafe_match.group()!r}: {work_dir}"
        )
    os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)

    output_stem = re.sub(
        r"[^A-Za-z0-9_.-]", "_", os.path.splitext(os.path.basename(output_path))[0]
    )
    shard_root = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        f"_judge_shards_{output_stem}",
    )
    os.makedirs(shard_root, exist_ok=True)

    print(f"Splitting golden+predictions into {num_nodes} shards under {shard_root}")
    shards = _split_jsonl_pair(golden_path, preds_path, num_nodes, shard_root)
    for i, (g, _p) in enumerate(shards):
        with open(g) as f:
            n = sum(1 for _ in f)
        print(f"  Shard {i}: {n} conversations")

    extra_sbatch = _build_extra_sbatch(partition, reservation)
    env_setup, conda_exports, srun_env_setup, _ = _resolve_conda_env(
        conda_env, conda_base
    )

    extra_str = " ".join(shlex.quote(a) for a in extra_eval_args)
    q_extra = shlex.quote(extra_str) if extra_str else "''"
    q_work_dir = shlex.quote(work_dir)
    q_script = shlex.quote(script)
    q_shard_root = shlex.quote(shard_root)
    q_task = shlex.quote(task)
    last_idx = num_nodes - 1

    array_script = f"""#!/bin/bash
#SBATCH --array=0-{last_idx}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node={gpus_per_node}
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time={time_limit}
#SBATCH --job-name=judge_{output_stem}
#SBATCH --output={work_dir}/logs/judge_{output_stem}_%A_%a.log
#SBATCH --error={work_dir}/logs/judge_{output_stem}_%A_%a.log
{extra_sbatch}
set -euo pipefail
export PYTHONUNBUFFERED=1
cd {q_work_dir}
{env_setup}
{conda_exports}export _SHARD_ROOT={q_shard_root}
export _SCRIPT={q_script}
export _TASK={q_task}
export _EXTRA={q_extra}
bash -c '
  set -e
  {srun_env_setup}
  SHARD_DIR="${{_SHARD_ROOT}}/shard_${{SLURM_ARRAY_TASK_ID}}"
  SHARD_T0=$(date -u +%s)
  echo "[shard ${{SLURM_ARRAY_TASK_ID}}/{num_nodes}] starting on $(hostname) at $(date -u +%FT%TZ) PID=$$"
  # Capture python3 exit code so the trailing echo (which always returns 0)
  # cannot mask a failure and let the dependent merge job (afterok) run on
  # bad shards.
  set +e
  eval python3 "${{_SCRIPT}}" \\
    --task "${{_TASK}}" \\
    --eval-only \\
    --llm-judge \\
    --llm-judge-nodes 0 \\
    --golden "${{SHARD_DIR}}/golden.jsonl" \\
    --predictions "${{SHARD_DIR}}/predictions.jsonl" \\
    --output "${{SHARD_DIR}}/results.json" \\
    ${{_EXTRA}}
  rc=$?
  SHARD_T1=$(date -u +%s)
  echo "[shard ${{SLURM_ARRAY_TASK_ID}}/{num_nodes}] DONE in $((SHARD_T1 - SHARD_T0))s (rc=$rc) — results at ${{SHARD_DIR}}/results.json"
  exit $rc
'
"""
    array_script_path = os.path.join(work_dir, f"_slurm_judge_{output_stem}.sh")
    with open(array_script_path, "w") as f:
        f.write(array_script)

    result = subprocess.run(
        ["sbatch", "--parsable", array_script_path],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"ERROR submitting array: {result.stderr}", file=sys.stderr)
        raise RuntimeError("sbatch (array) failed")
    array_job_id = result.stdout.strip().split(";")[0]
    print(f"\nSubmitted SLURM array {array_job_id} ({num_nodes} tasks)")

    merge_script = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=judge_merge_{output_stem}
#SBATCH --output={work_dir}/logs/judge_merge_{output_stem}_%j.log
#SBATCH --error={work_dir}/logs/judge_merge_{output_stem}_%j.log
#SBATCH --dependency=afterok:{array_job_id}
{extra_sbatch}
set -euo pipefail
export PYTHONUNBUFFERED=1
cd {q_work_dir}
{env_setup}
{conda_exports}python3 -c "
import os, sys
sys.path.insert(0, {work_dir!r})
from slurm_runner import _merge_judge_shards
_merge_judge_shards({shard_root!r}, {output_path!r}, {num_nodes})
"
echo "MERGE DONE"
"""
    merge_script_path = os.path.join(work_dir, f"_slurm_judge_merge_{output_stem}.sh")
    with open(merge_script_path, "w") as f:
        f.write(merge_script)

    result = subprocess.run(
        ["sbatch", "--parsable", merge_script_path],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print(f"ERROR submitting merge: {result.stderr}", file=sys.stderr)
        raise RuntimeError("sbatch (merge) failed")
    merge_job_id = result.stdout.strip()
    print(f"Submitted merge job {merge_job_id} (afterok:{array_job_id})")
    print(
        f"Monitor: sacct -j {array_job_id},{merge_job_id} "
        f"--format=JobID,State,Elapsed -X"
    )
    print(f"Logs: {work_dir}/logs/judge_{output_stem}_{array_job_id}_*.log")
    print(f"      {work_dir}/logs/judge_merge_{output_stem}_{merge_job_id}.log")
    print(f"Final results will be written to: {output_path}")

    return array_job_id, merge_job_id


def add_slurm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--slurm-nodes",
        type=int,
        default=0,
        help="Number of SLURM nodes for distributed generation. 0 = run locally.",
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="", help="SLURM partition name."
    )
    parser.add_argument(
        "--slurm-reservation", type=str, default="", help="SLURM reservation name."
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default="",
        help=(
            "Conda env path with all deps installed (orchestrator + vllm + "
            "starter-kit deps — same env is used locally and activated on "
            "every SLURM worker)."
        ),
    )
    parser.add_argument(
        "--conda-base",
        type=str,
        default="",
        help=(
            "Path to miniconda/anaconda installation (the one that owns "
            "`--conda-env`). Used to source `conda.sh` before activating "
            "the env on a SLURM worker."
        ),
    )
    parser.add_argument(
        "--slurm-gpus", type=int, default=8, help="GPUs per node (default: 8)."
    )
    parser.add_argument(
        "--slurm-time",
        type=str,
        default="4:00:00",
        help="SLURM time limit (default: 4:00:00).",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-node SLURM runner.")
    parser.add_argument("--script", required=True, help="Generation script to run.")
    parser.add_argument(
        "--num-nodes", type=int, required=True, help="Number of SLURM nodes."
    )
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--output", required=True, help="Output prediction JSONL.")
    parser.add_argument("--partition", default="", help="SLURM partition.")
    parser.add_argument("--reservation", default="", help="SLURM reservation.")
    parser.add_argument("--conda-env", default="", help="Conda environment.")
    parser.add_argument(
        "--conda-base",
        default="",
        help="Path to conda installation (for sourcing conda.sh).",
    )
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--time-limit", default="4:00:00")

    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    submit(
        script=args.script,
        input_path=args.input,
        output_path=args.output,
        num_nodes=args.num_nodes,
        extra_args=extra,
        partition=args.partition,
        reservation=args.reservation,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        gpus_per_node=args.gpus_per_node,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    main()
