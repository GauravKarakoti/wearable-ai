#!/usr/bin/env python3
"""Generate Proactive AI predictions.

For each session, processes ~8s video chunks sequentially:
  - Extracts video frames from all chunks up to (and including) the current
    chunk's interval (no future leaking), sampled at 2 fps
  - Builds context from the session-level `query` plus the last
    --max-history-turns turns of `dialog[i]` history (default 4; -1 = all)
  - Generates the assistant response, expected to start with either
    `$interrupt$<utterance>` (model decides to speak) or `$silent$`
    (model decides to stay silent)

Usage:
  python run_generate_proactive.py --video-folder /path/to/videos
"""

from __future__ import annotations

import argparse
import json
import os
import shlex

SYSTEM_PROMPT: str = (
    "You are a proactive AI assistant watching a first-person video of the "
    "user performing a procedural task. The user has issued a single "
    "high-level query. As the video unfolds you observe a series of short "
    "(~8s) chunks; after each chunk you decide whether to speak or stay "
    "silent.\n\n"
    "Output format (single line, no preamble):\n"
    "  - If you should speak: start with the literal token `$interrupt$` "
    "followed by your suggestion or answer in plain text.\n"
    "  - If you should stay silent: output the single literal token "
    "`$silent$` and nothing else.\n\n"
    "Speak when the user asks you something, when an earlier action needs "
    "correction, or when you have useful, timely guidance for the next step. "
    "Stay silent when nothing useful needs to be said."
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize_dialog_turns(
    dialog_at_chunk: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Convert one chunk's dialog history into chat-template messages.

    Each input turn is `{"role": "user"|"assistant", "text": str}`.
    Roles other than user/assistant default to user. Empty turns are skipped.
    """
    history: list[dict[str, str]] = []
    for turn in dialog_at_chunk:
        text = turn.get("text") or ""
        if not text:
            continue
        role = str(turn.get("role", "user")).strip().lower()
        if role not in ("user", "assistant"):
            role = "user"
        history.append({"role": role, "content": str(text)})
    return history


def _run_local(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
    """Run inference locally on this machine and write per-session predictions."""
    from model import create_model, extract_frames, setup_gpus

    backend = getattr(args, "backend", "hf")
    if backend != "vllm":
        setup_gpus(args.num_gpus, args.model_type)

    data = load_jsonl(input_path)
    if args.max_samples is not None:
        data = data[: args.max_samples]

    print(f"Generating predictions for {len(data)} sessions (backend={backend})...")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with (
        create_model(
            args.model_type,
            args.llm_model,
            backend=backend,
            tp_size=getattr(args, "tp", None),
            concurrency=getattr(args, "concurrency", 16),
            max_frames=args.max_frames,
        ) as model,
        open(output_path, "w") as out_f,
    ):
        for i, row in enumerate(data):
            if "video_path" not in row or "video_intervals" not in row:
                print(f"  Skipping row {i}: missing video_path or video_intervals")
                pred: dict[str, object] = {
                    "video_path": row.get("video_path", f"unknown_{i}"),
                    "answers": ["$silent$"] * len(row.get("video_intervals", [])),
                }
                out_f.write(json.dumps(pred) + "\n")
                out_f.flush()
                continue
            video_path = os.path.join(video_folder, str(row["video_path"]))
            intervals: list[list[float]] = row["video_intervals"]
            num_chunks = len(intervals)
            query: str = str(row.get("query", ""))
            dialog: list[list[dict[str, object]]] = row.get("dialog", [])

            all_intervals = [(float(s), float(e)) for s, e in intervals]
            frames_per_interval: list[list[object]] = []
            for interval in all_intervals:
                frames_per_interval.append(
                    extract_frames(
                        video_path,
                        intervals=[interval],
                        frames_per_interval=args.frames_per_interval,
                    )
                )

            generated_answers: list[str] = []
            for j in range(num_chunks):
                # Cumulative frames: all intervals up to and including chunk j.
                frames: list[object] = []
                for k in range(j + 1):
                    frames.extend(frames_per_interval[k])
                if args.max_frames > 0 and len(frames) > args.max_frames:
                    stride = len(frames) / args.max_frames
                    frames = [
                        frames[int(idx * stride)] for idx in range(args.max_frames)
                    ]

                # Take dialog turns AFTER the initial query (which is
                # surfaced separately as the first user message), then
                # slice to the last --max-history-turns turns. 0 = no
                # past turns; -1 = keep all.
                turns_after_query = (
                    dialog[j][1:] if j < len(dialog) and len(dialog[j]) >= 1 else []
                )
                if args.max_history_turns == 0:
                    turns_after_query = []
                elif args.max_history_turns > 0:
                    turns_after_query = turns_after_query[-args.max_history_turns :]
                history = _normalize_dialog_turns(turns_after_query)

                messages: list[dict[str, str]] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]
                if query:
                    messages.append({"role": "user", "content": query})
                messages.extend(history)

                response = model.generate(
                    frames, messages, max_new_tokens=args.max_new_tokens
                )
                generated_answers.append(response)

            pred = {
                "video_path": row["video_path"],
                "answers": generated_answers,
            }
            out_f.write(json.dumps(pred) + "\n")
            out_f.flush()

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i + 1}/{len(data)}")

    print(f"Predictions written to {output_path}")


def _run_eval(
    input_path: str,
    output_path: str,
    eval_output_arg: str | None,
) -> None:
    """Score predictions against golden via `score_proactive` and write a results JSON."""
    if not os.path.exists(output_path):
        print(
            f"Warning: prediction file not found at {output_path}, skipping evaluation."
        )
        return

    from run_evaluation import load_jsonl as load_eval_jsonl, score_proactive

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        raise ValueError(
            f"Evaluation failed: golden has {len(golden)} entries but "
            f"predictions has {len(preds)} (did you use --max-samples?)."
        )

    results = score_proactive(golden, preds)
    eval_output = eval_output_arg
    if not eval_output:
        base = os.path.splitext(os.path.basename(output_path))[0]
        eval_output = os.path.join(
            os.path.dirname(output_path) or ".",
            "..",
            "output",
            f"{base}_results.json",
        )
    eval_output = os.path.normpath(_resolve_path(eval_output))
    os.makedirs(os.path.dirname(eval_output) or ".", exist_ok=True)
    with open(eval_output, "w") as f:
        json.dump(results, f, indent=2)
    overall = results["overall"]
    print(
        f"Proactive Macro F1: {overall['macro_f1']}  G-mean F1: {overall['gmean_f1']}"
    )
    print(f"Results written to {eval_output}")


def _submit_slurm(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
    """Dispatch generation across multiple SLURM nodes via slurm_runner."""
    from slurm_runner import submit

    extra = [
        "--model-type",
        args.model_type,
        "--video-folder",
        video_folder,
        "--max-frames",
        str(args.max_frames),
        "--frames-per-interval",
        str(args.frames_per_interval),
        "--max-history-turns",
        str(args.max_history_turns),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.llm_model:
        extra.extend(["--llm-model", args.llm_model])
    if args.max_samples is not None:
        extra.extend(["--max-samples", str(args.max_samples)])
    if args.num_gpus is not None:
        extra.extend(["--num-gpus", str(args.num_gpus)])
    if getattr(args, "backend", "hf") != "hf":
        extra.extend(["--backend", args.backend])
    if getattr(args, "tp", None) is not None:
        extra.extend(["--tp", str(args.tp)])
    if getattr(args, "concurrency", 16) != 16:
        extra.extend(["--concurrency", str(args.concurrency)])
    post_merge_commands: list[str] = []
    if not args.no_eval:
        if args.eval_output:
            eval_output = args.eval_output
        else:
            base = os.path.splitext(os.path.basename(output_path))[0]
            eval_output = os.path.join(
                os.path.dirname(output_path) or ".",
                "..",
                "output",
                f"{base}_results.json",
            )
        post_merge_commands = [
            "echo 'Running post-merge proactive eval...'",
            (
                "python3 run_evaluation.py --task proactive --eval-only "
                f"--golden {shlex.quote(input_path)} "
                f"--predictions {shlex.quote(output_path)} "
                f"--eval-output {shlex.quote(eval_output)}"
            ),
        ]
    submit(
        script=os.path.abspath(__file__),
        input_path=input_path,
        output_path=output_path,
        num_nodes=args.slurm_nodes,
        extra_args=extra,
        partition=args.slurm_partition,
        reservation=args.slurm_reservation,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        gpus_per_node=args.slurm_gpus,
        time_limit=args.slurm_time,
        post_merge_commands=post_merge_commands,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Proactive AI predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl",
        help=(
            "Input JSONL file (relative to script dir; default points at the "
            "egoproactive split next to starter_kit/ in the HF dataset repo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/egoproactive/predictions.jsonl",
        help="Output prediction JSONL file (relative to script dir).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="../egoproactive/val",
        help=(
            "Folder containing the video files (relative to script dir; "
            "default mirrors the HF repo layout)."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama4",
        choices=["llama4", "qwen"],
        help="Model type to use.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="HuggingFace model ID override (default: per model type).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum total video frames per chunk decision.",
    )
    parser.add_argument(
        "--frames-per-interval",
        type=int,
        default=16,
        help="Frames to sample per video interval (16 = 2 fps over an 8s chunk).",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=4,
        help=(
            "Maximum number of prior dialog turns (after the initial query) "
            "to include in the context. 0 = only the current query, no past "
            "turns. -1 = include all."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Maximum tokens to generate per chunk decision (default: 512). "
            "Typical outputs are `$silent$` (~3 tokens) or "
            "`$interrupt$ <utterance>` (~30-200 tokens). Bump higher if your "
            "model produces longer utterances; lower if you hit OOM."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N sessions (for debugging).",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip automatic evaluation after generation.",
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON.",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: auto-detect).",
    )
    # --- vLLM backend args (mirror run_generate_convqa.py) ---
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' or 'vllm' (default: hf).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size (vllm only).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent HTTP requests (vllm only).",
    )
    from slurm_runner import add_slurm_args

    add_slurm_args(parser)
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    video_folder = _resolve_path(args.video_folder)

    if args.slurm_nodes > 0:
        # SLURM dispatch is asynchronous; the merged output file won't exist
        # until the job completes. Skip the local eval block to avoid silently
        # no-op'ing — the SLURM job is responsible for its own post-merge eval.
        _submit_slurm(args, input_path, output_path, video_folder)
        return

    _run_local(args, input_path, output_path, video_folder)

    if not args.no_eval:
        _run_eval(input_path, output_path, args.eval_output)


if __name__ == "__main__":
    main()
