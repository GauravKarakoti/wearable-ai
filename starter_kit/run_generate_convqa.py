#!/usr/bin/env python3
"""Generate conversational QA predictions.

For each conversation, processes turns sequentially:
  - Extracts video frames up to the current turn's interval (no future leaking)
  - Builds multi-turn context from previous Q&A
  - Generates the assistant response

Usage:
  python run_generate_convqa.py --video-folder /path/to/videos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions about a video the user is "
    "watching. Answer directly and concisely based on what you see in the video. "
    'Do not start your response with phrases like "Sure!", "Of course!", '
    '"The answer is", or similar preambles. Just provide the answer.'
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _preextract_conv_frames(
    data: list[dict[str, object]],
    video_folder: str,
    frames_per_interval: int,
    max_frames: int,
) -> list[list[list[object]]]:
    """Pre-extract video frames for all conversations before inference.

    Memory usage is bounded by ``len(data)`` (the number of conversations in
    the current shard/batch).  Workshop datasets are small (< 200 conversations
    per shard), so holding all frames in memory is safe.  Each conversation
    stores at most ``max_frames`` PIL images per turn interval.
    """
    from model import extract_frames

    all_conv_frames: list[list[list[object]]] = []
    for row in data:
        video_path = os.path.join(video_folder, str(row["video_path"]))
        intervals = [(float(s), float(e)) for s, e in row["video_intervals"]]
        conv_frames = [
            extract_frames(
                video_path,
                intervals=[interval],
                frames_per_interval=frames_per_interval,
                max_frames=max_frames,
            )
            for interval in intervals
        ]
        all_conv_frames.append(conv_frames)
    return all_conv_frames


def _build_turn_inputs(
    active: list[tuple[int, dict[str, object], list[list[object]]]],
    batch_answers: list[list[str]],
    turn: int,
    max_frames: int,
) -> tuple[list[list[object]], list[list[dict[str, str]]], list[int]]:
    turn_frames: list[list[object]] = []
    turn_messages: list[list[dict[str, str]]] = []
    active_indices: list[int] = []

    for bi, row, conv_frames in active:
        frames: list[object] = []
        for k in range(min(turn + 1, len(conv_frames))):
            frames.extend(conv_frames[k])
        if len(frames) > max_frames:
            stride = len(frames) / max_frames
            frames = [frames[int(idx * stride)] for idx in range(max_frames)]

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        questions = row["questions"]
        for k in range(turn):
            messages.append({"role": "user", "content": questions[k]})
            messages.append({"role": "assistant", "content": batch_answers[bi][k]})
        messages.append({"role": "user", "content": questions[turn]})

        turn_frames.append(frames)
        turn_messages.append(messages)
        active_indices.append(bi)

    return turn_frames, turn_messages, active_indices


def _run_convqa_eval(
    input_path: str,
    output_path: str,
    llm_model: str | None,
    llm_judge: bool,
    eval_output: str | None,
) -> None:
    from run_evaluation import evaluate_convqa, load_jsonl as load_eval_jsonl

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden (%d) and predictions (%d) have different lengths",
            len(golden),
            len(preds),
        )

    results = evaluate_convqa(
        golden,
        preds,
        run_llm_judge=llm_judge,
        **({"judge_model": llm_model} if llm_model else {}),
    )

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

    print(f"ConvQA BLEU: {results.get('bleu', 'N/A')}")
    if "llm_judge" in results:
        print(f"ConvQA LLM-Judge: {results['llm_judge']}")
    print(f"Results written to {eval_output}")


def _generate_predictions(
    model: object,
    data: list[dict[str, object]],
    all_conv_frames: list[list[list[object]]],
    batch_size: int,
    max_frames: int,
) -> list[list[str]]:
    all_generated: list[list[str]] = [[] for _ in data]

    for batch_start in range(0, len(data), batch_size):
        batch_indices = list(
            range(batch_start, min(batch_start + batch_size, len(data)))
        )
        batch_data = [data[i] for i in batch_indices]
        batch_conv_frames = [all_conv_frames[i] for i in batch_indices]
        batch_answers: list[list[str]] = [[] for _ in batch_indices]

        try:
            batch_max_turns = max(len(row["questions"]) for row in batch_data)
        except KeyError as e:
            bad_ids = [
                row.get("video_uid", f"index-{i}")
                for i, row in enumerate(batch_data)
                if "questions" not in row
            ]
            raise ValueError(f"Input rows missing 'questions' key: {bad_ids}") from e

        for turn in range(batch_max_turns):
            active = [
                (bi, batch_data[bi], batch_conv_frames[bi])
                for bi in range(len(batch_data))
                if turn < len(batch_data[bi]["questions"])
            ]

            if not active:
                break

            turn_frames, turn_messages, active_indices = _build_turn_inputs(
                active, batch_answers, turn, max_frames
            )

            responses = model.generate_batch(
                turn_frames, turn_messages, max_new_tokens=512
            )

            for bi, response in zip(active_indices, responses):
                batch_answers[bi].append(response)

        for bi, idx in enumerate(batch_indices):
            all_generated[idx] = batch_answers[bi]

        done = min(batch_start + batch_size, len(data))
        print(f"  Progress: {done}/{len(data)} conversations")

    return all_generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ConvQA predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../egoconv/wearable_ai_2026_egoconv_val_700.jsonl",
        help=(
            "Input JSONL file (relative to script dir; default points at the "
            "egoconv split next to starter_kit/ in the HF dataset repo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/egoconv/predictions.jsonl",
        help="Output prediction JSONL file (relative to script dir).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="../egoconv/val",
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
        help="Maximum total video frames across all accumulated turns. In ConvQA, frames from earlier turns are carried forward — by turn N, there are up to N × --frames-per-interval frames. This cap downsamples to at most --max-frames by striding. Lower values reduce GPU memory (Scout OOMs above 16 on long conversations).",
    )
    parser.add_argument(
        "--frames-per-interval",
        type=int,
        default=4,
        help="Frames to sample per video interval.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N conversations (for debugging).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (default: auto per model type).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available).",
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
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run LLM-Judge scoring during evaluation (default: BLEU only).",
    )
    # --- vLLM backend args (forwarded from run_evaluation.py via SLURM) ---
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' or 'vllm' (default: hf).",
    )
    parser.add_argument(
        "--tp", type=int, default=None, help="Tensor parallel size (vllm only)."
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
        _submit_slurm(args, input_path, output_path, video_folder)
        return

    _run_local(args, input_path, output_path, video_folder)


def _submit_slurm(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
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
    ]
    if args.llm_model:
        extra.extend(["--llm-model", args.llm_model])
    if args.max_samples:
        extra.extend(["--max-samples", str(args.max_samples)])
    if args.batch_size:
        extra.extend(["--batch-size", str(args.batch_size)])
    if args.num_gpus is not None:
        extra.extend(["--num-gpus", str(args.num_gpus)])
    if getattr(args, "backend", "hf") != "hf":
        extra.extend(["--backend", args.backend])
    if getattr(args, "tp", None) is not None:
        extra.extend(["--tp", str(args.tp)])
    if getattr(args, "concurrency", 16) != 16:
        extra.extend(["--concurrency", str(args.concurrency)])
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
    )


def _run_local(
    args: argparse.Namespace,
    input_path: str,
    output_path: str,
    video_folder: str,
) -> None:
    from model import DEFAULT_GPU_COUNTS, detect_gpu_count

    all_data = load_jsonl(input_path)
    if args.max_samples is not None:
        all_data = all_data[: args.max_samples]

    available = detect_gpu_count()
    num_gpus = args.num_gpus if args.num_gpus is not None else available
    if num_gpus > available:
        raise RuntimeError(
            f"Requested {num_gpus} GPUs but only {available} available. "
            f"Check --num-gpus or CUDA_VISIBLE_DEVICES."
        )
    backend = getattr(args, "backend", "hf")
    # vllm: each worker is an independent server with TP=args.tp (defaults to
    # model.DEFAULT_TP_SIZES[model_type]). Use that as the per-worker GPU count
    # so we can data-parallel across the remaining GPUs on the node.
    if backend == "vllm":
        from model import DEFAULT_TP_SIZES

        gpus_per_model = (
            args.tp
            if getattr(args, "tp", None)
            else DEFAULT_TP_SIZES.get(args.model_type, 1)
        )
    else:
        gpus_per_model = DEFAULT_GPU_COUNTS.get(args.model_type, 1)
        if num_gpus < gpus_per_model:
            raise RuntimeError(
                f"{args.model_type} requires at least {gpus_per_model} GPUs but only "
                f"{num_gpus} available. Allocate more GPUs or choose a smaller model "
                f"(e.g. qwen)."
            )
    num_workers = max(1, num_gpus // gpus_per_model)
    if num_workers <= 1:
        _run_single_convqa(args, all_data, output_path, video_folder)
    else:
        _run_parallel_convqa(
            args,
            all_data,
            output_path,
            video_folder,
            num_workers,
            gpus_per_model,
        )

    if not args.no_eval and os.path.exists(output_path):
        _run_convqa_eval(
            input_path, output_path, args.llm_model, args.llm_judge, args.eval_output
        )


def _run_single_convqa(
    args: object, data: list, output_path: str, video_folder: str
) -> None:
    from model import create_model, DEFAULT_BATCH_SIZES, setup_gpus

    backend = getattr(args, "backend", "hf")
    if backend != "vllm":
        setup_gpus(args.num_gpus, args.model_type)
    model = create_model(
        args.model_type,
        args.llm_model,
        backend=backend,
        tp_size=getattr(args, "tp", None),
        concurrency=getattr(args, "concurrency", 16),
        max_frames=args.max_frames,
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(args.model_type, 1)

    print(
        f"Generating predictions for {len(data)} conversations "
        f"(1 worker, batch_size={batch_size}, backend={backend})..."
    )

    all_conv_frames = _preextract_conv_frames(
        data, video_folder, args.frames_per_interval, args.max_frames
    )
    with model:
        all_generated = _generate_predictions(
            model, data, all_conv_frames, batch_size, args.max_frames
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as out_f:
        for row, answers in zip(data, all_generated):
            pred: dict[str, object] = {
                "video_path": row["video_path"],
                "answers": answers,
            }
            out_f.write(json.dumps(pred) + "\n")
    print(f"Predictions written to {output_path}")


def _convqa_worker_fn(
    rank: int,
    gpus_per_model: int,
    args: object,
    shard: list,
    out_file: str,
    video_folder: str,
) -> None:
    parent_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if parent_cvd:
        visible = [x for x in parent_cvd.split(",") if x.strip()]
        gpu_ids = visible[rank * gpus_per_model : (rank + 1) * gpus_per_model]
    else:
        gpu_ids = [
            str(g) for g in range(rank * gpus_per_model, (rank + 1) * gpus_per_model)
        ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    from model import create_model, DEFAULT_BATCH_SIZES

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=getattr(args, "backend", "hf"),
        tp_size=getattr(args, "tp", None),
        concurrency=getattr(args, "concurrency", 16),
        max_frames=args.max_frames,
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(args.model_type, 1)

    all_conv_frames = _preextract_conv_frames(
        shard, video_folder, args.frames_per_interval, args.max_frames
    )
    # `with model:` is required so VLLMModel.__enter__ starts the vllm
    # subprocess and assigns self._port; without it generate_batch fails
    # with "nonnumeric port: 'None'". HF models tolerate no-op __enter__.
    with model:
        all_generated = _generate_predictions(
            model, shard, all_conv_frames, batch_size, args.max_frames
        )

    with open(out_file, "w") as out_f:
        for row, answers in zip(shard, all_generated):
            pred: dict[str, object] = {
                "video_path": row["video_path"],
                "answers": answers,
            }
            out_f.write(json.dumps(pred) + "\n")
    print(f"  [Worker {rank}] Done: {out_file}")


def _predownload_model_convqa(args: object) -> None:
    """Pre-download model weights so workers load from cache."""
    from model import DEFAULT_MODEL_IDS

    model_id = args.llm_model or DEFAULT_MODEL_IDS.get(args.model_type, "")
    if model_id and not os.path.isdir(model_id):
        print(f"Pre-downloading model {model_id}...")
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(model_id)
        from huggingface_hub import snapshot_download

        snapshot_download(model_id)


def _spawn_convqa_workers(
    data: list,
    output_path: str,
    video_folder: str,
    num_workers: int,
    gpus_per_model: int,
    args: object,
) -> tuple[list[str], list]:
    """Create shards and spawn one worker process per shard."""
    import torch.multiprocessing as mp

    shard_files = []
    processes = []
    for rank in range(num_workers):
        shard = data[rank::num_workers]
        base, ext = os.path.splitext(output_path)
        shard_file = f"{base}.shard{rank}{ext}"
        shard_files.append(shard_file)
        p = mp.Process(
            target=_convqa_worker_fn,
            args=(rank, gpus_per_model, args, shard, shard_file, video_folder),
        )
        p.start()
        processes.append(p)
    return shard_files, processes


def _join_convqa_workers(processes: list) -> None:
    """Wait for all worker processes and terminate any that exceed the timeout."""
    for p in processes:
        p.join(timeout=3600)
        if p.is_alive():
            logger.error(
                "Worker pid=%d still alive after 3600s timeout, terminating", p.pid
            )
            p.terminate()
            p.join(timeout=30)
    failed = [i for i, p in enumerate(processes) if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Workers {failed} failed. Check logs above.")


def _merge_convqa_shards(
    shard_files: list[str],
    data: list,
    output_path: str,
    num_workers: int,
) -> None:
    """Merge per-worker shard files back into a single output in original order."""
    shard_data: dict[int, list[dict[str, object]]] = {}
    for rank, f in enumerate(shard_files):
        try:
            shard_data[rank] = load_jsonl(f)
        except FileNotFoundError:
            logger.warning(
                "Shard file %s not found (worker %d exited 0 but produced no output)",
                f,
                rank,
            )
            shard_data[rank] = []
    missing_count = 0
    with open(output_path, "w") as out_f:
        for idx in range(len(data)):
            rank = idx % num_workers
            shard_idx = idx // num_workers
            if shard_idx < len(shard_data[rank]):
                out_f.write(json.dumps(shard_data[rank][shard_idx]) + "\n")
            else:
                missing_count += 1
                placeholder = {
                    "video_path": data[idx].get("video_path", ""),
                    "answers": [],
                }
                out_f.write(json.dumps(placeholder) + "\n")
                logger.warning(
                    "Missing prediction for sample %d (worker %d, shard_idx %d): "
                    "shard has %d items, expected at least %d — wrote placeholder",
                    idx,
                    rank,
                    shard_idx,
                    len(shard_data[rank]),
                    shard_idx + 1,
                )
    if missing_count > 0:
        logger.warning(
            "Total missing predictions: %d / %d — output may be incomplete",
            missing_count,
            len(data),
        )
    for f in shard_files:
        Path(f).unlink(missing_ok=True)


def _run_parallel_convqa(
    args: object,
    data: list,
    output_path: str,
    video_folder: str,
    num_workers: int,
    gpus_per_model: int,
) -> None:
    import torch.multiprocessing as mp

    print(
        f"Generating predictions for {len(data)} conversations "
        f"({num_workers} workers, {gpus_per_model} GPU(s)/worker)..."
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _predownload_model_convqa(args)

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    shard_files, processes = _spawn_convqa_workers(
        data, output_path, video_folder, num_workers, gpus_per_model, args
    )
    _join_convqa_workers(processes)
    _merge_convqa_shards(shard_files, data, output_path, num_workers)
    print(f"Predictions written to {output_path} (merged from {num_workers} shards)")


if __name__ == "__main__":
    main()
