#!/usr/bin/env python3
"""Generate MCQ predictions for the LongQA dataset.

For each question, feeds the video and MCQ options to the model
and writes the predicted answer letter to the output file.

Usage:
  python run_generate_longqa.py --video-folder /path/to/videos
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

LONGQA_PROMPT_TEMPLATE: str = (
    "Watch the video and answer the following multiple-choice question.\n\n"
    "Question: {question}\n\n"
    "Options:\n{mcq_options}\n\n"
    "Answer with ONLY the single letter of the correct option (A, B, C, or D). "
    "Do not include any other text."
)


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LongQA MCQ predictions.")
    parser.add_argument(
        "--input",
        type=str,
        default="../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl",
        help=(
            "Input JSONL file (relative to script dir; default points at the "
            "egolongqa split next to starter_kit/ in the HF dataset repo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/egolongqa/predictions.jsonl",
        help="Output prediction JSONL file (relative to script dir).",
    )
    parser.add_argument(
        "--video-folder",
        type=str,
        default="../egolongqa/val",
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
        help="Maximum number of frames to sample from each video. Frames are sampled uniformly across the video duration. Higher values give the model more visual context but use more GPU memory. For a 3-min video, 32 frames ≈ 1 frame every 5.6 seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only first N samples (for debugging).",
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
        help="Number of GPUs to use (default: auto-detect).",
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
        help="Output path for evaluation results JSON (default: output/<name>_results.json).",
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
        _run_single(args, all_data, output_path, video_folder)
    else:
        _run_parallel(
            args,
            all_data,
            output_path,
            video_folder,
            num_workers,
            gpus_per_model,
        )

    if not args.no_eval and os.path.exists(output_path):
        _run_longqa_eval(input_path, output_path, args.eval_output)


def _run_longqa_eval(
    input_path: str, output_path: str, eval_output: str | None
) -> None:
    from run_evaluation import evaluate_longqa, load_jsonl as load_eval_jsonl

    golden = load_eval_jsonl(input_path)
    preds = load_eval_jsonl(output_path)
    if len(golden) != len(preds):
        logger.warning(
            "Golden (%d) and predictions (%d) have different lengths",
            len(golden),
            len(preds),
        )
    results = evaluate_longqa(golden, preds)

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

    accuracy = results.get("accuracy", 0.0)
    correct = results.get("correct", 0)
    total = results.get("total", 0)
    print(f"LongQA Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"Results written to {eval_output}")


def _run_single(args: object, data: list, output_path: str, video_folder: str) -> None:
    from model import create_model, DEFAULT_BATCH_SIZES, extract_frames, setup_gpus

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
        f"Generating predictions for {len(data)} samples "
        f"(1 worker, batch_size={batch_size}, backend={backend})..."
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with model, open(output_path, "w") as out_f:
        for batch_start in range(0, len(data), batch_size):
            batch = data[batch_start : batch_start + batch_size]
            batch_frames = [
                extract_frames(
                    os.path.join(video_folder, str(row["video_path"])),
                    max_frames=args.max_frames,
                )
                for row in batch
            ]
            batch_messages = [
                [
                    {
                        "role": "user",
                        "content": LONGQA_PROMPT_TEMPLATE.format(
                            question=row["question"],
                            mcq_options=row["mcq_options"],
                        ),
                    }
                ]
                for row in batch
            ]
            responses = model.generate_batch(
                batch_frames, batch_messages, max_new_tokens=16
            )
            for row, response in zip(batch, responses):
                pred = dict(row)
                pred["mcq_answer"] = response
                out_f.write(json.dumps(pred) + "\n")
                out_f.flush()
            done = min(batch_start + batch_size, len(data))
            print(f"  Progress: {done}/{len(data)}")
    print(f"Predictions written to {output_path}")


def _worker_fn(
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

    from model import create_model, DEFAULT_BATCH_SIZES, extract_frames

    model = create_model(
        args.model_type,
        args.llm_model,
        backend=getattr(args, "backend", "hf"),
        tp_size=getattr(args, "tp", None),
        concurrency=getattr(args, "concurrency", 16),
        max_frames=args.max_frames,
    )
    batch_size = args.batch_size or DEFAULT_BATCH_SIZES.get(args.model_type, 1)

    # `with model:` is required so VLLMModel.__enter__ starts the vllm
    # subprocess and assigns self._port; without it generate_batch fails
    # with "nonnumeric port: 'None'". HF models tolerate no-op __enter__,
    # so the context wrapper is safe for both backends.
    with model, open(out_file, "w") as out_f:
        for batch_start in range(0, len(shard), batch_size):
            batch = shard[batch_start : batch_start + batch_size]
            batch_frames = [
                extract_frames(
                    os.path.join(video_folder, str(row["video_path"])),
                    max_frames=args.max_frames,
                )
                for row in batch
            ]
            batch_messages = [
                [
                    {
                        "role": "user",
                        "content": LONGQA_PROMPT_TEMPLATE.format(
                            question=row["question"],
                            mcq_options=row["mcq_options"],
                        ),
                    }
                ]
                for row in batch
            ]
            responses = model.generate_batch(
                batch_frames, batch_messages, max_new_tokens=16
            )
            for row, response in zip(batch, responses):
                pred = dict(row)
                pred["mcq_answer"] = response
                out_f.write(json.dumps(pred) + "\n")
                out_f.flush()
            done = min(batch_start + batch_size, len(shard))
            print(f"  [Worker {rank}] Progress: {done}/{len(shard)}")
    print(f"  [Worker {rank}] Done: {out_file}")


def _predownload_model(args: object) -> None:
    """Pre-download model weights so workers load from cache."""
    from model import DEFAULT_MODEL_IDS

    model_id = args.llm_model or DEFAULT_MODEL_IDS.get(args.model_type, "")
    if model_id and not os.path.isdir(model_id):
        print(f"Pre-downloading model {model_id}...")
        from transformers import AutoProcessor

        AutoProcessor.from_pretrained(model_id)
        from huggingface_hub import snapshot_download

        snapshot_download(model_id)


def _spawn_workers(
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
            target=_worker_fn,
            args=(rank, gpus_per_model, args, shard, shard_file, video_folder),
        )
        p.start()
        processes.append(p)
    return shard_files, processes


def _join_workers(processes: list) -> None:
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


def _merge_shards(
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
                    "mcq_answer": "",
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


def _run_parallel(
    args: object,
    data: list,
    output_path: str,
    video_folder: str,
    num_workers: int,
    gpus_per_model: int,
) -> None:
    import torch.multiprocessing as mp

    print(
        f"Generating predictions for {len(data)} samples "
        f"({num_workers} workers, {gpus_per_model} GPU(s)/worker)..."
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _predownload_model(args)

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    shard_files, processes = _spawn_workers(
        data, output_path, video_folder, num_workers, gpus_per_model, args
    )
    _join_workers(processes)
    _merge_shards(shard_files, data, output_path, num_workers)
    print(f"Predictions written to {output_path} (merged from {num_workers} shards)")


if __name__ == "__main__":
    main()
