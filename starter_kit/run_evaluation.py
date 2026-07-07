#!/usr/bin/env python3
"""Unified entry point for the ECCV 2026 Wearable AI Workshop starter kit.

Single script for generation and evaluation. Supports three tasks:
  - longqa:     Multiple-choice QA accuracy (A/B/C/D) with lenient answer parsing
  - convqa:     Conversational QA with BLEU + optional LLM-as-judge scoring
  - proactive:  Streaming interrupt/silent classification with Macro F1 and G-mean F1

Usage:
  # Generate + evaluate (default)
  python run_evaluation.py --task longqa --video-folder videos
  python run_evaluation.py --task convqa --video-folder videos --model-type qwen

  # Generate only (test submissions without ground truth)
  python run_evaluation.py --task longqa --video-folder videos --no-eval

  # Evaluate only (existing prediction files)
  python run_evaluation.py --task longqa --eval-only

Setup:
  pip install -r requirements.txt
  huggingface-cli login  # only needed for Llama models
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict

logger: logging.Logger = logging.getLogger(__name__)

# Map from short task name to the HuggingFace config name. All default paths
# (data, video folders, output) are derived from this so the layout mirrors
# the facebook/wearable-ai HF repo, where starter_kit/ ships at the top level
# alongside the per-config data directories: ../egoconv/, ../egolongqa/,
# ../egoproactive/.
_TASK_CONFIG: dict[str, str] = {
    "longqa": "egolongqa",
    "convqa": "egoconv",
    "proactive": "egoproactive",
}


def _default_golden(task: str) -> str:
    cfg = _TASK_CONFIG[task]
    return f"../{cfg}/wearable_ai_2026_{cfg}_val_700.jsonl"


def _default_preds(task: str) -> str:
    return f"output/{_TASK_CONFIG[task]}/predictions.jsonl"


def _default_output(task: str) -> str:
    return f"output/{_TASK_CONFIG[task]}/results.json"


DEFAULT_LONGQA_GOLDEN: str = _default_golden("longqa")
DEFAULT_LONGQA_PREDS: str = _default_preds("longqa")
DEFAULT_CONVQA_GOLDEN: str = _default_golden("convqa")
DEFAULT_CONVQA_PREDS: str = _default_preds("convqa")
DEFAULT_PROACTIVE_GOLDEN: str = _default_golden("proactive")
DEFAULT_PROACTIVE_PREDS: str = _default_preds("proactive")

_TASK_DEFAULTS: dict[str, tuple[str, str, str]] = {
    task: (_default_golden(task), _default_preds(task), _default_output(task))
    for task in _TASK_CONFIG
}

# Default video folder per task (mirrors the HF repo layout: starter_kit/ at
# the top level alongside per-config dirs, so videos are at ../<config>/val).
_DEFAULT_VIDEO_FOLDER: dict[str, str] = {
    task: f"../{cfg}/val" for task, cfg in _TASK_CONFIG.items()
}

# LLM-as-judge model. The leaderboard's official ConvQA judge is Llama-4-Maverick
# FP8, run via vLLM (TP=8 + online fp8 quant) — see the judge backend/quant
# defaults below. Maverick does not fit the `hf` backend, so the default judge
# backend is vllm. Revision is left unpinned (None) because the previous pin was
# Scout-specific; pin a Maverick commit here for stricter reproducibility.
DEFAULT_JUDGE_MODEL: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
DEFAULT_JUDGE_REVISION: str | None = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_path(path: str) -> str:
    """Resolve a path relative to the script directory."""
    if os.path.isabs(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, path)


def load_jsonl(path: str) -> list[dict[str, object]]:
    """Load a JSONL file into a list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_results(output_path: str, results: dict[str, object]) -> str:
    """Write full results and a per_row-free summary; return the summary path."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    summary = {k: v for k, v in results.items() if k != "per_row"}
    base, ext = os.path.splitext(output_path)
    summary_path = f"{base}_summary{ext or '.json'}"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary_path


# ---------------------------------------------------------------------------
# LongQA evaluation — MCQ accuracy
# ---------------------------------------------------------------------------


def normalize_answer(raw: str) -> str:
    """Extract the letter choice (A/B/C/D) from various prediction formats.

    Handles formats like:
      "A", "B.", "(C)", "Option D", "The answer is A.", "answer: B" etc.
    """
    raw = raw.strip()
    if not raw:
        return ""

    upper = raw.upper()

    # Exact single letter
    if upper in ("A", "B", "C", "D"):
        return upper

    # "A.", "A:", "A)"
    m = re.match(r"^([A-Da-d])\s*[\.\:\)]\s*", raw)
    if m:
        return m.group(1).upper()

    # "(A)"
    m = re.match(r"^\(([A-Da-d])\)", raw)
    if m:
        return m.group(1).upper()

    # "Option A", "Answer: B"
    m = re.match(r"^(?:option|answer)\s*[:.]?\s*([A-Da-d])\b", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # "... is A.", "... answer: B"
    m = re.search(r"\b(?:is|answer)\s*[:.]?\s*([A-Da-d])\s*\.?\s*$", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Any standalone A/B/C/D
    m = re.search(r"\b([A-Da-d])\b", raw)
    if m:
        return m.group(1).upper()

    # Last resort: first character if it is A-D
    return upper[0] if upper[0] in "ABCD" else ""


def evaluate_longqa(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
) -> dict[str, object]:
    """Evaluate LongQA MCQ predictions.

    Returns dict with accuracy, per-row results, and category breakdown.

    Raises:
        ValueError: If golden and preds have different lengths.
    """
    if len(golden) != len(preds):
        raise ValueError(
            f"Golden ({len(golden)}) and predictions ({len(preds)}) "
            f"must have the same number of entries."
        )
    results_per_row: list[dict[str, object]] = []
    correct = 0

    for i, (g, p) in enumerate(zip(golden, preds)):
        gold_answer = str(g["mcq_answer"]).strip().upper()
        pred_raw = str(p.get("mcq_answer", ""))
        pred_answer = normalize_answer(pred_raw)
        is_correct = pred_answer == gold_answer

        if is_correct:
            correct += 1

        results_per_row.append(
            {
                "index": i,
                "video_path": g.get("video_path", ""),
                "question": g.get("question", ""),
                "gold_answer": gold_answer,
                "pred_raw": pred_raw,
                "pred_parsed": pred_answer,
                "correct": is_correct,
                "category": g.get("category", ""),
            }
        )

    accuracy = correct / len(preds) if preds else 0.0

    # Per-category breakdown
    category_stats: dict[str, dict[str, int]] = {}
    for r in results_per_row:
        cat = str(r["category"])
        if cat not in category_stats:
            category_stats[cat] = {"correct": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if r["correct"]:
            category_stats[cat]["correct"] += 1

    category_accuracy = {
        cat: round(s["correct"] / s["total"], 4)
        for cat, s in sorted(category_stats.items())
    }

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(preds),
        "category_accuracy": category_accuracy,
        "per_row": results_per_row,
    }


# ---------------------------------------------------------------------------
# ConvQA evaluation — BLEU + LLM-as-judge
# ---------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    """Compute n-gram counts for a token sequence."""
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu(
    reference: list[str],
    hypothesis: list[str],
    max_n: int = 4,
) -> float:
    """BLEU with add-1 smoothing.

    Self-contained implementation — no nltk dependency.
    """
    if not hypothesis:
        return 0.0

    bp = (
        math.exp(1 - len(reference) / len(hypothesis))
        if len(hypothesis) < len(reference)
        else 1.0
    )

    effective_n = min(max_n, len(hypothesis))
    if effective_n == 0:
        return 0.0

    adjusted_weights = tuple(1.0 / effective_n for _ in range(effective_n))

    log_bleu = 0.0
    for n in range(1, effective_n + 1):
        ref_ngrams = _ngrams(reference, n)
        hyp_ngrams = _ngrams(hypothesis, n)
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = len(hypothesis) - n + 1
        # Add-1 smoothing
        precision = (clipped + 1) / (total + 1) if total > 0 else 1e-7
        log_bleu += adjusted_weights[n - 1] * math.log(precision)

    return bp * math.exp(log_bleu)


def compute_bleu_scores(
    golden: list[dict[str, object]], preds: list[dict[str, object]]
) -> list[list[float]]:
    """Compute sentence-level BLEU for each turn in each conversation."""
    all_scores: list[list[float]] = []

    for g, p in zip(golden, preds):
        gold_answers: list[str] = g["answers"]
        pred_answers: list[str] = p.get("answers", [])
        turn_scores: list[float] = []

        for j, gold_ans in enumerate(gold_answers):
            if j < len(pred_answers):
                ref_tokens = gold_ans.lower().split()
                hyp_tokens = pred_answers[j].lower().split()
                score = _sentence_bleu(ref_tokens, hyp_tokens) if hyp_tokens else 0.0
            else:
                score = 0.0
            turn_scores.append(score)

        all_scores.append(turn_scores)

    return all_scores


def _build_judge_prompt(
    question: str, gold_ans: str, pred_ans: str
) -> list[dict[str, str]]:
    """Build the LLM-as-judge prompt for a single turn."""
    return [
        {
            "role": "user",
            "content": (
                "You are an evaluation judge. Given a question, a reference answer, "
                "and a predicted answer, rate the predicted answer.\n\n"
                "Score 1.0 if the predicted answer is correct and captures the key "
                "information.\n"
                "Score 0.5 if the predicted answer is partially correct (some key "
                "info present but incomplete or slightly wrong).\n"
                "Score 0.0 if the predicted answer is wrong or irrelevant.\n\n"
                "Reply with ONLY a single number: 1.0, 0.5, or 0.0\n\n"
                f"Question: {question}\n"
                f"Reference Answer: {gold_ans}\n"
                f"Predicted Answer: {pred_ans}\n\n"
                "Score:"
            ),
        }
    ]


def _parse_judge_score(text: str) -> float:
    """Parse the LLM judge output into 0.0, 0.5, or 1.0."""
    text = text.strip().strip(".")
    for token in text.split():
        token = token.strip(".,;:")
        try:
            val = float(token)
            if val >= 0.75:
                return 1.0
            elif val >= 0.25:
                return 0.5
            else:
                return 0.0
        except ValueError:
            continue
    return 0.0


def _load_judge_model(
    model_id: str,
    revision: str | None = None,
) -> tuple[object, object]:
    """Load processor and model once for LLM-as-judge scoring.

    `model_id` may be an HF Hub id (e.g. ``meta-llama/Llama-4-Scout-…``) or
    an absolute local path (e.g. ``/path/to/checkpoints/Llama-4-Maverick-…``). For local
    paths we drop ``revision`` since git refs are meaningless there and
    transformers raises if the directory isn't a git checkout.

    Kept for the HF backend only. New callers should prefer the vllm-backed
    judge via :func:`compute_llm_judge_scores`, which auto-routes through
    :class:`_VllmJudgeServer` for Maverick-class models.
    """
    import torch
    from transformers import AutoProcessor, Llama4ForConditionalGeneration

    if os.path.isabs(model_id) or os.path.isdir(model_id):
        revision = None

    print(f"Loading judge model: {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = Llama4ForConditionalGeneration.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("Judge model loaded.")
    return processor, model


class _VllmJudgeServer:
    """Minimal vllm server wrapper for text-only LLM-as-judge inference.

    Spawns ``vllm serve <model_id> ...`` as a subprocess, waits for the
    OpenAI-compatible health endpoint, and exposes ``score_turn()`` which
    sends a single chat-completion request and parses the response.

    Designed for Maverick FP8 (or bf16 + ``--quantization fp8`` online
    quantization) on 8×H100 with TP=8 + expert parallel, following
    upstream vLLM recommendations for H100 DGX hardware. Use as a context
    manager so the subprocess is killed on exit.
    """

    def __init__(
        self,
        model_id: str,
        tp_size: int = 8,
        enable_expert_parallel: bool = True,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 32768,
        online_quantization: str | None = None,
        request_timeout: int = 3600,
    ) -> None:
        self.model_id = model_id
        self.tp_size = tp_size
        self.enable_expert_parallel = enable_expert_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.online_quantization = online_quantization
        self.request_timeout = request_timeout
        self._port: int | None = None
        self._log = None
        self._proc = None

    def __enter__(self) -> "_VllmJudgeServer":
        import subprocess
        import tempfile

        from model import find_free_port  # reuse vllm port helper

        self._port = find_free_port()
        log_dir = os.environ.get("VLLM_LOG_DIR", os.getcwd())
        os.makedirs(log_dir, exist_ok=True)
        self._log = tempfile.NamedTemporaryFile(
            mode="w",
            prefix="vllm_judge_",
            suffix=".log",
            delete=False,
            dir=log_dir,
        )
        logger.info("vLLM judge server log: %s", self._log.name)

        server_args = [
            "serve",
            self.model_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--tensor-parallel-size",
            str(self.tp_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--max-model-len",
            str(self.max_model_len),
            "--trust-remote-code",
            "--enforce-eager",
            "--kv-cache-dtype",
            "fp8",
            "--no-enable-prefix-caching",
        ]
        if self.enable_expert_parallel:
            server_args.append("--enable-expert-parallel")
        if self.online_quantization:
            server_args.extend(["--quantization", self.online_quantization])

        # vLLM env vars passed to the subprocess explicitly so they're set
        # even when the orchestrator was started without them.
        env = os.environ.copy()
        env.setdefault("VLLM_USE_V1", "1")
        env.setdefault("LLM_DISABLE_COMPILE_CACHE", "1")
        env.setdefault("VLLM_FLASH_ATTN_VERSION", "3")
        env.setdefault("PYTHONNOUSERSITE", "1")

        import sys

        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main"] + server_args
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=self._log,
            start_new_session=True,
            env=env,
        )

        self._wait_for_health()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._kill_server()

    def _wait_for_health(self) -> None:
        import time
        import urllib.request

        start_ts = time.time()
        deadline = start_ts + self.request_timeout
        last_log_size = 0
        last_heartbeat = start_ts
        logger.info(
            "judge server warming up (timeout %ds, log %s)",
            self.request_timeout,
            self._log.name,
        )
        while time.time() < deadline:
            if self._proc.poll() is not None:
                self._log.flush()
                with open(self._log.name) as f:
                    tail = f.read()[-4000:]
                self._kill_server()
                raise RuntimeError(
                    f"vLLM judge server exited with code {self._proc.returncode}. "
                    f"Log tail:\n{tail}"
                )
            try:
                req = urllib.request.Request(
                    f"http://localhost:{self._port}/health", method="GET"
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(
                            "vLLM judge server ready on port %d (pid %d, warmup took %ds)",
                            self._port,
                            self._proc.pid,
                            int(time.time() - start_ts),
                        )
                        return
            except (OSError, ValueError):
                pass
            try:
                self._log.flush()
                log_size = os.path.getsize(self._log.name)
            except OSError:
                # Log file may briefly disappear during vllm subprocess
                # teardown on Lustre or other network filesystems; skip
                # progress print on this tick.
                log_size = last_log_size
            now = time.time()
            # Heartbeat at least every 30s so silent warmups (Maverick takes
            # ~5 min to load weights and may emit <1 MB during that window)
            # don't look like a hang. Also fire on +1 MiB of new log output.
            if log_size > last_log_size + 1024 * 1024 or now - last_heartbeat >= 30:
                logger.info(
                    "judge server still warming up (elapsed %ds, log %d KiB)",
                    int(now - start_ts),
                    log_size // 1024,
                )
                last_log_size = log_size
                last_heartbeat = now
            time.sleep(5)
        self._kill_server()
        raise RuntimeError(
            f"vLLM judge server did not become healthy within "
            f"{self.request_timeout}s. Log: {self._log.name}"
        )

    def _kill_server(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            import signal
            import subprocess

            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                self._proc.wait(timeout=15)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except OSError:
                    pass

    def score_turn(self, question: str, gold_ans: str, pred_ans: str) -> float:
        """Send a single chat-completion request, return parsed judge score."""
        import json as _json
        import urllib.request

        messages = _build_judge_prompt(question, gold_ans, pred_ans)
        body = _json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": 10,
                "temperature": 0.0,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"http://localhost:{self._port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = _json.loads(resp.read())
        except (OSError, ValueError) as e:
            logger.warning("Judge request failed: %s", e)
            return 0.0
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            logger.warning("Unexpected judge response shape: %s", payload)
            return 0.0
        return _parse_judge_score(text)


def compute_llm_judge_scores(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
    model_id: str,
    batch_size: int,
    backend: str = "hf",
    vllm_tp_size: int = 8,
    vllm_online_quantization: str | None = None,
) -> list[list[float]]:
    """Score each turn using a local LLM as judge.

    Args:
        golden: Golden-set conversations.
        preds: Prediction conversations.
        model_id: HuggingFace model ID or absolute local path for the judge.
        batch_size: Reserved for future batching (current: sequential).
        backend: "hf" (default, Llama4ForConditionalGeneration + device_map=auto)
            or "vllm" (spawns a `vllm serve` subprocess with TP + expert
            parallel — required for Maverick which won't fit in HF on
            8×H100 without aggressive CPU offload).
        vllm_tp_size: tensor parallel size for the vllm judge server
            (default: 8, the full H100 DGX host).
        vllm_online_quantization: pass to `--quantization` so vllm quantizes
            the checkpoint on load. Useful when only a bf16 Maverick is
            available locally (e.g. ``"fp8"``).

    Returns:
        Nested list of scores (one list per conversation, one float per turn).
    """
    if backend == "vllm":
        return _judge_loop_vllm(
            golden,
            preds,
            model_id,
            tp_size=vllm_tp_size,
            online_quantization=vllm_online_quantization,
        )
    return _judge_loop_hf(golden, preds, model_id)


def _judge_loop_hf(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
    model_id: str,
) -> list[list[float]]:
    """Original HF transformers judge loop (Scout-sized models only)."""
    import torch

    revision = DEFAULT_JUDGE_REVISION if model_id == DEFAULT_JUDGE_MODEL else None
    processor, model = _load_judge_model(model_id, revision=revision)

    all_scores: list[list[float]] = []
    total_turns = sum(len(g["answers"]) for g in golden)
    done = 0

    for i, (g, p) in enumerate(zip(golden, preds)):
        gold_answers: list[str] = g["answers"]
        pred_answers: list[str] = p.get("answers", [])
        questions: list[str] = g["questions"]
        turn_scores: list[float] = []

        for j, gold_ans in enumerate(gold_answers):
            question = questions[j] if j < len(questions) else ""
            pred_ans = pred_answers[j] if j < len(pred_answers) else ""

            if not pred_ans.strip():
                turn_scores.append(0.0)
                done += 1
                continue

            messages = _build_judge_prompt(question, gold_ans, pred_ans)

            try:
                input_text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(input_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=10,
                        do_sample=False,
                    )
                new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
                text = processor.decode(new_tokens, skip_special_tokens=True)
                score = _parse_judge_score(text)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("LLM judge error at conv %d turn %d: %s", i, j, e)
                score = 0.0

            turn_scores.append(score)
            done += 1
            if done % 5 == 0:
                print(f"  LLM judge progress: {done}/{total_turns}")

        all_scores.append(turn_scores)

    return all_scores


def _judge_loop_vllm(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
    model_id: str,
    tp_size: int,
    online_quantization: str | None,
) -> list[list[float]]:
    """vLLM-backed judge loop. Spawns one server, scores all turns."""
    server = _VllmJudgeServer(
        model_id,
        tp_size=tp_size,
        online_quantization=online_quantization,
    )
    import time as _time

    all_scores: list[list[float]] = []
    total_turns = sum(len(g["answers"]) for g in golden)
    done = 0
    with server:
        score_start = _time.time()
        last_print = score_start
        print(f"  scoring {total_turns} turns across {len(golden)} conversations...")
        for _i, (g, p) in enumerate(zip(golden, preds)):
            gold_answers: list[str] = g["answers"]
            pred_answers: list[str] = p.get("answers", [])
            questions: list[str] = g["questions"]
            turn_scores: list[float] = []
            for j, gold_ans in enumerate(gold_answers):
                question = questions[j] if j < len(questions) else ""
                pred_ans = pred_answers[j] if j < len(pred_answers) else ""
                if not pred_ans.strip():
                    turn_scores.append(0.0)
                    done += 1
                    continue
                score = server.score_turn(question, gold_ans, pred_ans)
                turn_scores.append(score)
                done += 1
                # Print at most every 5 turns OR every 30 s, whichever
                # comes first — keeps short shards (~66 turns) chatty
                # enough to confirm progress without spamming.
                now = _time.time()
                if done % 5 == 0 or now - last_print >= 30:
                    elapsed = now - score_start
                    avg = elapsed / done if done else 0.0
                    remaining = max(total_turns - done, 0)
                    eta_min = (avg * remaining) / 60.0
                    print(
                        f"  LLM judge progress: {done}/{total_turns} turns "
                        f"({elapsed:.0f}s elapsed, {avg:.1f}s/turn, "
                        f"ETA {eta_min:.1f} min)"
                    )
                    last_print = now
            all_scores.append(turn_scores)
        total_elapsed = _time.time() - score_start
        print(
            f"  scoring done: {total_turns} turns in {total_elapsed:.0f}s "
            f"({total_elapsed / max(total_turns, 1):.1f}s/turn)"
        )
    return all_scores


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_convqa_row(
    i: int,
    g: dict[str, object],
    p: dict[str, object],
    bleu: list[float] | None,
    judge: list[float] | None,
) -> dict[str, object]:
    category = str(g.get("category", g.get("task", "")))
    row: dict[str, object] = {
        "index": i,
        "video_path": g.get("video_path", ""),
        "task": g.get("task", ""),
        "category": category,
        "num_turns": len(g["answers"]),
        "questions": g["questions"],
        "gold_answers": g["answers"],
        "pred_answers": p.get("answers", []),
    }
    if bleu is not None:
        row["bleu_per_turn"] = [round(s, 4) for s in bleu]
        row["bleu_avg"] = round(_safe_mean(bleu), 4)
    if judge is not None:
        row["llm_judge_per_turn"] = judge
        row["llm_judge_avg"] = round(_safe_mean(judge), 4)
    return row


def _compute_category_scores(
    cat_bleu: dict[str, list[float]],
    cat_judge: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    category_scores: dict[str, dict[str, float]] = {}
    all_cats = sorted(set(list(cat_bleu.keys()) + list(cat_judge.keys())))
    for cat in all_cats:
        cat_entry: dict[str, float] = {}
        if cat in cat_bleu:
            cat_entry["bleu"] = round(_safe_mean(cat_bleu[cat]), 4)
        if cat in cat_judge:
            cat_entry["llm_judge"] = round(_safe_mean(cat_judge[cat]), 4)
        category_scores[cat] = cat_entry
    return category_scores


def _aggregate_convqa_output(
    golden: list[dict[str, object]],
    bleu_scores: list[list[float]] | None,
    judge_scores: list[list[float]] | None,
    all_bleu: list[float],
    all_judge: list[float],
    cat_bleu: dict[str, list[float]],
    cat_judge: dict[str, list[float]],
    per_row: list[dict[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}

    if bleu_scores is not None:
        output["bleu"] = round(_safe_mean(all_bleu), 4)
        print(f"BLEU: {_safe_mean(all_bleu):.4f}")

    if judge_scores is not None:
        output["llm_judge"] = round(_safe_mean(all_judge), 4)
        print(f"LLM-Judge: {_safe_mean(all_judge):.4f}")

    output["total_conversations"] = len(golden)
    output["total_turns"] = len(all_bleu) or len(all_judge)

    category_scores = _compute_category_scores(cat_bleu, cat_judge)
    if category_scores:
        output["category_scores"] = category_scores

    output["per_row"] = per_row
    return output


def evaluate_convqa(
    golden: list[dict[str, object]],
    preds: list[dict[str, object]],
    run_bleu: bool = True,
    run_llm_judge: bool = False,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_batch_size: int = 1,
    judge_backend: str = "hf",
    judge_vllm_tp_size: int = 8,
    judge_vllm_online_quantization: str | None = None,
) -> dict[str, object]:
    """Evaluate ConvQA predictions with BLEU and/or LLM-as-judge.

    Returns dict with aggregate scores, per-row breakdown, and category scores.

    Raises:
        ValueError: If golden and preds have different lengths.
    """
    if len(golden) != len(preds):
        raise ValueError(
            f"Golden ({len(golden)}) and predictions ({len(preds)}) "
            f"must have the same number of entries."
        )

    bleu_scores: list[list[float]] | None = None
    judge_scores: list[list[float]] | None = None

    if run_bleu:
        print("Computing BLEU scores...")
        bleu_scores = compute_bleu_scores(golden, preds)

    if run_llm_judge:
        print(f"Computing LLM-Judge scores (backend={judge_backend})...")
        judge_scores = compute_llm_judge_scores(
            golden,
            preds,
            judge_model,
            judge_batch_size,
            backend=judge_backend,
            vllm_tp_size=judge_vllm_tp_size,
            vllm_online_quantization=judge_vllm_online_quantization,
        )

    per_row: list[dict[str, object]] = []
    all_bleu: list[float] = []
    all_judge: list[float] = []
    cat_bleu: dict[str, list[float]] = {}
    cat_judge: dict[str, list[float]] = {}

    for i, (g, p) in enumerate(zip(golden, preds)):
        bleu = bleu_scores[i] if bleu_scores is not None else None
        judge = judge_scores[i] if judge_scores is not None else None
        row = _build_convqa_row(i, g, p, bleu, judge)
        category = str(row["category"])

        if bleu is not None:
            all_bleu.extend(bleu)
            cat_bleu.setdefault(category, []).extend(bleu)
        if judge is not None:
            all_judge.extend(judge)
            cat_judge.setdefault(category, []).extend(judge)

        per_row.append(row)

    return _aggregate_convqa_output(
        golden,
        bleu_scores,
        judge_scores,
        all_bleu,
        all_judge,
        cat_bleu,
        cat_judge,
        per_row,
    )


# ---------------------------------------------------------------------------
# Generation (delegates to run_generate_*.py)
# ---------------------------------------------------------------------------


def _compute_num_workers(
    model_type: str,
    num_gpus: int | None,
    backend: str = "hf",
    tp: int | None = None,
) -> tuple[int, int]:
    """Return (num_workers, gpus_per_model) for the given model type.

    For backend=vllm, gpus_per_model is set from `tp` (or the model's
    DEFAULT_TP_SIZES) so the remaining GPUs on the node can host extra
    independent vllm servers for data-parallel work.
    """
    from model import (
        DEFAULT_GPU_COUNTS,
        DEFAULT_TP_SIZES,
        detect_gpu_count,
        MODEL_REGISTRY,
    )

    if model_type not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model_type {model_type!r}. Known types: {known}")

    detected_gpus = detect_gpu_count() if num_gpus is None else num_gpus
    if backend == "vllm":
        gpus_per_model = tp if tp is not None else DEFAULT_TP_SIZES.get(model_type, 1)
    else:
        gpus_per_model = DEFAULT_GPU_COUNTS.get(model_type, 1)
        if detected_gpus < gpus_per_model:
            raise ValueError(
                f"{model_type} requires {gpus_per_model} GPUs but only "
                f"{detected_gpus} detected"
            )
    num_workers = max(1, detected_gpus // gpus_per_model)
    return num_workers, gpus_per_model


def _generate_longqa_preds(
    input_path: str,
    output_path: str,
    video_folder: str,
    model_type: str,
    llm_model: str | None,
    num_gpus: int | None,
    batch_size: int | None,
    max_frames: int,
    max_samples: int | None,
    backend: str = "hf",
    tp: int | None = None,
    concurrency: int = 16,
) -> None:
    if not video_folder:
        raise ValueError("video_folder is required for LongQA generation")
    from run_generate_longqa import _run_parallel, _run_single

    data = load_jsonl(input_path)
    if max_samples is not None:
        data = data[:max_samples]

    num_workers, gpus_per_model = _compute_num_workers(
        model_type, num_gpus, backend, tp
    )

    gen_args = argparse.Namespace(
        model_type=model_type,
        llm_model=llm_model,
        batch_size=batch_size,
        max_frames=max_frames,
        num_gpus=num_gpus,
        backend=backend,
        tp=tp,
        concurrency=concurrency,
    )

    if num_workers <= 1:
        _run_single(gen_args, data, output_path, video_folder)
    else:
        _run_parallel(
            gen_args, data, output_path, video_folder, num_workers, gpus_per_model
        )


def _generate_convqa_preds(
    input_path: str,
    output_path: str,
    video_folder: str,
    model_type: str,
    llm_model: str | None,
    num_gpus: int | None,
    batch_size: int | None,
    max_frames: int,
    frames_per_interval: int,
    max_samples: int | None,
    backend: str = "hf",
    tp: int | None = None,
    concurrency: int = 16,
) -> None:
    if not video_folder:
        raise ValueError("video_folder is required for ConvQA generation")
    from run_generate_convqa import _run_parallel_convqa, _run_single_convqa

    data = load_jsonl(input_path)
    if max_samples is not None:
        data = data[:max_samples]

    num_workers, gpus_per_model = _compute_num_workers(
        model_type, num_gpus, backend, tp
    )

    gen_args = argparse.Namespace(
        model_type=model_type,
        llm_model=llm_model,
        batch_size=batch_size,
        max_frames=max_frames,
        frames_per_interval=frames_per_interval,
        num_gpus=num_gpus,
        backend=backend,
        tp=tp,
        concurrency=concurrency,
    )

    if num_workers <= 1:
        _run_single_convqa(gen_args, data, output_path, video_folder)
    else:
        _run_parallel_convqa(
            gen_args, data, output_path, video_folder, num_workers, gpus_per_model
        )


def _generate_proactive_preds(
    input_path: str,
    output_path: str,
    video_folder: str,
    model_type: str,
    llm_model: str | None,
    num_gpus: int | None,
    max_frames: int,
    frames_per_interval: int,
    max_history_turns: int,
    max_new_tokens: int,
    max_samples: int | None,
    backend: str = "hf",
    tp: int | None = None,
    concurrency: int = 16,
) -> None:
    """Generate proactive predictions by delegating to run_generate_proactive._run_local."""
    if not video_folder:
        raise ValueError("video_folder is required for Proactive generation")
    from run_generate_proactive import _run_local

    gen_args = argparse.Namespace(
        model_type=model_type,
        llm_model=llm_model,
        max_frames=max_frames,
        frames_per_interval=frames_per_interval,
        max_history_turns=max_history_turns,
        max_new_tokens=max_new_tokens,
        max_samples=max_samples,
        num_gpus=num_gpus,
        backend=backend,
        tp=tp,
        concurrency=concurrency,
    )
    _run_local(gen_args, input_path, output_path, video_folder)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified evaluation for ECCV 2026 Wearable AI Workshop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_evaluation.py --task longqa --video-folder videos\n"
            "  python run_evaluation.py --task convqa --video-folder videos --model-type qwen\n"
            "  python run_evaluation.py --task proactive --eval-only\n"
            "  python run_evaluation.py --task longqa --video-folder videos --no-eval\n"
            "  python run_evaluation.py --task longqa --eval-only\n"
        ),
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["longqa", "convqa", "proactive", "all"],
        required=True,
        help="Which task(s) to evaluate.",
    )

    # --- Single-task file args ---
    parser.add_argument(
        "--golden",
        type=str,
        default=None,
        help="Path to golden JSONL (for --task longqa, convqa, or proactive).",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions JSONL (for --task longqa, convqa, or proactive).",
    )

    # --- Multi-task file args (--task all) ---
    parser.add_argument(
        "--golden-longqa",
        type=str,
        default=None,
        help="Golden JSONL for LongQA (--task all).",
    )
    parser.add_argument(
        "--predictions-longqa",
        type=str,
        default=None,
        help="Predictions JSONL for LongQA (--task all).",
    )
    parser.add_argument(
        "--golden-convqa",
        type=str,
        default=None,
        help="Golden JSONL for ConvQA (--task all).",
    )
    parser.add_argument(
        "--predictions-convqa",
        type=str,
        default=None,
        help="Predictions JSONL for ConvQA (--task all).",
    )
    parser.add_argument(
        "--golden-proactive",
        type=str,
        default=None,
        help="Golden JSONL for Proactive (--task all).",
    )
    parser.add_argument(
        "--predictions-proactive",
        type=str,
        default=None,
        help="Predictions JSONL for Proactive (--task all).",
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Default: output/<config>/results.json (e.g. output/egolongqa/results.json).",
    )

    # --- ConvQA-specific ---
    parser.add_argument(
        "--llm-judge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable LLM-as-judge scoring for ConvQA (default: disabled). Pass --no-llm-judge to explicitly disable.",
    )
    parser.add_argument(
        "--llm-judge-model",
        "--judge-model",  # legacy alias
        dest="llm_judge_model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Judge model — HuggingFace ID (e.g. "
            f"{DEFAULT_JUDGE_MODEL!r}) or absolute local path (e.g. "
            "/path/to/Llama-4-Maverick-17B-128E-Instruct). "
            "When a local path is given, the pinned HF revision is ignored. "
            f"Default: {DEFAULT_JUDGE_MODEL}."
        ),
    )
    parser.add_argument(
        "--llm-judge-backend",
        choices=["hf", "vllm"],
        default="vllm",
        help=(
            "Backend for the LLM judge. `vllm` (default) spawns `vllm serve` with "
            "TP=8 + --enable-expert-parallel — required for the official Maverick "
            "judge, which won't fit in HF on 8×H100 without aggressive CPU offload. "
            "`hf` uses Llama4ForConditionalGeneration with device_map='auto' "
            "(works for Scout-sized models on lighter setups). Defaults to vllm."
        ),
    )
    parser.add_argument(
        "--llm-judge-vllm-tp-size",
        type=int,
        default=8,
        help="Tensor-parallel size for the vllm judge server (default: 8 = full H100 DGX host).",
    )
    parser.add_argument(
        "--llm-judge-vllm-online-quantization",
        type=str,
        default="fp8",
        help=(
            "Pass to vllm's --quantization so it quantizes the checkpoint at "
            "load time (default: fp8, which is how the official Maverick FP8 judge "
            "runs from the bf16 source checkpoint). Pass an empty string ('') to "
            "disable online quantization."
        ),
    )
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=1,
        help="Batch size for LLM judge (default: 1, sequential).",
    )
    parser.add_argument(
        "--llm-judge-nodes",
        type=int,
        default=0,
        help=(
            "Shard `--eval-only --llm-judge` across N SLURM nodes (1 vllm "
            "judge server per node, TP=`--llm-judge-vllm-tp-size`). 0 = run "
            "single-node (default). Submits a sbatch array of size N plus a "
            "dependent merge job that aggregates per-shard `results.json` "
            "into the requested `--output`. Currently supports `--task convqa` only."
        ),
    )

    # --- Generation mode ---
    parser.add_argument(
        "--video-folder",
        type=str,
        default=None,
        help="Folder containing video files. When provided, generates predictions before evaluating.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        default=False,
        help="Skip evaluation after generation (for test submissions without ground truth).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Evaluate existing prediction files only (no generation).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="llama4",
        choices=["llama4", "qwen"],
        help="Model type for generation (default: llama4).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="HuggingFace model ID override (default: per model type).",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for generation (default: auto per model type).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=32,
        help="Maximum video frames to extract per sample.",
    )
    parser.add_argument(
        "--frames-per-interval",
        type=int,
        default=None,
        help=(
            "Frames per video interval. Per-task default if not set: 4 for "
            "ConvQA, 16 for Proactive (= 2 fps over an 8s chunk)."
        ),
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=4,
        help=(
            "Proactive only: max prior dialog turns to keep in the context "
            "(0 = query only, -1 = all). Ignored for other tasks."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Proactive only: max tokens to generate per chunk decision. "
            "Ignored for other tasks."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Process only first N samples (for debugging). "
            "Applied per-shard under --slurm-nodes (so the effective total is "
            "min(N, shard_size) * num_nodes), and as a global cap otherwise."
        ),
    )
    parser.add_argument(
        "--eval-output",
        type=str,
        default=None,
        help="Output path for evaluation results JSON (default: output/<config>/results.json).",
    )

    # --- vLLM backend ---
    parser.add_argument(
        "--backend",
        type=str,
        choices=["hf", "vllm"],
        default="hf",
        help="Inference backend: 'hf' for HuggingFace, 'vllm' for vLLM server (default: hf).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="Tensor parallel size (vllm only, default: auto per model type).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Max concurrent HTTP requests to vLLM server (default: 16).",
    )

    # --- SLURM multi-node (optional — only available on AWS) ---
    # Provide defaults so args.slurm_nodes is always defined even when
    # slurm_runner is not available (e.g. non-AWS environments).
    parser.set_defaults(slurm_nodes=0)
    try:
        from slurm_runner import add_slurm_args

        add_slurm_args(parser)
    except ImportError:
        pass

    return parser


def _filter_subset(
    golden: list[dict],
    preds: list[dict],
    task: str,
) -> tuple[list[dict], list[dict]]:
    """Filter golden and preds to matching composite keys.

    LongQA uses ``(video_path, question)`` as the composite key.
    ConvQA uses ``(video_path, task)`` because ConvQA entries have a ``task``
    field instead of ``question`` (each entry is a multi-turn conversation).

    When predictions are a subset of golden (fewer entries), this filters
    both lists to the intersection so evaluation proceeds on matched pairs.
    Predictions missing ``video_path`` are logged as warnings and dropped.
    Duplicate composite keys in predictions are deduplicated (first wins).
    Returns (filtered_golden, filtered_preds).
    """
    # ConvQA entries use "task" field, LongQA/Proactive use "question"
    is_convqa = task.lower() == "convqa"
    key_field = "task" if is_convqa else "question"

    # Build lookup from golden
    golden_by_key: dict[tuple[str | None, str | None], dict] = {}
    for g in golden:
        key = (g.get("video_path"), g.get(key_field))
        if key in golden_by_key:
            logger.warning(
                "%s: duplicate golden key %s — keeping first, skipping duplicate",
                task,
                key,
            )
            continue
        golden_by_key[key] = g

    filtered_golden: list[dict] = []
    filtered_preds: list[dict] = []
    seen_keys: set[tuple[str | None, str | None]] = set()
    for p in preds:
        vp = p.get("video_path")
        if vp is None:
            logger.warning(
                "%s: prediction missing video_path — skipping: %s",
                task,
                {k: p.get(k) for k in (key_field, "mcq_answer")},
            )
            continue
        key = (vp, p.get(key_field))
        if key in seen_keys:
            logger.warning(
                "%s: duplicate prediction key %s — keeping first, skipping duplicate",
                task,
                key,
            )
            continue
        seen_keys.add(key)
        if key in golden_by_key:
            filtered_golden.append(golden_by_key[key])
            filtered_preds.append(p)

    if not filtered_preds:
        logger.warning(
            "%s: zero predictions matched golden entries — evaluation will be empty",
            task,
        )
    elif len(filtered_preds) * 2 < len(golden):
        logger.warning(
            "%s: only %d/%d predictions matched golden entries (< 50%%) — "
            "check that predictions are from the correct dataset",
            task,
            len(filtered_preds),
            len(golden),
        )

    return filtered_golden, filtered_preds


def _run_longqa(
    golden_path: str,
    preds_path: str,
    output_path: str,
) -> None:
    """Run LongQA evaluation and write results."""
    golden = load_jsonl(golden_path)
    preds = load_jsonl(preds_path)

    if len(golden) != len(preds):
        logger.warning(
            "LongQA golden has %d entries but predictions has %d — "
            "filtering to matched subset",
            len(golden),
            len(preds),
        )
        golden, preds = _filter_subset(golden, preds, "LongQA")
        if not preds:
            raise ValueError(
                "LongQA: zero predictions matched golden entries "
                "-- check that prediction file has correct video_path and question fields"
            )

    print(f"Evaluating LongQA: {len(golden)} samples")
    results = evaluate_longqa(golden, preds)

    summary_path = write_results(output_path, results)

    print(
        f"LongQA Accuracy: {results['accuracy']:.4f} "
        f"({results['correct']}/{results['total']})"
    )
    if results.get("category_accuracy"):
        print("  Per-category accuracy:")
        for cat, acc in results["category_accuracy"].items():
            cat_label = cat if cat else "(no category)"
            print(f"    {cat_label}: {acc:.4f}")
    print(f"Results written to {output_path}")
    print(f"Summary written to {summary_path}")


def _run_convqa(
    golden_path: str,
    preds_path: str,
    output_path: str,
    run_llm_judge: bool,
    judge_model: str,
    judge_batch_size: int,
    judge_backend: str = "hf",
    judge_vllm_tp_size: int = 8,
    judge_vllm_online_quantization: str | None = None,
) -> None:
    """Run ConvQA evaluation and write results."""
    golden = load_jsonl(golden_path)
    preds = load_jsonl(preds_path)

    if len(golden) != len(preds):
        logger.warning(
            "ConvQA golden has %d entries but predictions has %d — "
            "filtering to matched subset",
            len(golden),
            len(preds),
        )
        golden, preds = _filter_subset(golden, preds, "ConvQA")
        if not preds:
            raise ValueError(
                "ConvQA: zero predictions matched golden entries "
                "-- check that prediction file has correct video_path and task fields"
            )

    # Warn on turn-count mismatches
    for i, (g, p) in enumerate(zip(golden, preds)):
        g_answers: list[object] = g.get("answers", [])
        p_answers: list[object] = p.get("answers", [])
        if len(g_answers) != len(p_answers):
            logger.warning(
                "Conversation %d has %d gold turns but %d pred turns.",
                i,
                len(g_answers),
                len(p_answers),
            )

    print(f"Evaluating ConvQA: {len(golden)} conversations")
    results = evaluate_convqa(
        golden,
        preds,
        run_bleu=True,
        run_llm_judge=run_llm_judge,
        judge_model=judge_model,
        judge_batch_size=judge_batch_size,
        judge_backend=judge_backend,
        judge_vllm_tp_size=judge_vllm_tp_size,
        judge_vllm_online_quantization=judge_vllm_online_quantization,
    )

    summary_path = write_results(output_path, results)

    if results.get("category_scores"):
        print("  Per-category scores:")
        for cat, scores in results["category_scores"].items():
            cat_label = cat if cat else "(no category)"
            parts = [f"{k}={v:.4f}" for k, v in scores.items()]
            print(f"    {cat_label}: {', '.join(parts)}")
    print(f"Results written to {output_path}")
    print(f"Summary written to {summary_path}")


# ---------------------------------------------------------------------------
# Proactive evaluation — Macro F1 over $interrupt$ / $silent$ chunks
# ---------------------------------------------------------------------------


def parse_tag(response: str) -> str:
    """Return 'interrupt' if response starts with $interrupt$, else 'silent'."""
    return "interrupt" if response.lstrip().startswith("$interrupt$") else "silent"


def binary_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    """Per-class precision/recall/F1 plus macro F1 and G-mean F1.

    Treats `interrupt` as the positive class. G-mean F1 is the geometric mean
    of the two per-class F1s, which penalises asymmetry between the classes
    more sharply than the (arithmetic) macro F1.
    """
    int_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    int_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    int_f1 = 2 * int_p * int_r / (int_p + int_r) if (int_p + int_r) > 0 else 0.0

    sil_p = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    sil_r = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sil_f1 = 2 * sil_p * sil_r / (sil_p + sil_r) if (sil_p + sil_r) > 0 else 0.0

    macro_f1 = (int_f1 + sil_f1) / 2
    gmean_f1 = math.sqrt(int_f1 * sil_f1)

    return {
        "macro_f1": round(macro_f1, 4),
        "gmean_f1": round(gmean_f1, 4),
        "interrupt_precision": round(int_p, 4),
        "interrupt_recall": round(int_r, 4),
        "interrupt_f1": round(int_f1, 4),
        "silent_precision": round(sil_p, 4),
        "silent_recall": round(sil_r, 4),
        "silent_f1": round(sil_f1, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "support": tp + fp + tn + fn,
    }


def _score_proactive_session(
    g: dict[str, object],
    p: dict[str, object],
    i: int,
) -> tuple[dict[str, int], dict[str, object], int]:
    """Score one proactive session. Returns (counts, per_row_entry, skipped_chunks)."""
    gold_answers: list[str] = g["answers"]  # type: ignore
    pred_answers: list[str] = p.get("answers", [])  # type: ignore
    task: str = str(g.get("task", "unknown"))

    if len(gold_answers) != len(pred_answers):
        logger.warning(
            "Session %d (%s) has %d gold chunks but %d pred chunks; "
            "scoring on the first min(gold,pred) chunks.",
            i,
            g.get("video_path", "?"),
            len(gold_answers),
            len(pred_answers),
        )

    n = min(len(gold_answers), len(pred_answers))
    skipped = abs(len(gold_answers) - len(pred_answers))
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    row_tags: list[dict[str, str]] = []

    for j in range(n):
        gold_tag = parse_tag(gold_answers[j])
        pred_tag = parse_tag(pred_answers[j])
        row_tags.append({"gold": gold_tag, "pred": pred_tag})
        if gold_tag == "interrupt" and pred_tag == "interrupt":
            counts["tp"] += 1
        elif gold_tag == "silent" and pred_tag == "interrupt":
            counts["fp"] += 1
        elif gold_tag == "silent" and pred_tag == "silent":
            counts["tn"] += 1
        elif gold_tag == "interrupt" and pred_tag == "silent":
            counts["fn"] += 1

    per_row_entry = {
        "index": i,
        "video_path": g.get("video_path", ""),
        "task": task,
        "num_chunks": len(gold_answers),
        "tags": row_tags,
    }
    return counts, per_row_entry, skipped


def score_proactive(
    golden: list[dict[str, object]], preds: list[dict[str, object]]
) -> dict[str, object]:
    """Pure scoring: compare per-chunk gold vs pred answers, aggregate metrics.

    Returns a dict with keys:
      - overall: binary_metrics dict for all chunks
      - per_task: {task: binary_metrics dict}
      - total_sessions: int
      - skipped_chunks: int (count of length-mismatch chunks not scored)
      - per_row: list of per-session breakdowns

    Raises ValueError if golden and preds have different lengths.
    """
    if len(golden) != len(preds):
        raise ValueError(
            f"golden has {len(golden)} entries but predictions has {len(preds)}"
        )

    tp = fp = tn = fn = 0
    per_task: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )
    per_row: list[dict[str, object]] = []
    skipped_chunks = 0

    for i, (g, p) in enumerate(zip(golden, preds)):
        counts, row, skipped = _score_proactive_session(g, p, i)
        task = str(row["task"])
        tp += counts["tp"]
        fp += counts["fp"]
        tn += counts["tn"]
        fn += counts["fn"]
        per_task[task]["tp"] += counts["tp"]
        per_task[task]["fp"] += counts["fp"]
        per_task[task]["tn"] += counts["tn"]
        per_task[task]["fn"] += counts["fn"]
        skipped_chunks += skipped
        per_row.append(row)

    overall = binary_metrics(tp, fp, tn, fn)
    per_task_metrics: dict[str, dict[str, float]] = {
        task: binary_metrics(c["tp"], c["fp"], c["tn"], c["fn"])
        for task, c in sorted(per_task.items())
    }

    return {
        "overall": overall,
        "per_task": per_task_metrics,
        "total_sessions": len(golden),
        "skipped_chunks": skipped_chunks,
        "per_row": per_row,
    }


def _print_proactive_summary(results: dict[str, object], output_path: str) -> None:
    """Print a human-readable summary of proactive eval results."""
    overall: dict[str, float] = results["overall"]  # type: ignore
    per_task_metrics: dict[str, dict[str, float]] = results["per_task"]  # type: ignore
    total_sessions: int = results["total_sessions"]  # type: ignore
    skipped_chunks: int = results["skipped_chunks"]  # type: ignore

    print("\nProactive AI — Objective Duplex Metrics")
    print(f"  Sessions:       {total_sessions}")
    print(f"  Chunks scored:  {overall['support']}")
    if skipped_chunks:
        print(f"  Chunks skipped: {skipped_chunks} (length mismatch)")
    print(
        f"  Interrupt:      P={overall['interrupt_precision']:.3f} "
        f"R={overall['interrupt_recall']:.3f} F1={overall['interrupt_f1']:.3f}"
    )
    print(
        f"  Silent:         P={overall['silent_precision']:.3f} "
        f"R={overall['silent_recall']:.3f} F1={overall['silent_f1']:.3f}"
    )
    print(f"  Macro F1:       {overall['macro_f1']:.4f}")
    print(f"  G-mean F1:      {overall['gmean_f1']:.4f}")

    if per_task_metrics:
        print("\nPer-Task (Macro F1 / G-mean F1):")
        for task, m in per_task_metrics.items():
            print(
                f"  {task:20s}  n={m['support']:>5d}  "
                f"macro_f1={m['macro_f1']:.4f}  gmean_f1={m['gmean_f1']:.4f}"
            )

    print(f"\nResults written to {output_path}")


def _run_proactive(
    golden_path: str,
    preds_path: str,
    output_path: str,
) -> None:
    """Run Proactive evaluation and write results."""
    golden = load_jsonl(golden_path)
    preds = load_jsonl(preds_path)

    print(f"Evaluating Proactive: {len(golden)} sessions")
    try:
        results = score_proactive(golden, preds)
    except ValueError as e:
        logger.error("Proactive evaluation failed: %s", e)
        sys.exit(1)

    summary_path = write_results(output_path, results)

    _print_proactive_summary(results, output_path)
    print(f"Summary written to {summary_path}")


def _build_slurm_extra_args(
    args: argparse.Namespace,
    task: str,
    video_folder: str,
) -> list[str]:
    """Build the extra CLI args forwarded to the per-node generate script."""
    extra: list[str] = [
        "--model-type",
        args.model_type,
        "--video-folder",
        video_folder,
        "--max-frames",
        str(args.max_frames),
    ]
    if args.llm_model:
        extra.extend(["--llm-model", args.llm_model])
    if args.batch_size is not None:
        extra.extend(["--batch-size", str(args.batch_size)])
    if args.max_samples is not None:
        extra.extend(["--max-samples", str(args.max_samples)])
    if args.num_gpus is not None:
        extra.extend(["--num-gpus", str(args.num_gpus)])
    if task == "convqa":
        fpi = args.frames_per_interval if args.frames_per_interval is not None else 4
        extra.extend(["--frames-per-interval", str(fpi)])
    elif task == "proactive":
        fpi = args.frames_per_interval if args.frames_per_interval is not None else 16
        extra.extend(
            [
                "--frames-per-interval",
                str(fpi),
                "--max-history-turns",
                str(args.max_history_turns),
                "--max-new-tokens",
                str(args.max_new_tokens),
            ]
        )
    if args.backend != "hf":
        extra.extend(["--backend", args.backend])
    if args.tp is not None:
        extra.extend(["--tp", str(args.tp)])
    if args.concurrency != 16:
        extra.extend(["--concurrency", str(args.concurrency)])
    return extra


def _submit_slurm(args: argparse.Namespace) -> None:
    """Submit a multi-node SLURM job that generates, merges, and evaluates."""
    from slurm_runner import submit

    task = args.task
    if task == "all":
        logger.error(
            "--slurm-nodes does not support --task all. "
            "Run longqa, convqa, and proactive separately."
        )
        sys.exit(1)

    default_g, default_p, _ = _TASK_DEFAULTS[task]
    golden_path = _resolve_path(args.golden or default_g)
    preds_path = _resolve_path(args.predictions or default_p)
    chosen_vf = args.video_folder or _DEFAULT_VIDEO_FOLDER.get(task)
    if chosen_vf is None:
        logger.error("--video-folder is required for SLURM generation.")
        sys.exit(1)
    video_folder = _resolve_path(chosen_vf)
    eval_output = _resolve_path(
        args.eval_output or args.output or _default_output(task)
    )

    generate_script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"run_generate_{task}.py"
    )

    extra_args = _build_slurm_extra_args(args, task, video_folder)

    post_merge: list[str] = []
    if not args.no_eval:
        import shlex

        eval_cmd = (
            f'echo "Running evaluation..."\n'
            f"python3 run_evaluation.py --task {task} --eval-only"
            f" --golden {shlex.quote(golden_path)} --predictions {shlex.quote(preds_path)}"
            f" --eval-output {shlex.quote(eval_output)}"
        )
        if args.llm_judge:
            eval_cmd += " --llm-judge"
            if args.llm_judge_model:
                eval_cmd += f" --llm-judge-model {shlex.quote(args.llm_judge_model)}"
            if args.judge_batch_size > 1:
                eval_cmd += f" --judge-batch-size {args.judge_batch_size}"
        post_merge.append(eval_cmd)

    submit(
        script=generate_script,
        input_path=golden_path,
        output_path=preds_path,
        num_nodes=args.slurm_nodes,
        extra_args=extra_args,
        partition=args.slurm_partition,
        reservation=args.slurm_reservation,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        gpus_per_node=args.slurm_gpus,
        time_limit=args.slurm_time,
        post_merge_commands=post_merge,
    )


def _validate_llm_judge_nodes_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Validate the prerequisites for --llm-judge-nodes > 0."""
    if not args.eval_only:
        parser.error(
            "--llm-judge-nodes requires --eval-only "
            "(generation is not sharded by this flag)."
        )
    if not args.llm_judge:
        parser.error("--llm-judge-nodes requires --llm-judge.")
    if args.task != "convqa":
        parser.error(
            f"--llm-judge-nodes currently supports --task convqa only, "
            f"got --task {args.task}."
        )
    if args.llm_judge_backend != "vllm":
        parser.error(
            "--llm-judge-nodes requires --llm-judge-backend vllm "
            "(sharded scoring is built around 1 vllm judge server per node)."
        )


def _submit_slurm_judge(args: argparse.Namespace) -> None:
    """Submit a sharded LLM-judge scoring job (1 vllm judge per node)."""
    import shlex as _shlex

    from slurm_runner import submit_judge

    task = args.task
    default_g, default_p, default_o = _TASK_DEFAULTS[task]
    golden_path = _resolve_path(args.golden or default_g)
    preds_path = _resolve_path(args.predictions or default_p)
    output_path = _resolve_path(args.eval_output or args.output or default_o)

    extra: list[str] = [
        "--llm-judge-model",
        args.llm_judge_model,
        "--llm-judge-backend",
        args.llm_judge_backend,
        "--llm-judge-vllm-tp-size",
        str(args.llm_judge_vllm_tp_size),
    ]
    if args.llm_judge_vllm_online_quantization:
        extra.extend(
            [
                "--llm-judge-vllm-online-quantization",
                args.llm_judge_vllm_online_quantization,
            ]
        )
    if args.judge_batch_size > 1:
        extra.extend(["--judge-batch-size", str(args.judge_batch_size)])
    del _shlex  # not needed; submit_judge does the quoting

    submit_judge(
        task=task,
        golden_path=golden_path,
        preds_path=preds_path,
        output_path=output_path,
        num_nodes=args.llm_judge_nodes,
        extra_eval_args=extra,
        partition=args.slurm_partition,
        reservation=args.slurm_reservation,
        conda_env=args.conda_env,
        conda_base=args.conda_base,
        gpus_per_node=args.slurm_gpus,
        time_limit=args.slurm_time,
    )


def _run_task(
    task: str,
    args: argparse.Namespace,
    golden_path: str,
    preds_path: str,
    output_path: str,
    video_folder: str | None,
    do_generate: bool,
    run_eval: bool,
) -> None:
    """Run generation and/or evaluation for a single task."""
    if do_generate and video_folder is None:
        raise ValueError("--video-folder is required for generation")
    if do_generate:
        assert video_folder is not None  # narrowed by guard above
    if do_generate and task == "longqa":
        _generate_longqa_preds(
            golden_path,
            preds_path,
            video_folder,
            args.model_type,
            args.llm_model,
            args.num_gpus,
            args.batch_size,
            args.max_frames,
            args.max_samples,
            backend=args.backend,
            tp=args.tp,
            concurrency=args.concurrency,
        )
    elif do_generate and task == "convqa":
        _generate_convqa_preds(
            golden_path,
            preds_path,
            video_folder,
            args.model_type,
            args.llm_model,
            args.num_gpus,
            args.batch_size,
            args.max_frames,
            args.frames_per_interval if args.frames_per_interval is not None else 4,
            args.max_samples,
            backend=args.backend,
            tp=args.tp,
            concurrency=args.concurrency,
        )
    elif do_generate and task == "proactive":
        _generate_proactive_preds(
            golden_path,
            preds_path,
            video_folder,
            args.model_type,
            args.llm_model,
            args.num_gpus,
            args.max_frames,
            args.frames_per_interval if args.frames_per_interval is not None else 16,
            args.max_history_turns,
            args.max_new_tokens,
            args.max_samples,
            backend=args.backend,
            tp=args.tp,
            concurrency=args.concurrency,
        )

    if task == "proactive":
        if run_eval:
            _run_proactive(golden_path, preds_path, output_path)
        return

    if not run_eval:
        return

    use_llm_judge = args.llm_judge
    if task == "longqa":
        _run_longqa(golden_path, preds_path, output_path)
    elif task == "convqa":
        _run_convqa(
            golden_path,
            preds_path,
            output_path,
            run_llm_judge=use_llm_judge,
            judge_model=args.llm_judge_model,
            judge_batch_size=args.judge_batch_size,
            judge_backend=args.llm_judge_backend,
            judge_vllm_tp_size=args.llm_judge_vllm_tp_size,
            judge_vllm_online_quantization=args.llm_judge_vllm_online_quantization,
        )


def _resolve_all_output(
    sub_task: str,
    args: argparse.Namespace,
) -> str:
    """Resolve output path for a sub-task when running --task all."""
    cfg = _TASK_CONFIG[sub_task]
    base = args.eval_output or args.output
    if base:
        dirname = os.path.dirname(base) or "."
        return _resolve_path(os.path.join(dirname, cfg, "results.json"))
    return _resolve_path(f"output/{cfg}/results.json")


def _validate_main_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Validate mutually exclusive flags and emit warnings."""
    if args.no_eval and args.eval_only:
        parser.error("--no-eval and --eval-only are mutually exclusive")

    # vllm spawns in the orchestrator's `sys.executable`, which IS the env
    # activated by `--conda-env` on a SLURM worker (or the interactive env
    # locally). Install vllm into that single env — there is nothing else
    # to wire up.
    if args.backend == "vllm" and args.slurm_nodes == 0:
        # Only the local path can be import-checked from here — SLURM workers
        # will fail fast on the compute node with a clear ImportError if vllm
        # isn't installed in the --conda-env, which is the right place for
        # that error to surface.
        try:
            import vllm  # noqa: F401
        except ImportError:
            parser.error(
                "--backend vllm requires vllm in the current Python environment "
                "(`pip install vllm`); for SLURM runs, install it into your "
                "--conda-env and the workers will pick it up automatically."
            )

    if args.eval_only and args.video_folder is not None:
        logger.warning("--video-folder is ignored when --eval-only is set.")

    if args.eval_only and args.slurm_nodes > 0:
        parser.error(
            "--eval-only cannot be used with --slurm-nodes. "
            "SLURM submission is for generation only. For sharded judge "
            "scoring, use --llm-judge-nodes instead."
        )
    if args.llm_judge_nodes > 0:
        _validate_llm_judge_nodes_args(parser, args)


def _resolve_video_folder(
    args: argparse.Namespace,
    task: str,
) -> tuple[str | None, bool]:
    """Return (resolved_video_folder, do_generate) for the given task.

    Generation is enabled by default for every task with a known default video
    folder unless `--eval-only` is set. An explicit `--video-folder` overrides
    the default.
    """
    if args.eval_only:
        return None, False
    chosen = args.video_folder or _DEFAULT_VIDEO_FOLDER.get(task)
    if chosen is None:
        return None, False
    return _resolve_path(chosen), True


def _run_all_tasks(
    args: argparse.Namespace,
    run_eval: bool,
) -> None:
    """Run longqa, convqa, and proactive tasks in sequence."""
    for sub_task in ("longqa", "convqa", "proactive"):
        default_g, default_p, _ = _TASK_DEFAULTS[sub_task]
        g_attr = getattr(args, f"golden_{sub_task}", None)
        p_attr = getattr(args, f"predictions_{sub_task}", None)
        golden = _resolve_path(g_attr or args.golden or default_g)
        preds = _resolve_path(p_attr or args.predictions or default_p)
        output = _resolve_all_output(sub_task, args)
        video_folder, do_generate = _resolve_video_folder(args, sub_task)

        if sub_task == "proactive" and not os.path.exists(preds):
            print("=" * 60)
            print(f"Task: {sub_task.capitalize()}")
            print("=" * 60)
            logger.warning(
                "Skipping proactive evaluation: prediction file not found at %s",
                preds,
            )
            print()
            continue

        print("=" * 60)
        print(f"Task: {sub_task.capitalize()}")
        print("=" * 60)
        _run_task(
            sub_task,
            args,
            golden,
            preds,
            output,
            video_folder,
            do_generate,
            run_eval,
        )
        print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = _build_parser()
    args = parser.parse_args()
    _validate_main_args(parser, args)

    if args.slurm_nodes > 0:
        _submit_slurm(args)
        return

    if args.llm_judge_nodes > 0:
        _submit_slurm_judge(args)
        return

    task = args.task
    run_eval = not args.no_eval

    if task in _TASK_DEFAULTS:
        default_g, default_p, default_o = _TASK_DEFAULTS[task]
        golden = _resolve_path(args.golden or default_g)
        preds = _resolve_path(args.predictions or default_p)
        output = _resolve_path(args.eval_output or args.output or default_o)
        video_folder, do_generate = _resolve_video_folder(args, task)
        _run_task(
            task, args, golden, preds, output, video_folder, do_generate, run_eval
        )
    elif task == "all":
        _run_all_tasks(args, run_eval)


if __name__ == "__main__":
    main()
