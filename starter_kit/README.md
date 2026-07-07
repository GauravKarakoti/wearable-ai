# ECCV 2026 Wearable AI Workshop - Starter Kit

Part of the [Wearable AI Workshop at ECCV 2026](https://wearable-ai-workshop.github.io/).

This starter kit provides scripts to generate and evaluate predictions on three video understanding benchmarks for wearable AI.

## Table of Contents

- [Tasks](#tasks)
- [Inference Strategy](#inference-strategy)
  - [Frame Sampling Strategy & Resource Limits](#frame-sampling-strategy--resource-limits)
  - [Frame Extraction (Default Baseline)](#frame-extraction-default-baseline)
  - [LongQA](#longqa)
  - [ConvQA](#convqa)
- [File Structure](#file-structure)
- [Setup](#setup)
  - [1. Environment](#1-environment)
  - [2. HuggingFace Access](#2-huggingface-access-only-for-inference--validation-with-the-built-in-hf-models)
  - [3. Data](#3-data)
- [Quick Start](#quick-start)
  - [Lighter hardware (Qwen)](#lighter-hardware-qwen)
  - [Common flags](#common-flags)
  - [Multi-node SLURM](#multi-node-slurm)
- [Using Your Own Model](#using-your-own-model)
- [Data Formats](#data-formats)
  - [LongQA format](#longqa-format)
  - [ConvQA format](#convqa-format)
  - [Proactive format](#proactive-format)
- [Evaluation Metrics](#evaluation-metrics)
- [LLM-as-Judge (ConvQA)](#llm-as-judge-convqa)
  - [Prerequisites](#prerequisites)
  - [Run](#run)
  - [Smoke test](#smoke-test)
  - [Alternative judge models](#alternative-judge-models)
  - [CLI reference](#cli-reference)
- [Supported Models](#supported-models)
- [vLLM Backend](#vllm-backend)
  - [vLLM prerequisites](#vllm-prerequisites)
  - [Usage](#usage)
  - [How It Works](#how-it-works)
  - [Comparison: HuggingFace vs vLLM](#comparison-huggingface-vs-vllm)
  - [Known Limitations](#known-limitations)
- [CLI Options](#cli-options)
- [Parallelism](#parallelism)
- [Hardware Requirements](#hardware-requirements)
- [Workshop](#workshop)
- [License](#license)
- [Citation](#citation)

## Tasks

**LongQA** - Long-form video question answering with multiple-choice answers. Given a video and a question with four options (A/B/C/D), predict the correct answer.

**ConvQA** - Conversational video question answering. Given a video and a multi-turn conversation, generate free-form answers for each turn. The model sees only the video up to the current turn (no future leaking) and uses its own previous answers as context.

**Proactive** - Proactive AI assistant for streaming video. Given a single high-level query (e.g., "Help me make an espresso step by step") and a video segmented into 8s chunks, the model decides at each chunk whether to speak (`$interrupt$<utterance>`) or stay silent (`$silent$`). The model sees only past chunks and the conversation history up to the current chunk (no future leaking), and is scored by macro F1 over the binary `interrupt` / `silent` decision.

> **Naming conventions.** This README and the CLI use short names — `LongQA` / `ConvQA` / `Proactive` for display, and `--task longqa` / `--task convqa` / `--task proactive` for flags. On HuggingFace and in on-disk paths the same datasets appear as `EgoLongQA` / `EgoConv` / `EgoProactive` (configs `egolongqa` / `egoconv` / `egoproactive`).

## Inference Strategy

### Frame Sampling Strategy & Resource Limits

The frame extraction and sampling logic described below is the **default baseline** provided for convenience. Participants are encouraged to develop their own sampling strategies — for example, adaptive frame selection, keyframe detection, temporal attention mechanisms, or task-specific sampling rates — as long as they stay within the resource limits below.

**Resource limits (enforced during leaderboard evaluation):**

| Constraint | Limit |
|------------|-------|
| **Compute budget** | 16 nodes × 8 H100 GPUs (80 GB each) = 128 GPUs total |
| **Per-query timeout** | **300 seconds** (5 minutes) per sample, across all tasks |
| **Inference only** | The timeout covers inference (model forward pass + decoding); data loading and frame extraction are excluded |

A "query" is one sample: a single question for LongQA, a full multi-turn conversation for ConvQA, or a complete streaming session (all chunks) for Proactive. Submissions that exceed the per-query timeout on any sample will receive a null prediction for that sample.

### Frame Extraction (Default Baseline)

Videos are sampled into a fixed number of frames using `model.extract_frames()`:

1. The video is opened with OpenCV and its FPS/duration are read.
2. For each time interval, `--frames-per-interval` frames are sampled uniformly within that interval (default 4 for LongQA/ConvQA, 16 for Proactive — gives ~2 fps over 8s chunks).
3. If the total frame count exceeds `--max-frames` (default 32), frames are downsampled by striding to fit the cap.
4. Frames are converted to PIL Images and passed to the model's vision encoder.

### LongQA

Each sample is independent. The full video is treated as a single interval:

```
Video (3 min) → extract 4 frames uniformly → feed [frames + question + MCQ options] → model outputs "A"/"B"/"C"/"D"
```

### ConvQA

Each conversation has multiple turns with time-aligned video intervals. Turns are processed sequentially because each turn's answer depends on the previous ones:

```
Turn 1: extract frames from interval [0s, 25s]
        feed [frames_turn1 + question_1] → model outputs answer_1

Turn 2: extract frames from interval [27s, 60s]
        feed [frames_turn1 + frames_turn2 + Q1 + A1 + question_2] → model outputs answer_2

Turn N: accumulate all frames from intervals [0..N]
        if total frames > --max-frames: downsample by uniform striding
        feed [capped_frames + full Q&A history + question_N] → model outputs answer_N
```

When the accumulated frame count exceeds `--max-frames`, uniform striding selects every Kth frame to fit the cap. This preserves temporal coverage across the full video timeline but reduces per-turn frame density. For example, at turn 10 with `--max-frames=16` and `--frames-per-interval=4`, 40 accumulated frames are strided to 16 (~1.6 frames per turn instead of 4).

For Llama 4 Scout (17B-active, 109B-total MoE), `--max-frames` above 16 may cause GPU OOM on conversations with 10+ turns under the HF backend. The vLLM backend handles this gracefully via PagedAttention — see [vLLM Backend](#vllm-backend).

## File Structure

The starter kit ships **inside** the HuggingFace dataset repo `facebook/wearable-ai`. After `git clone`, the layout is:

```
wearable-ai/                            # HF dataset repo root
├── README.md                           # dataset card
├── LICENSE                             # CC-BY-NC-4.0 (dataset)
├── egolongqa/
│   ├── wearable_ai_2026_egolongqa_val_700.jsonl
│   └── val/<id>.mp4
├── egoconv/
│   ├── wearable_ai_2026_egoconv_val_700.jsonl
│   └── val/<id>.mp4
├── egoproactive/
│   ├── wearable_ai_2026_egoproactive_val_700.jsonl
│   └── val/<id>.mp4
└── starter_kit/                        # this directory
    ├── README.md
    ├── requirements.txt                # Runtime dependencies
    ├── requirements-dev.txt            # Dev/test dependencies (pytest)
    ├── model.py                        # Model interface (subclass to use your own model)
    ├── run_evaluation.py               # Single entry point: generate + evaluate
    ├── run_generate_longqa.py          # LongQA generation (called by run_evaluation.py)
    ├── run_generate_convqa.py          # ConvQA generation (called by run_evaluation.py)
    ├── run_generate_proactive.py       # Proactive generation (called by run_evaluation.py)
    ├── slurm_runner.py                 # Multi-node SLURM utilities
    ├── tests/                          # Unit tests
    └── output/                         # Generated predictions + eval results (created at runtime)
        ├── egolongqa/{predictions.jsonl, results.json}
        ├── egoconv/{predictions.jsonl, results.json}
        └── egoproactive/{predictions.jsonl, results.json}
```

All starter-kit scripts default to `../<config>/...` for data paths and `output/<config>/...` for predictions. Run them from inside `starter_kit/`.

## Setup

### 1. Environment

```bash
conda create -n wearable_eval python=3.12
conda activate wearable_eval

# Install PyTorch with CUDA support (required for GPU inference)
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
# Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements.txt
```

**Important:** `pip install torch` without `--index-url` installs the CPU-only build on some platforms. Always install PyTorch with CUDA support first using the command from [pytorch.org](https://pytorch.org/get-started/locally/), then install the rest of the requirements.

Verify GPU is available:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

Verified with: Python 3.12, PyTorch 2.6.0+cu124, Transformers 4.57.3, Accelerate 1.13.0, OpenCV 4.13.0, H100 80GB. Also verified with PyTorch 2.10.0+cu128 / Transformers 5.8.1 on H100.

#### Troubleshooting

**`ImportError: cannot import name 'X' from 'transformers'` (or similar from `torch`/`huggingface_hub`)** — your conda env is being shadowed by packages in `~/.local/lib/python*/site-packages/`. Conda includes user-site by default. Either run with user-site disabled:
```bash
PYTHONNOUSERSITE=1 python run_evaluation.py ...
```
or wipe the offending user-site install:
```bash
~/.local/bin/pip uninstall -y torch transformers huggingface_hub
```

**`undefined symbol: cuptiActivityEnableDriverApi, version libcupti.so.12`** — pip-installed PyTorch loads `libcupti.so` via the dynamic linker, and a newer-CUDA PyTorch wheel can land in a conda env on a host where an older system `libcupti` is on `LD_LIBRARY_PATH` (common on multi-CUDA boxes). Prepend the env's bundled CUPTI:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH
```
To make this permanent for the env, drop it into an activate hook:
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh <<'EOF'
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cuda_cupti/lib:$LD_LIBRARY_PATH
EOF
```

### 2. HuggingFace Access (only for inference / validation with the built-in HF models)

You only need this step if you plan to run generation or LLM-as-judge evaluation using the built-in HF backends. For your own model or for `--eval-only` runs that don't invoke the LLM judge, skip this section. (For dataset access, see [3. Data](#3-data) below.)

The default model is [Llama 4 Scout](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct). [Qwen2.5-VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) is also supported out of the box.

For Llama 4 Scout, you need approved access to the `meta-llama` models on HuggingFace:

1. Request access at the model page on huggingface.co
2. Log in locally:
   ```bash
   huggingface-cli login
   ```

Qwen2.5-VL models are publicly available and do not require special access.

### 3. Data

The annotations and videos are hosted on HuggingFace at [`facebook/wearable-ai`](https://huggingface.co/datasets/facebook/wearable-ai). The starter kit ships inside that repo, so `git clone` gives you the code and the data together:

```bash
# Full clone (~317 GB — pulls all videos + annotations + starter_kit)
git clone https://huggingface.co/datasets/facebook/wearable-ai
cd wearable-ai/starter_kit

# Annotations only (~13 MB — videos as LFS pointers; great for development on a laptop)
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/facebook/wearable-ai

# Or, fetch just the annotations + starter kit via the HF CLI
huggingface-cli download facebook/wearable-ai \
  --include "*.jsonl" "starter_kit/*" "README.md" "LICENSE" \
  --local-dir wearable-ai
```

> **Videos:** H.265 / 1080p / 15 fps, audio-stripped, face-blurred. Per-task sizes: EgoConv ≈ 91 GB, EgoLongQA ≈ 203 GB, EgoProactive ≈ 23 GB (≈ 317 GB total).

After `git clone`, all starter-kit script defaults resolve to the sibling per-task directories (`../egolongqa/`, `../egoconv/`, `../egoproactive/`) and the bundled video folders (`../<config>/val/`). Run scripts from inside `starter_kit/`.

> **Annotations-only via `datasets`:** if you only need the JSONL annotations programmatically (no videos, no git), use `datasets.load_dataset("facebook/wearable-ai", "<config>", split="val")` — replace `<config>` with `egolongqa`, `egoconv`, or `egoproactive`.

**Splits.** Only the **val** split (700 samples per task) is published. Every `--task <name>` invocation operates on val. The **test** split is held out and will not be released; leaderboard test scores are produced by the organizers when participants submit their model predictions or inference code.

## Quick Start

After `git clone` and `cd wearable-ai/starter_kit`, `run_evaluation.py` finds the JSONLs and videos at the default per-task paths — no extra flags needed.

> **Hardware:** the canonical Quick Start runs **Llama 4 Scout (17B-16E) with the vLLM backend on 8x H100 GPUs**, using online **FP8** quantization (TP=8). This is the recommended starting point for participants.
>
> If you don't have an 8x H100 node, jump to [Lighter hardware (Qwen)](#lighter-hardware-qwen) below — Qwen 2.5-VL-7B runs on a single GPU.

The canonical Quick Start (Scout + vLLM, default `--max-frames 32`):

```bash
# LongQA — generates + evaluates (MCQ accuracy)
python run_evaluation.py --task longqa --backend vllm

# ConvQA — generates predictions; pair with --llm-judge for full scoring
python run_evaluation.py --task convqa --backend vllm

# Proactive — generates + evaluates (Macro F1)
python run_evaluation.py --task proactive --backend vllm
```

The vLLM backend launches an OpenAI-compatible server with TP=8 and online FP8 quantization at startup. Scout warmup takes ~18–25 minutes per node before inference begins (see [vLLM Backend](#vllm-backend) for details).

### Lighter hardware (Qwen)

Qwen 2.5-VL-7B is a 7B-parameter model that runs on a single GPU. Same commands as above — swap `--model-type qwen` and drop `--backend vllm` if you prefer HuggingFace:

```bash
# LongQA with Qwen on a single GPU
python run_evaluation.py --task longqa --model-type qwen

# ConvQA with Qwen
python run_evaluation.py --task convqa --model-type qwen

# Proactive with Qwen
python run_evaluation.py --task proactive --model-type qwen
```

`run_generate_proactive.py` is still available as a standalone entry point (same flags), but `run_evaluation.py --task proactive` is the recommended unified path.

### Common flags

Override the default video location with `--video-folder <path>`.

Use `--max-samples 5` for a quick sanity check on just 5 examples.

Use `--llm-model <HuggingFace ID>` to override the default model ID for a given model type.

Use `--no-eval` to skip evaluation (e.g., for test submissions without ground truth).

Use `--eval-only` to evaluate existing prediction files without generating:

```bash
# LongQA - MCQ accuracy (CPU only). --golden/--predictions default to the
# sibling paths under ../egolongqa/ and output/egolongqa/.
python run_evaluation.py --task longqa --eval-only

# ConvQA - BLEU only (CPU only)
python run_evaluation.py --task convqa --eval-only --no-llm-judge

# Proactive - Macro F1 (CPU only)
python run_evaluation.py --task proactive --eval-only
```

### Multi-node SLURM

Pass `--slurm-nodes N` to shard generation across N GPU nodes (data is auto-split, predictions are merged, evaluation runs automatically after merge). Requires being on a host with `sbatch` access.

```bash
# LongQA across 8 nodes
python run_evaluation.py --task longqa --model-type qwen \
  --slurm-nodes 8 \
  --slurm-partition <partition> --slurm-reservation <reservation>

# ConvQA across 16 nodes with vLLM backend (Qwen, faster on long conversations)
python run_evaluation.py --task convqa --model-type qwen --backend vllm \
  --slurm-nodes 16 \
  --conda-env <conda env with vllm + starter-kit deps> \
  --conda-base <miniconda/anaconda install prefix> \
  --slurm-partition <partition> --slurm-reservation <reservation>

# Scout (Llama 4) on 4 nodes (8 GPUs/node for model-parallel sharding)
python run_evaluation.py --task longqa --model-type llama4 \
  --slurm-nodes 4 \
  --slurm-partition <partition> --slurm-reservation <reservation>
```

Useful additional flags:
- `--conda-env <path>` / `--conda-base <path>` — conda env (with vllm + all starter-kit deps) to activate on each worker
- `--slurm-gpus <N>` — GPUs per node (defaults to model's typical requirement: 1 for Qwen, 8 for Scout)
- `--slurm-time <hh:mm:ss>` — wall-clock cap
- `--no-eval` — submit generation only, run eval manually later

## Using Your Own Model

The starter kit is designed to make it easy to plug in any model. Subclass `VideoQAModel` in `model.py`:

```python
from model import VideoQAModel

class MyModel(VideoQAModel):
    def __init__(self):
        # Load your model here
        ...

    def generate(self, frames, messages, max_new_tokens=256):
        """
        Args:
            frames: list of PIL.Image.Image (video frames)
            messages: list of {"role": "user"/"assistant"/"system", "content": str}
            max_new_tokens: max tokens to generate

        Returns:
            str: generated text
        """
        # Your inference code here
        ...
```

Then register it in `MODEL_REGISTRY` in `model.py` and use `--model-type <your_key>` with `run_evaluation.py --video-folder videos`.

Alternatively, use `--model-type` to select a built-in model (`llama4` or `qwen`) and `--llm-model` to override the default HuggingFace model ID.

## Data Formats

### LongQA format

Each line in the JSONL file:
```json
{
  "video_path": "9e6f118e71c951d9.mp4",
  "question": "What did the person do after ...?",
  "answer": "The person picked up ...",
  "mcq_options": "A. ... B. ... C. ... D. ...",
  "mcq_answer": "C",
  "category": "Shopping"
}
```

Prediction file: same format, with `mcq_answer` replaced by the model's prediction.

### ConvQA format

Each line in the JSONL file:
```json
{
  "video_path": "4ea836d92de9865f.mp4",
  "duration_in_sec": 174.0,
  "video_intervals": [[0.0, 25.0], [27.0, 60.0], ...],
  "questions": ["What game am I looking at?", "How do you play?", ...],
  "answers": ["You're looking at Tic Tac Toe ...", "Easy to play: ...", ...],
  "task": "Hobbies",
  "dialog": [
    {"text": "What game am I looking at?", "role": "P1", "start_time": "00:08", "end_time": "00:10", "question_type": "Multimodal_relevant"},
    {"text": "You're looking at Tic Tac Toe.", "role": "Assistant", "start_time": "00:11", "end_time": "00:21"},
    {"text": "Cool, how do you play?", "role": "P2", "start_time": "01:05", "end_time": "01:07"}
  ]
}
```

> The `dialog` field captures the **raw turn-by-turn conversation** that the `questions`/`answers` arrays were extracted from. Each turn has `text`, a `role` (`P0`/`P1`/`P2` for participants, `Assistant` for the AI), `start_time`/`end_time` as `"MM:SS"` strings, and an optional `question_type` on user turns (e.g. `Multimodal_relevant`, `Unimodal_relevant`, `Multimodal_irrelevant`, `Unimodal_irrelevant`).

Prediction file:
```json
{
  "video_path": "4ea836d92de9865f.mp4",
  "answers": ["predicted answer 1", "predicted answer 2", ...]
}
```

### Proactive format

Each line in the JSONL file:
```json
{
  "video_path": "98baff001c60c2cc.mp4",
  "duration_in_sec": 91.9,
  "video_intervals": [[0.0, 2.0], [10.4, 18.4], [18.4, 26.4], [42.4, 50.4], [75.9, 83.9], [83.9, 91.9]],
  "query": "How do I create a mosaic coaster with glass gems?",
  "domain": "Arts and Crafts",
  "task": "Creating a mosaic coaster with glass gems",
  "answers": [
    "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!",
    "$interrupt$Time to glue those gems! Dab a bit of craft glue on the back of each one, then press it firmly onto your coaster — stick to your design plan.",
    "$silent$",
    "$silent$",
    "$interrupt$Check each gem's fit and position while the glue dries — make sure they're just right!",
    "$silent$"
  ],
  "dialog": [
    [{"role": "user", "text": "How do I create a mosaic coaster with glass gems?"}],
    [
      {"role": "user", "text": "How do I create a mosaic coaster with glass gems?"},
      {"role": "assistant", "text": "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!"}
    ],
    [
      {"role": "user", "text": "How do I create a mosaic coaster with glass gems?"},
      {"role": "assistant", "text": "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!"},
      {"role": "assistant", "text": "$interrupt$Time to glue those gems! Dab a bit of craft glue on the back of each one, then press it firmly onto your coaster — stick to your design plan."}
    ],
    [
      {"role": "user", "text": "How do I create a mosaic coaster with glass gems?"},
      {"role": "assistant", "text": "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!"},
      {"role": "assistant", "text": "$interrupt$Time to glue those gems! Dab a bit of craft glue on the back of each one, then press it firmly onto your coaster — stick to your design plan."}
    ],
    [
      {"role": "user", "text": "How do I create a mosaic coaster with glass gems?"},
      {"role": "assistant", "text": "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!"},
      {"role": "assistant", "text": "$interrupt$Time to glue those gems! Dab a bit of craft glue on the back of each one, then press it firmly onto your coaster — stick to your design plan."}
    ],
    [
      {"role": "user", "text": "How do I create a mosaic coaster with glass gems?"},
      {"role": "assistant", "text": "$interrupt$Time to grab 5 clear glass gems and lay them out in a fun pattern on your coaster!"},
      {"role": "assistant", "text": "$interrupt$Time to glue those gems! Dab a bit of craft glue on the back of each one, then press it firmly onto your coaster — stick to your design plan."},
      {"role": "assistant", "text": "$interrupt$Check each gem's fit and position while the glue dries — make sure they're just right!"}
    ]
  ]
}
```

`video_intervals`, `answers`, and `dialog` are chunk-aligned (same length). `answers[i]` is either `$silent$` or `$interrupt$<utterance>`. By convention, `answers[0]` is always an `$interrupt$` response to the initial `query`. `dialog[i]` is the conversation history before chunk `i`, so it remains identical across consecutive chunks where the model was `$silent$` (no new assistant turn to append). `domain` is the high-level category (e.g., "Arts and Crafts", "Cooking") and `task` is the specific activity.

Prediction file:
```json
{
  "video_path": "98baff001c60c2cc.mp4",
  "answers": ["$interrupt$predicted utterance", "$silent$", "..."]
}
```

## Evaluation Metrics

| Task | Metric | Description |
|------|--------|-------------|
| LongQA | **MCQ Accuracy** | Exact match on answer letter (A/B/C/D), with lenient parsing |
| ConvQA | **BLEU** | Sentence-level BLEU-4 with smoothing, averaged across all turns |
| ConvQA | **LLM-Judge** | A LLaMA-family judge model rates each answer 1.0 (correct), 0.5 (partial), 0.0 (wrong) — see [LLM-as-Judge](#llm-as-judge-convqa). **The leaderboard requires the official `Llama-4-Maverick-17B-128E-Instruct-FP8` judge** (the code default Scout is for local convenience only) |
| Proactive | **Macro F1** | Per-chunk binary classification (`$interrupt$` vs `$silent$`); arithmetic mean of per-class F1s, with per-task breakdown |
| Proactive | **G-mean F1** | Geometric mean of `interrupt_f1` and `silent_f1`; penalises class asymmetry more sharply than Macro F1 |

## LLM-as-Judge (ConvQA)

> ### ⚠️ Required for the leaderboard (ConvQA / EgoConv)
> EgoConv is **ranked on the LLM-as-Judge metric**. Because the judge is too costly for the organizers to run on every submission, during the **validation phase you compute it yourself and report the score** on the [leaderboard](https://wearable-ai-workshop.github.io/) Submit tab (it is a **required** field for EgoConv and is what appears on the board, badged *self-reported*; organizers compute verified BLEU as a cross-check).
>
> For your score to be comparable to other teams and to the organizers' eventual verified run, **you must use the exact official judge**: `Llama-4-Maverick-17B-128E-Instruct-FP8` via vLLM (the [Run](#run) command below). Take the **mean `llm_judge` score across all turns** (the top-level `llm_judge` field in the results JSON, a value in 0–1) and **add it as one extra line in your `predictions.jsonl`** — alongside the 700 prediction rows, add `{"llm_judge": <score>}` (e.g. `echo '{"llm_judge": 0.83}' >> predictions.jsonl`). That is how the score is submitted (there is no manual entry field). When organizers later run the judge on shared infrastructure, the verified score supersedes your self-report.

ConvQA reports a BLEU score by default and can optionally run a separate **LLM-as-judge** pass that rates each predicted answer against the gold answer on a 0.0 / 0.5 / 1.0 scale. The judge is itself a large multimodal LLM served via vLLM; you point `--llm-judge-model` at any HuggingFace LLaMA-family chat model. **For a leaderboard-valid score, use the official Maverick FP8 judge** (see the callout above and the [Run](#run) command).

This section shows how to run the judge **locally on a single 8x H100 (or equivalent) node** — no orchestrator, no SLURM. It assumes you already produced a ConvQA `predictions.jsonl` (see "Quick Start" above) and want to score it against the gold annotations.

> The default `--llm-judge-model` is the official **Llama-4 Maverick FP8** judge, run via **vLLM** (TP=8 + online fp8 quant) — this is what the leaderboard requires, and the command below uses exactly those defaults. On lighter hardware you can instead pass `--llm-judge-backend hf --llm-judge-model meta-llama/Llama-4-Scout-17B-16E-Instruct` for a local Scout judge, but that score is **not** leaderboard-comparable.

### Prerequisites

- A GPU node with **8x 80 GB GPUs** (Llama-4 Maverick at TP=8 with online FP8 quantization fits in this footprint).
- vLLM in your environment (see the "vLLM Backend" section below).
- Read access to the judge model on HuggingFace (Llama-4 Maverick is gated — request access at https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct first, then run `huggingface-cli login`).

### Run

```bash
# Activate the env that has vllm + all starter-kit deps installed
conda activate <your-env>

# Run the judge on the full ConvQA val set (~40 min wall: ~5-7 min warmup + ~35 min scoring)
python run_evaluation.py \
  --task convqa --eval-only --llm-judge \
  --llm-judge-backend vllm \
  --llm-judge-model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --llm-judge-vllm-tp-size 8 \
  --llm-judge-vllm-online-quantization fp8 \
  --golden ../egoconv/wearable_ai_2026_egoconv_val_700.jsonl \
  --predictions output/egoconv/predictions.jsonl \
  --output output/egoconv/results.json
```

The result file (`output/egoconv/results.json`) gets a top-level `llm_judge` field (mean across all turns) alongside `bleu`, plus `category_scores` for per-category breakdowns. A `results_summary.json` is also written with the same content minus per-row detail.

### Smoke test

Trim the inputs to score in a few minutes instead of ~40:

```bash
head -50 ../egoconv/wearable_ai_2026_egoconv_val_700.jsonl > /tmp/golden_dev.jsonl
head -50 output/egoconv/predictions.jsonl > /tmp/preds_dev.jsonl
python run_evaluation.py --task convqa --eval-only --llm-judge \
  --llm-judge-backend vllm \
  --llm-judge-model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --llm-judge-vllm-tp-size 8 \
  --llm-judge-vllm-online-quantization fp8 \
  --golden /tmp/golden_dev.jsonl \
  --predictions /tmp/preds_dev.jsonl \
  --output /tmp/results_dev.json
```

### Alternative judge models

- **Leaderboard default (`--llm-judge-backend vllm`, Maverick FP8)** — the official judge; what the [Run](#run) command and the on-board score use.
- **Lighter fallback (`--llm-judge-backend hf --llm-judge-model meta-llama/Llama-4-Scout-17B-16E-Instruct`)** — runs Llama-4 Scout via HuggingFace `transformers` with `device_map="auto"`. Fits on 8x H100 without quantization but is significantly slower than vLLM and **not leaderboard-comparable** (different judge model). Useful for local iteration without vLLM.
- **`--llm-judge-model <other-llama4-or-llama3-chat-model>`** — any LLaMA-family chat model works as long as it follows the standard chat template, but only the Maverick FP8 judge is valid for a leaderboard self-report.

### CLI reference

| Flag | Description |
|------|-------------|
| `--llm-judge` | Enable the judge (off by default) |
| `--llm-judge-model` | HuggingFace ID or local checkpoint path (default: `meta-llama/Llama-4-Maverick-17B-128E-Instruct`, the official judge) |
| `--llm-judge-backend {hf,vllm}` | `vllm` (default) = OpenAI-compatible server (required for Maverick); `hf` = transformers (slow, no extra deps, Scout-sized models only) |
| `--llm-judge-vllm-tp-size` | Tensor-parallel size for the vllm judge server (default 8) |
| `--llm-judge-vllm-online-quantization` | Pass to vllm `--quantization` so it quantizes the checkpoint at load time (default `fp8` for the bf16 Maverick checkpoint; `''` to disable) |

## Supported Models

| Model Type | Default Model ID | Min GPUs | GPUs per Model | Notes |
|------------|-----------------|----------|---------------|-------|
| `llama4` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` (bf16) | 8x 80GB | 8 | 17B-active 109B-total MoE; `--backend vllm` recommended (online FP8 quant fits on 8x H100 80GB) — see "vLLM Backend" |
| `qwen` | `Qwen/Qwen2.5-VL-7B-Instruct` | 1x H100/A100 | 1 | 7B dense; data-parallel across available GPUs |

## vLLM Backend

The starter kit supports an optional [vLLM](https://github.com/vllm-project/vllm) backend for faster inference with continuous batching and PagedAttention. This is especially useful for:

- **Scout (Llama 4)**: vLLM is the recommended serving path for Scout. Use the **bf16 source** checkpoint (`meta-llama/Llama-4-Scout-17B-16E-Instruct`); the starter kit auto-applies `--quantization fp8` (online FP8 quantization) at server startup so the model fits on 8x 80 GB H100 without a separate pre-quantized checkpoint.
- **Throughput**: continuous batching + concurrent HTTP requests improve inference speed
- **Flexibility**: participants can use either HuggingFace or vLLM without changing their model code

### vLLM prerequisites

Install vLLM into your environment (on a GPU node):

```bash
pip install vllm==0.19.1
```

Verify:
```bash
python -c "import vllm; print(vllm.__version__)"
```

### Usage

Add `--backend vllm` to any generation command:

```bash
# LongQA with Qwen via vLLM
python run_evaluation.py --task longqa --video-folder videos --model-type qwen \
  --backend vllm

# ConvQA with Qwen via vLLM
python run_evaluation.py --task convqa --video-folder videos --model-type qwen \
  --backend vllm
```

The vLLM backend automatically:
1. Finds a free port and launches a vLLM OpenAI-compatible server
2. Waits for the server to be healthy (up to 3600 s — Scout's online FP8 quantization + MoE init can take ~20 min on 8x H100; a 30-second heartbeat prints `vLLM server still warming up (elapsed Ns, log KiB)` so the loop is visibly progressing)
3. Sends concurrent HTTP requests for inference
4. Kills the server when done

For Scout specifically, the server is launched with these auto-applied flags (no user intervention needed):

- `--quantization fp8` — online FP8 quant from the bf16 source weights (avoids the NVIDIA pre-quantized FP8 + ModelOpt loader path, which hangs on this stack)
- `--max-model-len 131072` — caps Scout's default 10 M-token context so KV cache fits on 8x80GB (default would need ~60 GiB/GPU and crashes engine init)
- `--enforce-eager` — skips CUDA-graph capture, saves several GB peak GPU memory
- `--tensor-parallel-size 8` (configurable via `--tp`)
- `--trust-remote-code`

Total Scout warmup on 8x H100 typically lands around **18–25 minutes** (load is <1 s; the slow phase is online FP8 quant + MoE expert init). Inference itself is fast (~3000 tokens/s prompt throughput with prefix caching).

### How It Works

When `--backend vllm` is selected, the starter kit skips the multi-worker `torch.multiprocessing` pattern entirely. Instead:

- **One vLLM server per node** uses all GPUs via tensor parallelism (TP=8 for Scout, TP=1 for Qwen)
- **Concurrency comes from HTTP requests**, not process-level parallelism — a ThreadPoolExecutor sends up to 16 (configurable via `--concurrency`) concurrent requests
- **Prefix caching** is enabled by default — in ConvQA multi-turn conversations, earlier turns' KV cache entries are reused automatically

### Comparison: HuggingFace vs vLLM

| Aspect | HuggingFace (`--backend hf`) | vLLM (`--backend vllm`) |
|--------|------------------------------|-------------------------|
| **Setup** | `pip install -r requirements.txt` | `pip install vllm==0.19.1` (same env) |
| **Memory** | Manual `device_map="auto"` sharding | PagedAttention, dynamic memory |
| **Throughput** | Sequential or multi-process | Continuous batching + concurrent HTTP |
| **Scout ConvQA** | May OOM on long conversations | PagedAttention handles gracefully |
| **Multi-turn caching** | No KV reuse across turns | Prefix caching reuses KV automatically |
| **Custom models** | Full Python API control | Must be vLLM-compatible |

### Known Limitations

- vLLM backend verified with both Qwen2.5-VL and Llama 4 Scout (via online FP8 quantization from the bf16 source checkpoint). The NVIDIA pre-quantized FP8 + ModelOpt loader path is not supported.
- The vLLM server startup time depends on model: Qwen2.5-VL = 1-3 min, Scout = 18-25 min (the online FP8 quant + MoE expert init is CPU-bound and slow). A 30-second heartbeat keeps the warmup visible.
- vLLM 0.19.1 is the recommended version (tested with PyTorch 2.10+cu128)

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | required | Task to run: `longqa`, `convqa`, `proactive`, or `all` (run_evaluation.py) |
| `--video-folder` | — | Folder containing video files (enables generation) |
| `--no-eval` | false | Skip evaluation after generation |
| `--eval-only` | false | Evaluate existing predictions only (no generation) |
| `--model-type` | `llama4` | Model type (`llama4` or `qwen`) |
| `--llm-model` | per model type | HuggingFace model ID override |
| `--num-gpus` | all available | Number of GPUs to use |
| `--max-samples` | all | Process only first N samples (for debugging) |
| `--max-frames` | 32 | Maximum video frames to extract |
| `--llm-judge` | false | Enable LLM-as-judge scoring for ConvQA |
| `--eval-output` | `output/<config>/results.json` (e.g. `output/egolongqa/results.json`) | Output path for evaluation results |
| `--backend` | `hf` | Inference backend: `hf` or `vllm` |
| `--tp` | auto per model | Tensor parallel size (vllm only) |
| `--concurrency` | 16 | Max concurrent HTTP requests (vllm only) |
| `--conda-env` | — | Conda env (vllm + all deps) to activate on each SLURM worker |
| `--conda-base` | — | Path to miniconda/anaconda installation that owns `--conda-env` |

**Proactive-only flags** (apply when `--task proactive`):

| Flag | Default | Description |
|------|---------|-------------|
| `--frames-per-interval` | 16 for proactive (4 for longqa/convqa) | Frames sampled per video interval; 16 = ~2 fps over an 8s chunk |
| `--max-history-turns` | 4 | Max prior dialog turns to include (0 = query only, -1 = all) |
| `--max-new-tokens` | 512 | Maximum tokens to generate per chunk decision |

`run_generate_proactive.py` exposes the same flags as a standalone entry — see `python run_generate_proactive.py --help`.

## Parallelism

**Single node (default):** GPU parallelism is automatic based on available GPUs:

- **Large models** (Llama 4 Scout, 17B-active 109B-total MoE): all GPUs are used for model sharding. 1 worker processes samples sequentially. Minimum 8x H100/A100 80GB required.
- **Small models** (Qwen2.5-VL-7B): each GPU runs an independent model instance. N GPUs → N workers, each processing 1/N of the data in parallel.

**Multi-node (SLURM):** Use `--slurm-nodes N` to distribute generation across N nodes. Data is automatically split into N shards, each node processes its shard, and predictions are merged after all nodes finish. Evaluation runs automatically after merging unless `--no-eval` is passed.

```bash
python run_evaluation.py --task longqa --video-folder videos --slurm-nodes 8 \
  --slurm-partition <your_partition> --slurm-reservation <your_reservation>
```

## Hardware Requirements

| Task | Llama 4 Scout | Qwen2.5-VL-7B |
|------|---------------|----------------|
| Generation | 8x H100/A100 80GB (model sharding) | 1x H100/A100 (data-parallel on all available) |
| ConvQA LLM-Judge | 8x H100/A100 80GB (Scout as judge) | N/A |
| LongQA MCQ eval | CPU only | CPU only |
| ConvQA BLEU eval | CPU only | CPU only |

## Workshop

For workshop details — schedule, invited speakers, submission instructions, evaluation metrics, deadlines, and contact / Q&A — see the workshop website:

**https://wearable-ai-workshop.github.io/**

## License

The dataset and the starter-kit code in this repository are both released under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC-4.0)** — research / academic use only.

- Dataset (jsonl annotations + video files): see the top-level [`LICENSE`](../LICENSE) in the `facebook/wearable-ai` repository.
- Starter-kit code (this directory): see [`LICENSE`](LICENSE) here.

Pre-trained model weights downloaded from HuggingFace (e.g., Llama 4 Scout, Llama 4 Maverick, Qwen2.5-VL) are governed by their respective licenses on the model pages — refer to each model's HuggingFace card before use.

## Citation

If you use this starter kit or the associated benchmarks in your research, please cite:

```bibtex
@misc{wearableaiworkshop2026,
  title = {Wearable AI Workshop at ECCV 2026},
  author = {Tuyen (Harry) Tran and Maxim Arap and Seungwhan Moon and Raffay Hamid and Alessandro Suglia and Zsolt Kira and Pascale Fung and Mubarak Shah},
  year = {2026},
  howpublished = {\url{https://wearable-ai-workshop.github.io/}},
  note = {Workshop at the European Conference on Computer Vision (ECCV) 2026}
}
```
