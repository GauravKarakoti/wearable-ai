---
license: cc-by-nc-4.0
language:
  - en
size_categories:
  - 1K<n<10K
task_categories:
  - visual-question-answering
  - video-text-to-text
tags:
  - egocentric
  - wearable
  - first-person
  - video
  - conversation
  - long-video-qa
  - eccv2026
pretty_name: Wearable AI (ECCV 2026)
configs:
  - config_name: egoconv
    data_files:
      - split: val
        path: egoconv/wearable_ai_2026_egoconv_val_700.jsonl
  - config_name: egolongqa
    data_files:
      - split: val
        path: egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl
  - config_name: egoproactive
    data_files:
      - split: val
        path: egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
---

# Wearable AI Dataset (ECCV 2026)

Part of the [Wearable AI Workshop at ECCV 2026](https://wearable-ai-workshop.github.io/).

A benchmark of egocentric (first-person, head-mounted wearable camera) videos paired with three complementary video question-answering tasks for evaluating wearable-AI assistants on real-world everyday activity videos.

> **▶ Baseline code & evaluation scripts:** see [`starter_kit/README.md`](starter_kit/README.md). The starter kit ships inside this repo, so `git clone` gives you the code and the data together.

## Tasks

**EgoLongQA** — Long-form video question answering with multiple-choice answers. Given a video and a question with four options (A/B/C/D), predict the correct answer.

**EgoConv** — Conversational video question answering. Given a video and a multi-turn conversation, generate free-form answers for each turn. The model sees only the video up to the current turn (no future leaking) and uses its own previous answers as context.

**EgoProactive** — Proactive assistant over streaming egocentric video. Given a video stream, the model must decide *when* to speak and *what* to say at each candidate moment, simulating a wearable AI that volunteers helpful information without being prompted.

## Configurations

| Config | Split | # samples | Description |
| --- | --- | --- | --- |
| `egoconv` | `val` | 700 | Multi-turn conversational QA grounded in egocentric video |
| `egolongqa` | `val` | 700 | Long-video MCQ over single, longer egocentric clips |
| `egoproactive` | `val` | 700 | Proactive-assistant moments over streaming egocentric video |

## Repo Layout

```
facebook/wearable-ai/
├── README.md                   # this file
├── LICENSE                     # CC-BY-NC-4.0 (dataset)
├── egoconv/
│   ├── wearable_ai_2026_egoconv_val_700.jsonl
│   └── val/<id>.mp4            # 700 videos, ~91 GB
├── egolongqa/
│   ├── wearable_ai_2026_egolongqa_val_700.jsonl
│   └── val/<id>.mp4            # 700 videos, ~203 GB
├── egoproactive/
│   ├── wearable_ai_2026_egoproactive_val_700.jsonl
│   └── val/<id>.mp4            # 700 videos, ~23 GB
└── starter_kit/                # baseline code + evaluation scripts
    ├── README.md
    ├── LICENSE                 # CC-BY-NC-4.0 (code)
    ├── run_evaluation.py
    └── ...
```

## Cloning & Disk Usage

The full dataset is **~317 GB** across 2,100 videos. Pick the recipe that matches what you need:

**Full clone** (code + annotations + all videos, ~317 GB):
```bash
git clone https://huggingface.co/datasets/facebook/wearable-ai
```

**Code + annotations only** (~25 MB; videos left as LFS pointers, downloadable on demand):
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/facebook/wearable-ai
```

**One task only** (annotations + the task's videos):
```bash
huggingface-cli download facebook/wearable-ai \
    --repo-type dataset \
    --include "egolongqa/**" "starter_kit/**" "*.md" "LICENSE" \
    --local-dir wearable-ai
```
Per-task video sizes: `egoconv` ≈ 91 GB · `egolongqa` ≈ 203 GB · `egoproactive` ≈ 23 GB.

## Loading

```python
from datasets import load_dataset

egoconv_val = load_dataset("facebook/wearable-ai", "egoconv", split="val")
egolongqa_val = load_dataset("facebook/wearable-ai", "egolongqa", split="val")
egoproactive_val = load_dataset("facebook/wearable-ai", "egoproactive", split="val")
```

## Schemas

### `egoconv` — Conversational QA over egocentric video

| Field | Type | Description |
| --- | --- | --- |
| `video_path` | `str` | Basename of the video file (`<id>.mp4`) |
| `duration_in_sec` | `float` | Video duration in seconds |
| `video_intervals` | `list[[float, float]]` | Time intervals (start, end) covered by the conversation |
| `questions` | `list[str]` | Sequence of user questions over the video |
| `answers` | `list[str]` | Reference answers, aligned to `questions` |
| `task` | `str` | Coarse activity tag (e.g. `Tourism`, `Cooking`) |
| `dialog` | `list[dict]` | Raw turn-by-turn dialog. Each turn: `{text: str, role: "P0"|"P1"|"P2"|"Assistant", start_time: str ("MM:SS"), end_time: str ("MM:SS"), question_type: str}` |

### `egolongqa` — Long-form MCQ over egocentric video

| Field | Type | Description |
| --- | --- | --- |
| `video_path` | `str` | Basename of the video file (`<id>.mp4`) |
| `question` | `str` | Long-form question over the full video |
| `answer` | `str` | Open-ended reference answer |
| `mcq_options` | `str` | Formatted MCQ choices: `A. ... B. ... C. ... D. ...` |
| `mcq_answer` | `str` | Correct option letter (`A`/`B`/`C`/`D`) |
| `category` | `str` | Activity category (e.g. `Travel-Sightseeing (Indoors)`) |

### `egoproactive` — Proactive assistant over streaming egocentric video

| Field | Type | Description |
| --- | --- | --- |
| `video_path` | `str` | Basename of the video file (`<id>.mp4`) |
| `duration_in_sec` | `float` | Video duration in seconds |
| `video_intervals` | `list[[float, float]]` | Candidate decision intervals (start, end) over the streaming video |
| `query` | `str` | The user's initial high-level query (e.g. `"How do I decorate a notebook cover with stickers?"`) |
| `domain` | `str` | Coarse domain tag (e.g. `Arts and Crafts`, `Cooking`) |
| `task` | `str` | Specific task description (e.g. `"Decorating a notebook cover with stickers"`) |
| `answers` | `list[str]` | Reference assistant decisions, one per interval. Each entry is either `"$silent$"` (stay silent) or `"$interrupt$<utterance>"` (speak with the given utterance) |
| `dialog` | `list[list[dict]]` | Cumulative dialog state before each interval. Each turn: `{role: "user"|"assistant", text: str}` |

## Videos

Video files are bundled in this repository under each config's `val/` folder. The `video_path` field in each row is the basename of a `.mp4` file located at:

```
<repo_root>/<config>/val/<video_path>
```

where `<config>` is the config name (`egoconv`, `egolongqa`, or `egoproactive`). The repo holds **2,100 videos** (700 per task, ~317 GB total). Videos are H.265 / 1080p / 15 fps, audio-stripped, and face-blurred. See [Cloning & Disk Usage](#cloning--disk-usage) above for download recipes.

Example: looking up a video locally after downloading the repo snapshot:

```python
import os
from huggingface_hub import snapshot_download

repo_root = snapshot_download("facebook/wearable-ai", repo_type="dataset")
sample = egoconv_val[0]
video_file = os.path.join(repo_root, "egoconv", "val", sample["video_path"])
```

## License

The **dataset** in this repository (videos + JSONL annotations) is released under **CC-BY-NC-4.0** — see [`LICENSE`](LICENSE).

The **starter-kit code** under [`starter_kit/`](starter_kit/) is separately licensed under **CC-BY-NC-4.0** — see [`starter_kit/LICENSE`](starter_kit/LICENSE).

Both licenses permit academic / research use only; commercial use is not granted. Pre-trained model weights downloaded from HuggingFace (e.g., Llama 4 Scout, Llama 4 Maverick, Qwen2.5-VL) are governed by their respective licenses on the model pages.

## Citation

```bibtex
@misc{wearableaiworkshop2026,
  title = {Wearable AI Workshop at ECCV 2026},
  author = {Tuyen (Harry) Tran and Maxim Arap and Seungwhan Moon and Raffay Hamid and Alessandro Suglia and Zsolt Kira and Pascale Fung and Mubarak Shah},
  year = {2026},
  howpublished = {\url{https://wearable-ai-workshop.github.io/}},
  note = {Workshop at the European Conference on Computer Vision (ECCV) 2026}
}
```
