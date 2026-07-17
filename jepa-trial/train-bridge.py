"""
train-bridge.py -- scaled up for the full ~700-video run.

WHAT CHANGED FROM THE HF-DOWNLOAD VERSION:
  1. Videos are read from your local hard drive (HARD_DRIVE_VIDEO_DIR),
     not downloaded per-run -- use download.py separately to
     populate the drive first.
  2. NEW_BATCH is no longer hand-typed -- entries are pulled directly from
     your manifest jsonl, filtered to whatever's on disk and not yet
     embedded. Manually pasting 600 dict entries was never going to scale.
  3. Training now uses REAL minibatches (multiple examples per forward/
     backward pass, padded together) instead of one-example-at-a-time --
     this is the single biggest lever for making 700-video epochs
     tractable in your remaining time. Tune ML_BATCH_SIZE to your GPU's
     memory headroom; start at 8 and back off if you hit OOM.
  4. A fixed held-out validation set is chosen ONCE and saved to disk
     (holdout_ids.json), so loss/accuracy is comparable across sessions
     instead of being re-randomized every run.
  5. Training now stops on whichever comes first: avg_loss drops below
     TARGET_LOSS, or MAX_TIME_BUDGET_SECONDS is hit. The old fixed
     9-11 min cap wasn't enough once the library grew past ~60 videos.

WORKFLOW:
  1. download.py to populate HARD_DRIVE_VIDEO_DIR (separate script)
  2. Run this script -- it embeds whatever's on disk + not yet in the
     library (up to BATCH_SIZE new videos), then trains on the full
     accumulated library with minibatching until TARGET_LOSS or the time
     cap is hit.
  3. Re-run as often as you like -- checkpoint and embeddings persist.
"""

import os
import json
import glob
import random
import time
import torch
import torch.nn as nn

from transformers import AutoModel, AutoVideoProcessor, AutoModelForCausalLM, AutoTokenizer
from trial import load_video_frames

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_VIDEO_TOKENS = 16

MANIFEST_PATH = "train.jsonl"
HARD_DRIVE_VIDEO_DIR = "D:/wearable_ai_videos/egolongqa/val"

BATCH_SIZE = 100
ML_BATCH_SIZE = 8
MAX_NEW_TOKENS = 256
TARGET_LOSS = 0.6
MAX_TIME_BUDGET_SECONDS = 90 * 60
LEARNING_RATE = 1e-4
HELD_OUT_COUNT = 10
MIN_LIBRARY_SIZE_FOR_HOLDOUT = 20

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
HOLDOUT_IDS_PATH = os.path.join(OUTPUT_DIR, "holdout_ids.json")
os.makedirs(EMBEDDING_LIBRARY_DIR, exist_ok=True)


def library_path(video_id):
    return os.path.join(EMBEDDING_LIBRARY_DIR, f"{video_id}.pt")


def load_manifest():
    rows = []
    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["video_id"] = os.path.splitext(row["video_path"])[0]
                rows.append(row)
    return rows


def local_video_path(video_id):
    path = os.path.join(HARD_DRIVE_VIDEO_DIR, f"{video_id}.mp4")
    return path if os.path.exists(path) else None


def embed_new_batch(device):
    """Encode up to BATCH_SIZE videos that are on disk but not yet in the library."""
    manifest = load_manifest()
    candidates = [
        row for row in manifest
        if not os.path.exists(library_path(row["video_id"])) and local_video_path(row["video_id"])
    ]
    to_embed = candidates[:BATCH_SIZE]
    print(f"{len(candidates)} videos on disk and not yet embedded. Embedding {len(to_embed)} this run.")
    if not to_embed:
        return

    print(f"\nLoading V-JEPA2: {JEPA_MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID, device_map="auto", attn_implementation="sdpa")
    model.eval()
    num_frames = model.config.frames_per_clip

    for i, row in enumerate(to_embed, 1):
        video_path = local_video_path(row["video_id"])
        print(f"  [{i}/{len(to_embed)}] Encoding {row['video_id']} ...")
        frames = load_video_frames(video_path, num_frames=num_frames)
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            out.transpose(1, 2), NUM_VIDEO_TOKENS
        ).transpose(1, 2).squeeze(0).cpu()

        torch.save(
            {"embedding": pooled, "question": row["question"], "answer": row["answer"]},
            library_path(row["video_id"]),
        )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Embedded {len(to_embed)} new video(s). Library grew accordingly.")


def load_full_library():
    entries = []
    for path in sorted(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))):
        data = torch.load(path)
        data["video_id"] = os.path.splitext(os.path.basename(path))[0]
        entries.append(data)
    return entries


def get_or_create_holdout(library):
    """Fixed held-out split, chosen once and reused across all future runs."""
    if os.path.exists(HOLDOUT_IDS_PATH):
        with open(HOLDOUT_IDS_PATH) as f:
            holdout_ids = set(json.load(f))
    elif len(library) >= MIN_LIBRARY_SIZE_FOR_HOLDOUT:
        shuffled = library[:]
        random.shuffle(shuffled)
        holdout_ids = {e["video_id"] for e in shuffled[:HELD_OUT_COUNT]}
        with open(HOLDOUT_IDS_PATH, "w") as f:
            json.dump(list(holdout_ids), f)
        print(f"Created new fixed held-out set: {len(holdout_ids)} videos, saved to {HOLDOUT_IDS_PATH}")
    else:
        holdout_ids = set()

    held_out = [e for e in library if e["video_id"] in holdout_ids]
    train_set = [e for e in library if e["video_id"] not in holdout_ids]
    return held_out, train_set


def build_bridge(jepa_hidden, qwen_hidden, device, dtype):
    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=dtype)
    if os.path.exists(BRIDGE_CHECKPOINT_PATH):
        print(f"Loading existing bridge checkpoint from {BRIDGE_CHECKPOINT_PATH}")
        bridge.load_state_dict(torch.load(BRIDGE_CHECKPOINT_PATH, map_location=device))
    else:
        print("No existing checkpoint found -- initializing a fresh bridge.")
    return bridge


def build_minibatch_tensors(examples, tokenizer, bridge, qwen, device):
    """Pack several examples into one padded batch. Video-token length is
    always NUM_VIDEO_TOKENS (fixed), so only the text portion needs padding."""
    seq_embeds_list, seq_labels_list, lengths = [], [], []

    for entry in examples:
        jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
        projected = bridge(jepa_tokens).squeeze(0)

        prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
        target = entry["answer"] + tokenizer.eos_token
        q_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
        a_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)

        q_embeds = qwen.get_input_embeddings()(q_ids)
        a_embeds = qwen.get_input_embeddings()(a_ids)

        seq_embeds = torch.cat([projected, q_embeds, a_embeds], dim=0)
        seq_labels = torch.cat([
            torch.full((projected.shape[0] + q_ids.shape[0],), -100, dtype=torch.long, device=device),
            a_ids,
        ])
        seq_embeds_list.append(seq_embeds)
        seq_labels_list.append(seq_labels)
        lengths.append(seq_embeds.shape[0])

    max_len = max(lengths)
    hidden = seq_embeds_list[0].shape[-1]
    batch_size = len(examples)

    padded_embeds = torch.zeros(batch_size, max_len, hidden, device=device, dtype=qwen.dtype)
    padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)

    for i, (emb, lab, length) in enumerate(zip(seq_embeds_list, seq_labels_list, lengths)):
        padded_embeds[i, :length] = emb
        padded_labels[i, :length] = lab
        attention_mask[i, :length] = 1

    return padded_embeds, attention_mask, padded_labels


def evaluate(entry, tokenizer, bridge, qwen, device):
    bridge.eval()
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
        projected = bridge(jepa_tokens)
    prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = qwen.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = qwen.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    bridge.train()
    return output_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embed_new_batch(device)
    library = load_full_library()
    print(f"\nFull library size: {len(library)} video(s)")

    held_out, train_set = get_or_create_holdout(library)
    print(f"Held-out (fixed, reused across runs): {len(held_out)}  |  Training on: {len(train_set)}")

    print(f"\nLoading text backbone: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float32, device_map="auto")
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = build_bridge(jepa_hidden=1024, qwen_hidden=qwen_hidden, device=device, dtype=qwen.dtype)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining with ML_BATCH_SIZE={ML_BATCH_SIZE}, target_loss={TARGET_LOSS}, "
          f"max time {MAX_TIME_BUDGET_SECONDS}s ...")
    start = time.time()
    step, epoch = 0, 0
    last_avg_loss = None

    while time.time() - start < MAX_TIME_BUDGET_SECONDS:
        epoch += 1
        random.shuffle(train_set)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, len(train_set), ML_BATCH_SIZE):
            minibatch = train_set[i:i + ML_BATCH_SIZE]
            inputs_embeds, attention_mask, labels = build_minibatch_tensors(minibatch, tokenizer, bridge, qwen, device)
            outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1
            if time.time() - start >= MAX_TIME_BUDGET_SECONDS:
                break

        last_avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  epoch {epoch}  avg_loss={last_avg_loss:.4f}  elapsed={time.time() - start:.0f}s")

        if last_avg_loss < TARGET_LOSS:
            print(f"  Target loss {TARGET_LOSS} reached -- stopping early.")
            break

    torch.save(bridge.state_dict(), BRIDGE_CHECKPOINT_PATH)
    print(f"\nTraining stopped after {step} steps, {epoch} epoch(s), final avg_loss={last_avg_loss:.4f}. "
          f"Checkpoint saved to {BRIDGE_CHECKPOINT_PATH}")

    if held_out:
        print("\n--- FIXED HELD-OUT CHECK ---")
        for entry in held_out:
            output_text = evaluate(entry, tokenizer, bridge, qwen, device)
            print(f"\nVideo: {entry['video_id']}")
            print(f"Question: {entry['question']}")
            print(f"Output: {output_text}")
            print(f"Ground truth: {entry['answer']}")


if __name__ == "__main__":
    main()