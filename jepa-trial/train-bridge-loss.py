"""
train-bridge-loss.py -- keep training the bridge on the full (already fully
embedded) library, epoch after epoch, with NO time cap, until avg_loss
drops below TARGET_LOSS.

CHECKPOINT SEMANTICS:
  - Saves after every COMPLETED epoch, but ONLY if that epoch's avg_loss is
    lower than the best loss ever seen (persisted in best_loss.json, not
    just in-memory -- so this survives Ctrl+C and resuming later).
  - If you kill it mid-epoch, that epoch's progress is simply lost (it
    never completed), but every prior completed-and-improved epoch is
    already safely on disk.
  - Re-running this script picks up exactly where it left off: loads the
    existing checkpoint, loads the existing best_loss, and keeps going.
"""

import os
import json
import glob
import random
import time
import datetime
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ML_BATCH_SIZE = 8
LEARNING_RATE = 1e-4
TARGET_LOSS = 0.6
MAX_NEW_TOKENS = 256

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
HOLDOUT_IDS_PATH = os.path.join(OUTPUT_DIR, "holdout_ids.json")
BEST_LOSS_PATH = os.path.join(OUTPUT_DIR, "best_loss.json")
EPOCH_LOG_PATH = os.path.join(OUTPUT_DIR, "target_training_log.jsonl")


def load_full_library():
    entries = []
    for path in sorted(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))):
        data = torch.load(path)
        data["video_id"] = os.path.splitext(os.path.basename(path))[0]
        entries.append(data)
    return entries


def load_holdout_split(library):
    if not os.path.exists(HOLDOUT_IDS_PATH):
        print(f"WARNING: no {HOLDOUT_IDS_PATH} found -- training on the entire library with no held-out check.")
        return [], library
    with open(HOLDOUT_IDS_PATH) as f:
        holdout_ids = set(json.load(f))
    held_out = [e for e in library if e["video_id"] in holdout_ids]
    train_set = [e for e in library if e["video_id"] not in holdout_ids]
    return held_out, train_set


def load_best_loss():
    if os.path.exists(BEST_LOSS_PATH):
        with open(BEST_LOSS_PATH) as f:
            data = json.load(f)
        print(f"Resuming: best loss so far is {data['best_loss']:.4f} (from a previous session, epoch {data.get('epoch', '?')}).")
        return data["best_loss"]
    print("No prior best_loss.json found -- starting fresh (best loss = infinity).")
    return float("inf")


def save_best_loss(best_loss, epoch):
    with open(BEST_LOSS_PATH, "w") as f:
        json.dump({"best_loss": best_loss, "epoch": epoch, "saved_at": datetime.datetime.now().isoformat()}, f)


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


def held_out_teacher_forced_loss(held_out, tokenizer, bridge, qwen, device):
    """Cheap held-out check -- teacher forcing, no generation. Averaged over held_out."""
    if not held_out:
        return None
    bridge.eval()
    losses = []
    with torch.no_grad():
        for entry in held_out:
            jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
            projected = bridge(jepa_tokens)
            prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
            target = entry["answer"] + tokenizer.eos_token
            q_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            a_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            q_embeds = qwen.get_input_embeddings()(q_ids)
            a_embeds = qwen.get_input_embeddings()(a_ids)
            inputs_embeds = torch.cat([projected, q_embeds, a_embeds], dim=1)
            video_len, q_len = projected.shape[1], q_ids.shape[1]
            labels = torch.cat([
                torch.full((1, video_len + q_len), -100, dtype=torch.long, device=device),
                a_ids,
            ], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
            loss = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
            losses.append(loss.item())
    bridge.train()
    return sum(losses) / len(losses)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    library = load_full_library()
    print(f"Full library size: {len(library)} video(s)")
    held_out, train_set = load_holdout_split(library)
    print(f"Held-out: {len(held_out)}  |  Training on: {len(train_set)}")

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

    best_loss = load_best_loss()
    epoch = 0
    print(f"\nTraining with no time cap. Target avg_loss < {TARGET_LOSS}. "
          f"Ctrl+C any time -- last completed+improved epoch is already saved.\n")

    try:
        while True:
            epoch += 1
            epoch_start = time.time()
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

            avg_train_loss = epoch_loss / max(n_batches, 1)
            epoch_time = time.time() - epoch_start

            ho_loss = held_out_teacher_forced_loss(held_out, tokenizer, bridge, qwen, device)
            ho_str = f"  held_out_loss={ho_loss:.4f}" if ho_loss is not None else ""

            improved = avg_train_loss < best_loss
            if improved:
                torch.save(bridge.state_dict(), BRIDGE_CHECKPOINT_PATH)
                best_loss = avg_train_loss
                save_best_loss(best_loss, epoch)

            print(f"epoch {epoch}  avg_train_loss={avg_train_loss:.4f}{ho_str}  "
                  f"time={epoch_time:.0f}s  {'SAVED (new best)' if improved else 'not saved (no improvement)'}")

            with open(EPOCH_LOG_PATH, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch,
                    "avg_train_loss": avg_train_loss,
                    "held_out_loss": ho_loss,
                    "epoch_time_sec": epoch_time,
                    "saved": improved,
                    "best_loss_so_far": best_loss,
                    "timestamp": datetime.datetime.now().isoformat(),
                }) + "\n")

            if avg_train_loss < TARGET_LOSS:
                print(f"\nTarget loss {TARGET_LOSS} reached (avg_train_loss={avg_train_loss:.4f}). Stopping.")
                break

    except KeyboardInterrupt:
        print(f"\n\nInterrupted at epoch {epoch}. Best checkpoint on disk is from epoch "
              f"{json.load(open(BEST_LOSS_PATH))['epoch'] if os.path.exists(BEST_LOSS_PATH) else 'N/A'} "
              f"with avg_loss={best_loss:.4f}. Re-run this script to resume.")


if __name__ == "__main__":
    main()