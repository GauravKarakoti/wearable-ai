"""
test-predictions.py -- EgoLongQA Grand Challenge submission inference.

APPROACH: per-option teacher-forced loss scoring, NOT autoregressive
generation. For each question, we compute the cross-entropy of each of the
four candidate answer texts under the frozen bridge+Qwen system (one forward
pass per option, no decoding loop), and predict whichever option has the
lowest loss (highest likelihood). This:
  - Matches what the bridge was actually trained to do (predict correct
    free-text answers), rather than asking it to emit a bare letter it has
    never been trained to produce.
  - Is far faster than generation -- no risk of approaching the 300s/query
    limit. Your own earlier logs recorded generation times up to 458.8s for
    a single query using standard max_new_tokens generation; this approach
    avoids that failure mode entirely by never entering a decoding loop.
"""

import os
import re
import json
import time
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

MANIFEST_PATH = "../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"
OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")

PREDICTIONS_PATH = "./trial_output/predictions.jsonl"

PER_QUERY_WARN_THRESHOLD_SEC = 30


def load_manifest_in_order():
    rows = []
    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["video_id"] = os.path.splitext(row["video_path"])[0]
            rows.append(row)
    return rows


def parse_mcq_options(mcq_options_str):
    """'A. text one. B. text two. C. text three. D. text four.' -> {'A': 'text one.', ...}"""
    matches = re.findall(r'([A-D])\.\s*(.*?)(?=\s[A-D]\.\s|$)', mcq_options_str.strip())
    return {letter: text.strip() for letter, text in matches}


def load_bridge_architecture(jepa_hidden, qwen_hidden, device, dtype):
    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=dtype)
    return bridge


def option_teacher_forced_loss(jepa_tokens_projected, option_text, question, tokenizer, qwen, device):
    """Cross-entropy of ONE candidate option's text as the answer, no generation."""
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    target = option_text + tokenizer.eos_token

    q_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    q_embeds = qwen.get_input_embeddings()(q_ids)
    a_embeds = qwen.get_input_embeddings()(a_ids)
    inputs_embeds = torch.cat([jepa_tokens_projected, q_embeds, a_embeds], dim=1)

    video_len, q_len = jepa_tokens_projected.shape[1], q_ids.shape[1]
    labels = torch.cat([
        torch.full((1, video_len + q_len), -100, dtype=torch.long, device=device),
        a_ids,
    ], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        loss = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
    return loss.item()


def predict_mcq_answer(entry, question, options, tokenizer, bridge, qwen, device):
    """Score all four options, return the letter with lowest loss."""
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
        projected = bridge(jepa_tokens)

    losses = {}
    for letter in ["A", "B", "C", "D"]:
        if letter not in options:
            continue
        losses[letter] = option_teacher_forced_loss(projected, options[letter], question, tokenizer, qwen, device)

    if not losses:
        raise ValueError("No valid options parsed for this row -- check mcq_options formatting.")

    best_letter = min(losses, key=losses.get)
    return best_letter, losses


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    manifest = load_manifest_in_order()
    print(f"Manifest loaded: {len(manifest)} rows, in original order.")

    existing_predictions = set()
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        existing_predictions.add(data["video_path"])
                    except json.JSONDecodeError:
                        continue
    
    if existing_predictions:
        print(f"Found {len(existing_predictions)} existing predictions in {PREDICTIONS_PATH}. Resuming...")

    print(f"\nLoading text backbone: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float32, device_map="auto")
    qwen.eval()

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = load_bridge_architecture(jepa_hidden=1024, qwen_hidden=qwen_hidden, device=device, dtype=qwen.dtype)
    if not os.path.exists(BRIDGE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {BRIDGE_CHECKPOINT_PATH}.")
    bridge.load_state_dict(torch.load(BRIDGE_CHECKPOINT_PATH, map_location=device))
    bridge.eval()
    print(f"Loaded bridge checkpoint from {BRIDGE_CHECKPOINT_PATH}\n")

    total_start = time.time()
    processed_count = 0

    with open(PREDICTIONS_PATH, "a") as f_out:
        for i, row in enumerate(manifest, 1):
            video_path = row["video_path"]
            
            if video_path in existing_predictions:
                continue

            video_id = row["video_id"]
            embedding_path = os.path.join(EMBEDDING_LIBRARY_DIR, f"{video_id}.pt")
            if not os.path.exists(embedding_path):
                raise FileNotFoundError(
                    f"Row {i}/{len(manifest)}: no cached embedding for video_id '{video_id}'. "
                    f"You said all 700 are processed -- check EMBEDDING_LIBRARY_DIR / MANIFEST_PATH match."
                )
            
            entry = torch.load(embedding_path)
            options = parse_mcq_options(row["mcq_options"])

            t0 = time.time()
            predicted_letter, losses = predict_mcq_answer(
                entry, row["question"], options, tokenizer, bridge, qwen, device
            )
            elapsed = time.time() - t0

            if elapsed > PER_QUERY_WARN_THRESHOLD_SEC:
                print(f"  WARNING: row {i} ({video_id}) took {elapsed:.1f}s -- investigate before submitting "
                      f"if this happens repeatedly (limit is 300s/query).")

            pred_dict = {"video_path": video_path, "mcq_answer": predicted_letter}
            f_out.write(json.dumps(pred_dict) + "\n")
            f_out.flush()
            
            existing_predictions.add(video_path)
            processed_count += 1

            if processed_count % 50 == 0:
                avg_time = (time.time() - total_start) / processed_count
                print(f"[Processed {processed_count} new items (Total {len(existing_predictions)}/{len(manifest)})] avg {avg_time:.3f}s/query so far")

    total_elapsed = time.time() - total_start
    if processed_count > 0:
        print(f"\nDone. {processed_count} NEW predictions generated in {total_elapsed:.1f}s "
              f"({total_elapsed / processed_count:.3f}s/query average).")
    else:
        print("\nAll predictions were already present. Nothing new generated.")

    total_completed = len(existing_predictions)
    assert total_completed == len(manifest), (
        f"Prediction count mismatch: {total_completed} predictions vs {len(manifest)} manifest rows. "
        f"Do not submit until this matches exactly."
    )
    assert total_completed == 700, (
        f"Expected exactly 700 predictions, got {total_completed}. Check your manifest."
    )

    print(f"\nVerified {total_completed} completed rows in {PREDICTIONS_PATH}")
    print("File contains ONLY {\"video_path\": ..., \"mcq_answer\": ...} lines, ready for submission.")


if __name__ == "__main__":
    main()