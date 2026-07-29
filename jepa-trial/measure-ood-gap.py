"""
measure-ood-gap.py -- ID vs OOD performance gap, split by video category.

WHAT THIS ACTUALLY MEASURES: the difference in held-out quality between
categories the bridge has seen during training (in-distribution, ID) and
categories it has never seen at all (out-of-distribution, OOD). This is
the "ID-OOD Performance Gap" concept from the OOD-robustness literature --
NOT a canonical single-formula metric, just this well-established comparison,
applied here.
"""

import os
import json
import glob
import statistics
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MANIFEST_PATH = "../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")

# Set this explicitly if you already know which categories were deliberately
# held out of training (from a dedicated OOD retrain). Leave empty to have
# the script infer it automatically from what's actually in the library.
DELIBERATE_OOD_CATEGORIES = []


def load_video_id_to_category():
    mapping = {}
    with open(MANIFEST_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            vid = os.path.splitext(row["video_path"])[0]
            mapping[vid] = row.get("category", "UNKNOWN")
    return mapping


def load_library_video_ids():
    return {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))
    }


def check_ood_validity(vid_to_cat, trained_ids):
    """Figure out which categories are actually absent from what's been trained."""
    all_categories = set(vid_to_cat.values())
    trained_categories = {vid_to_cat.get(vid, "UNKNOWN") for vid in trained_ids}
    fully_absent = all_categories - trained_categories

    print(f"Total categories in manifest: {len(all_categories)} -> {sorted(all_categories)}")
    print(f"Categories present in the current library: {len(trained_categories)}")
    print(f"Categories NEVER seen by the current checkpoint: {len(fully_absent)} -> {sorted(fully_absent)}")

    if not fully_absent:
        print("\nWARNING: every category already appears in your training library.")
        print("There is no free/valid OOD split here -- any 'OOD' result computed against")
        print("the current checkpoint would actually just be measuring in-distribution")
        print("performance under a different label. You'd need a dedicated retrain that")
        print("deliberately excludes 1-2 categories to get a real answer.")
        return None
    return fully_absent


def load_bridge_architecture(jepa_hidden, qwen_hidden, device, dtype):
    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=dtype)
    return bridge


def teacher_forced_loss(entry, tokenizer, bridge, qwen, device):
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
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
    with torch.no_grad():
        loss = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
    return loss.item()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    vid_to_cat = load_video_id_to_category()
    trained_ids = load_library_video_ids()
    print(f"Library contains {len(trained_ids)} embedded videos.\n")

    if DELIBERATE_OOD_CATEGORIES:
        ood_categories = set(DELIBERATE_OOD_CATEGORIES)
        print(f"Using manually specified OOD categories: {sorted(ood_categories)}")
    else:
        ood_categories = check_ood_validity(vid_to_cat, trained_ids)
        if ood_categories is None:
            print("\nStopping here -- no valid OOD split to measure. See warning above.")
            return

    id_ids = [vid for vid in trained_ids if vid_to_cat.get(vid) not in ood_categories]
    ood_ids = [vid for vid, cat in vid_to_cat.items() if cat in ood_categories and vid not in trained_ids]

    if not ood_ids:
        print("No embedded examples found for the OOD categories -- embed a few videos from "
              "those categories (without adding them to training) before running this again.")
        return

    print(f"\nID examples (trained categories): {len(id_ids)}")
    print(f"OOD examples (never-trained categories): {len(ood_ids)}")

    print(f"\nLoading text backbone: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float32, device_map="auto")
    qwen.eval()
    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = load_bridge_architecture(1024, qwen_hidden, device, qwen.dtype)
    bridge.load_state_dict(torch.load(BRIDGE_CHECKPOINT_PATH, map_location=device))
    bridge.eval()

    def eval_group(ids, label):
        losses = []
        for vid in ids:
            path = os.path.join(EMBEDDING_LIBRARY_DIR, f"{vid}.pt")
            if not os.path.exists(path):
                continue
            entry = torch.load(path)
            losses.append(teacher_forced_loss(entry, tokenizer, bridge, qwen, device))
        if not losses:
            return None
        mean_loss = statistics.mean(losses)
        print(f"{label}: N={len(losses)}  mean_loss={mean_loss:.4f}")
        return mean_loss

    print("\n--- ID-OOD PERFORMANCE GAP ---")
    id_loss = eval_group(id_ids[:200], "In-distribution (trained categories)")   # cap for speed; raise if you want the full set
    ood_loss = eval_group(ood_ids, "Out-of-distribution (unseen categories)")

    if id_loss is not None and ood_loss is not None:
        gap = ood_loss - id_loss
        print(f"\nID-OOD Performance Gap (higher = less resilient): {gap:.4f}")
        print("(OOD loss minus ID loss, in teacher-forced cross-entropy units)")


if __name__ == "__main__":
    main()