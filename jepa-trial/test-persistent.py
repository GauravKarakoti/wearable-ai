"""
test_persistent.py -- check whether the bridge still "remembers" earlier
batches after subsequent training runs on newer batches.

WORKFLOW THIS SUPPORTS:
  1. train_bridge.py on batch 1 (10 videos) -> delete those .mp4s
  2. train_bridge.py on batch 2 (10 new videos) -> delete those .mp4s
  3. test_persistent.py with batch 1's video_ids -> did batch 2's training
     overwrite what was learned from batch 1, or did it retain it?
"""

import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
RESULTS_LOG_PATH = os.path.join(OUTPUT_DIR, "persistence_test_log.jsonl")

TEST_VIDEO_IDS = [
    "204df747cf88e632",
    "205a1ef3950ef283",
    "20665aadf887aa08",
    "2077ea2f35583ff6",
    "20b7e0d8cf3e158a",
    "21223c1e14298faf",
    "2192e93a36108081",
    "21a625ead630c23e",
    "22d2c11a300d339d",
    "231fceaac1eb5cac",
]


def load_bridge_architecture(jepa_hidden, qwen_hidden, device, dtype):
    import torch.nn as nn
    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=dtype)
    return bridge


def load_entries_to_test():
    if TEST_VIDEO_IDS:
        entries = []
        for vid in TEST_VIDEO_IDS:
            path = os.path.join(EMBEDDING_LIBRARY_DIR, f"{vid}.pt")
            if not os.path.exists(path):
                print(f"  WARNING: no cached embedding found for video_id '{vid}' -- skipping. "
                      f"(Was it actually embedded in a previous train_bridge.py run?)")
                continue
            data = torch.load(path)
            data["video_id"] = vid
            entries.append(data)
        return entries
    else:
        import glob
        entries = []
        for path in sorted(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))):
            data = torch.load(path)
            data["video_id"] = os.path.splitext(os.path.basename(path))[0]
            entries.append(data)
        return entries


def rough_word_overlap(pred, truth):
    """Crude, quick similarity signal -- NOT a real metric. Just for fast eyeballing
    across many runs. For anything you'd put in the paper, use the LLM-judge
    rubric approach discussed earlier instead."""
    pred_words = set(pred.lower().split())
    truth_words = set(truth.lower().split())
    if not truth_words:
        return 0.0
    return len(pred_words & truth_words) / len(truth_words)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(BRIDGE_CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"No checkpoint found at {BRIDGE_CHECKPOINT_PATH}. "
            f"Run train_bridge.py at least once before testing persistence."
        )

    entries = load_entries_to_test()
    if not entries:
        print("No entries to test (check TEST_VIDEO_IDS or the embedding library). Exiting.")
        return
    print(f"Testing {len(entries)} cached video(s): {[e['video_id'] for e in entries]}")

    print(f"\nLoading Qwen2-VL: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    qwen.eval()

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = load_bridge_architecture(jepa_hidden=1024, qwen_hidden=qwen_hidden, device=device, dtype=qwen.dtype)
    bridge.load_state_dict(torch.load(BRIDGE_CHECKPOINT_PATH, map_location=device))
    bridge.eval()
    print(f"Loaded bridge checkpoint from {BRIDGE_CHECKPOINT_PATH}")

    print("\n--- PERSISTENCE TEST RESULTS ---")
    results = []
    for entry in entries:
        jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
        with torch.no_grad():
            projected = bridge(jepa_tokens)

        prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        text_embeds = qwen.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([projected, text_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        with torch.no_grad():
            generated_ids = qwen.generate(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS
            )
        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        overlap = rough_word_overlap(output_text, entry["answer"])

        print(f"\nVideo: {entry['video_id']}")
        print(f"Question: {entry['question']}")
        print(f"Output:       {output_text if output_text.strip() else '(EMPTY -- likely forgotten/collapsed)'}")
        print(f"Ground truth: {entry['answer']}")
        print(f"Rough word overlap: {overlap:.2f}  (crude signal only, not a real metric)")

        results.append({
            "video_id": entry["video_id"],
            "question": entry["question"],
            "output": output_text,
            "ground_truth": entry["answer"],
            "rough_word_overlap": overlap,
        })

    avg_overlap = sum(r["rough_word_overlap"] for r in results) / len(results)
    empty_count = sum(1 for r in results if not r["output"].strip())
    print(f"\n--- SUMMARY ---")
    print(f"Tested: {len(results)}  |  Empty outputs: {empty_count}  |  Avg rough word overlap: {avg_overlap:.2f}")

    import json
    with open(RESULTS_LOG_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nAppended results to {RESULTS_LOG_PATH} -- run this after each new training batch")
    print("to build up a persistence/forgetting curve over time.")


if __name__ == "__main__":
    main()