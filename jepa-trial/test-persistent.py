"""
test_persistent.py -- check whether the bridge still "remembers" earlier
batches after subsequent training runs on newer/larger batches.

METRICS, per example:
  - teacher_forced_loss : cross-entropy of the GROUND TRUTH answer under the
    current bridge+model (teacher forcing, no generation). This is the most
    principled signal here -- it's directly comparable to your training
    loss numbers, and it still gives you a real number even when generation
    collapses to empty output (word overlap can't).
  - exact_match          : case-insensitive exact string match (strict)
  - rouge_l_f1            : order-aware overlap via longest common
    subsequence -- catches "right words, right order" better than the
    word-set overlap alone
  - word_overlap          : the original crude set-overlap signal, kept for
    continuity with earlier logs
  - repetition_ratio       : 1 - (unique words / total words) in the
    generated output -- flags the "bottle of water... bottle of water"
    degenerate-repeat failure mode directly, rather than you eyeballing it
  - output_word_count / truth_word_count / length_ratio
  - generation_time_sec
  - is_empty
"""

import os
import json
import glob
import time
import datetime
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 256

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
HOLDOUT_IDS_PATH = os.path.join(OUTPUT_DIR, "holdout_ids.json")
RESULTS_LOG_PATH = os.path.join(OUTPUT_DIR, "persistence_test_log.jsonl")

# Leave empty to default to the fixed held-out set from holdout_ids.json.
TEST_VIDEO_IDS = [
    "e174f252b4965685",
    "e1a178b47d998874",
    "e1cf4266007a09fc",
    "e1df5f7739249f5e",
    "e1e7ce832da64fd4",
    "e1fe951227c018f2",
    "e21cbde6a9acfc39",
    "e2c8555da7bb650a",
    "e2e76d2c5624e3de",
    "e31690b83652844a",
    "e31fff469914d9c8",
    "e35940287634d819",
    "e373a6d763abf3e5",
    "e3be3647b8bce029",
    "e3f2d04be59af3eb",
    "e40865d30d1c1594",
    "e4947cad50181471",
    "e531f121a4a804ad",
    "e58aa85ab695d9cf",
    "e5c41c61342fb8bb",
    "e5f0a71c73ee4f93",
    "e61a0c50e14505ad",
    "e6bd011662276977",
    "e6c6bf80b923fecb",
    "e76f7860b790d996",
    "e78f2a7a7ab24341",
    "e79b9a6810c607c8",
    "e7fcdfd76103e193",
    "e80db9b80583d44e",
    "e8797da62ff09125",
    "e9054b8f2eba960d",
    "e93c3bfaeed14fe6",
    "e9d15687eea0b786",
    "e9f21972ff8c3870",
    "ea130c0e5f45f32f",
    "ea1ebfee98ea5267",
    "ea24fc0365d5498a",
    "ea96f6fffa0cea82",
    "eab60fc3ebd45353",
    "eb2ca46b8e506463",
    "ebd8e191c386b280",
    "ecf8e9e9c8fcbd8a",
    "ed9b6346bee64283",
    "edaffe97b6620535",
    "ef25e0913ba7dcc9",
    "ef2ef1313755ec07",
    "ef51015464812b8c",
    "ef75ac3d92198596",
    "efe05555186eeea8",
    "f007e9c9cb65c590",
    "f028f4b2f34cc85d",
    "f03441accf18f5ca",
    "f06b20e6dc00da11",
    "f06bcd6accde47d9",
    "f11fb6958943a777",
    "f157244a935aeb48",
    "f21c3b8ce64dc2fb",
    "f267337fae95a881",
    "f27d47cd5dbd6586",
    "f2ec78a1ecc0e859",
    "f333d6e395e6afc5",
    "f3535a6623c8d73c",
    "f35cc4940be701c8",
    "f377d9635cb0bfb0",
    "f4641dd942f4f1df",
    "f4d2a0877e75856b",
    "f5780d9b0cfed901",
    "f5af4eee813b9c6c",
    "f5cd22aab57406dc",
    "f65855c5790ba916",
    "f6f3614a2d9f01b4",
    "f70eabbe9ea52f23",
    "f740c282872c240d",
    "f7695db49909a96f",
    "f86b147546baa484",
    "f86d9f011c206fcd",
    "f91c5e9d66190337",
    "f93a9b2e77ccdf89",
    "f9d729b8085cef5e",
    "fa31ca8100085de1",
    "fc12d4e78cb8b907",
    "fd2b5b76d5b9b925",
    "fd922a7f58506440",
    "fdb3a314f0780b11",
    "fe19b9dc6e384e4a",
    "fe34b3af082ac499",
    "febc10367e2090b2",
    "ff1b1c73b6ea250f",
    "ff20262920073129",
    "ff5854db35d7a9bc",
]


def load_bridge_architecture(jepa_hidden, qwen_hidden, device, dtype):
    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=dtype)
    return bridge


def resolve_test_ids():
    if TEST_VIDEO_IDS:
        print(f"Using explicit TEST_VIDEO_IDS override ({len(TEST_VIDEO_IDS)} videos).")
        return TEST_VIDEO_IDS
    if os.path.exists(HOLDOUT_IDS_PATH):
        with open(HOLDOUT_IDS_PATH) as f:
            ids = json.load(f)
        print(f"No override set -- using the fixed held-out set from {HOLDOUT_IDS_PATH} ({len(ids)} videos).")
        return ids
    print(f"No override and no {HOLDOUT_IDS_PATH} found yet -- testing the entire embedding library.")
    return None


def load_entries_to_test(ids_to_test):
    entries = []
    if ids_to_test is None:
        paths = sorted(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt")))
        for path in paths:
            data = torch.load(path)
            data["video_id"] = os.path.splitext(os.path.basename(path))[0]
            entries.append(data)
        return entries
    for vid in ids_to_test:
        path = os.path.join(EMBEDDING_LIBRARY_DIR, f"{vid}.pt")
        if not os.path.exists(path):
            print(f"  WARNING: no cached embedding found for video_id '{vid}' -- skipping.")
            continue
        data = torch.load(path)
        data["video_id"] = vid
        entries.append(data)
    return entries


# ---- Metrics ----

def word_overlap(pred, truth):
    pred_words, truth_words = set(pred.lower().split()), set(truth.lower().split())
    if not truth_words:
        return 0.0
    return len(pred_words & truth_words) / len(truth_words)


def exact_match(pred, truth):
    return pred.strip().lower() == truth.strip().lower()


def rouge_l_f1(pred, truth):
    """Longest-common-subsequence based F1 -- order-aware, unlike set overlap."""
    p_tokens, t_tokens = pred.lower().split(), truth.lower().split()
    if not p_tokens or not t_tokens:
        return 0.0
    m, n = len(p_tokens), len(t_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == t_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / m, lcs / n
    return 2 * precision * recall / (precision + recall)


def repetition_ratio(text):
    """1 - unique/total word ratio. 0 = no repeats, closer to 1 = heavily degenerate."""
    words = text.lower().split()
    if not words:
        return 0.0
    return 1 - (len(set(words)) / len(words))


def teacher_forced_loss(entry, tokenizer, bridge, qwen, device):
    """Cross-entropy of the GROUND TRUTH answer given the video + question,
    under the current model -- no generation/sampling involved. Directly
    comparable to training loss, and still meaningful even when greedy
    generation collapses to empty output."""
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


def generate(entry, tokenizer, bridge, qwen, device):
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
        projected = bridge(jepa_tokens)
    prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = qwen.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        generated_ids = qwen.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS)
    gen_time = time.time() - t0
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text, gen_time


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if not os.path.exists(BRIDGE_CHECKPOINT_PATH):
        raise FileNotFoundError(f"No checkpoint found at {BRIDGE_CHECKPOINT_PATH}. Run train_bridge.py first.")

    checkpoint_mtime = datetime.datetime.fromtimestamp(os.path.getmtime(BRIDGE_CHECKPOINT_PATH)).isoformat()
    library_size = len(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt")))
    run_id = datetime.datetime.now().isoformat()

    ids_to_test = resolve_test_ids()
    entries = load_entries_to_test(ids_to_test)
    if not entries:
        print("No entries to test. Exiting.")
        return
    print(f"Testing {len(entries)} cached video(s). Library size at test time: {library_size}. "
          f"Checkpoint last modified: {checkpoint_mtime}")

    print(f"\nLoading text backbone: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float32, device_map="auto")
    qwen.eval()

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = load_bridge_architecture(jepa_hidden=1024, qwen_hidden=qwen_hidden, device=device, dtype=qwen.dtype)
    bridge.load_state_dict(torch.load(BRIDGE_CHECKPOINT_PATH, map_location=device))
    bridge.eval()
    print(f"Loaded bridge checkpoint from {BRIDGE_CHECKPOINT_PATH}")

    print("\n--- PERSISTENCE TEST RESULTS ---")
    results = []
    for entry in entries:
        output_text, gen_time = generate(entry, tokenizer, bridge, qwen, device)
        tf_loss = teacher_forced_loss(entry, tokenizer, bridge, qwen, device)

        metrics = {
            "video_id": entry["video_id"],
            "question": entry["question"],
            "output": output_text,
            "ground_truth": entry["answer"],
            "teacher_forced_loss": tf_loss,
            "exact_match": exact_match(output_text, entry["answer"]),
            "rouge_l_f1": rouge_l_f1(output_text, entry["answer"]),
            "word_overlap": word_overlap(output_text, entry["answer"]),
            "repetition_ratio": repetition_ratio(output_text),
            "output_word_count": len(output_text.split()),
            "truth_word_count": len(entry["answer"].split()),
            "generation_time_sec": round(gen_time, 3),
            "is_empty": not output_text.strip(),
            "run_id": run_id,
            "library_size_at_test": library_size,
            "checkpoint_mtime": checkpoint_mtime,
        }
        metrics["length_ratio"] = (
            metrics["output_word_count"] / metrics["truth_word_count"]
            if metrics["truth_word_count"] else 0.0
        )

        print(f"\nVideo: {entry['video_id']}")
        print(f"Question: {entry['question']}")
        print(f"Output:       {output_text if output_text.strip() else '(EMPTY -- likely forgotten/collapsed)'}")
        print(f"Ground truth: {entry['answer']}")
        print(f"teacher_forced_loss={tf_loss:.4f}  exact_match={metrics['exact_match']}  "
              f"rouge_l_f1={metrics['rouge_l_f1']:.2f}  word_overlap={metrics['word_overlap']:.2f}  "
              f"repetition={metrics['repetition_ratio']:.2f}  length_ratio={metrics['length_ratio']:.2f}  "
              f"gen_time={gen_time:.2f}s")

        results.append(metrics)

    def avg(key):
        return sum(r[key] for r in results) / len(results)

    summary = {
        "run_id": run_id,
        "type": "summary",
        "tested": len(results),
        "library_size_at_test": library_size,
        "checkpoint_mtime": checkpoint_mtime,
        "avg_teacher_forced_loss": avg("teacher_forced_loss"),
        "exact_match_rate": sum(r["exact_match"] for r in results) / len(results),
        "avg_rouge_l_f1": avg("rouge_l_f1"),
        "avg_word_overlap": avg("word_overlap"),
        "avg_repetition_ratio": avg("repetition_ratio"),
        "avg_length_ratio": avg("length_ratio"),
        "empty_rate": sum(r["is_empty"] for r in results) / len(results),
        "avg_generation_time_sec": avg("generation_time_sec"),
    }

    print(f"\n--- SUMMARY (run {run_id}) ---")
    for k, v in summary.items():
        if k not in ("run_id", "type"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    with open(RESULTS_LOG_PATH, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(results)} per-example records + 1 summary record to {RESULTS_LOG_PATH}")
    print("Filter on \"type\": \"summary\" lines to build a trend curve across sessions without re-parsing everything.")


if __name__ == "__main__":
    main()