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
    "286c91162b2b9918",
    "286ebb5140e19aa3",
    "2886a72c2428e93a",
    "28aa19bcdf5e59da",
    "295e35c4ad8632a4",
    "2a35434e1c39f80a",
    "2a5113d47f90e558",
    "2a70e7ae5f20bdbb",
    "2ae6c4477585a8b3",
    "2ba0accbfc13712f",
    "2bc7ad352bd4b905",
    "2bfffb7b911d8655",
    "2c4e9ea1c554519b",
    "2c50721b06e0bdb2",
    "2c582f9beb138eab",
    "2d40456bebd844e6",
    "2d86878f4797fb11",
    "2e3bc21cf6685807",
    "2e5da7a66af53497",
    "2ec4cbd4fdf4a1b3",
    "2efdb9f426131303",
    "2f6df18b351825a7",
    "2fc07122287502b0",
    "304e82b751d3e20a",
    "307004bf807802e2",
    "3091b788dc993bfe",
    "31307eb6d9e335b3",
    "314dabb96387cef9",
    "316ebd12dea8cd7c",
    "31812970338785f4",
    "31a5289ea39eca44",
    "31cdcd6a7135a92b",
    "328f0088f8088305",
    "32d790a85a4b9997",
    "3305f03f9d1f59cb",
    "33635869f4013ce3",
    "337838c05c845e1c",
    "33a1126a1192ca2e",
    "33ebb9c842c4e4d9",
    "3409f3004e2a8078",
    "343e76a2bc8ee8c3",
    "3485bec65fb9c522",
    "3491310010b29800",
    "34b6409da071d5a5",
    "34e6a4629781783d",
    "354b9f490f6c9bda",
    "35c648e328355f23",
    "35dd376cb47e6ad7",
    "365bca27d34574e1",
    "36b75f41d05284cc",
    "36ba904b984d2728",
    "36f814e6c5afda98",
    "37017da9a7e18731",
    "37783571b945d66a",
    "377cee678a5b31ac",
    "37c66bcce1318f83",
    "380e2adf98758c1e",
    "38a6a97bf17d7eaa",
    "38ad6b6bf629e98d",
    "3947a6e7af8e9daf",
    "39ce44d892305734",
    "3a28a265e1f29959",
    "3a8a1a5119cec3a1",
    "3aa7a20249708db0",
    "3ad906ef59fd7a7e",
    "3b07bfcac97cf475",
    "3b5ca88d08858cc2",
    "3b98ab8aa7763741",
    "3be840b7358c22ee",
    "3bee9eedd712ce8a",
    "3c908cbbbd90b987",
    "3cfc9122829c55aa",
    "3d23a421b35367bf",
    "3d90471ed252f1ca",
    "3da36f3a8b3881c3",
    "3ec8a158983b80dd",
    "3fe39996fa55219f",
    "3fee5bc2891e37d2",
    "40031fc7023a8bf0",
    "40344e725fcd2287",
    "408e62080f5029bf",
    "40aa2da9113e3bcc",
    "40f996a931c65581",
    "416de6aab55d24fd",
    "419eed589c74bb83",
    "41d0934266db9f63",
    "41dd911b766cf702",
    "42342fe7ec0f0ecd",
    "4263039b604b9793",
    "428ce111514ecbf5",
    "438a7c0e65e1ecfd",
    "439f4429c121b912",
    "43e9700097961f34",
    "43f136ac06780df7",
    "44107e68e219c362",
    "451fb61d25ac9294",
    "461825ec4f8e89d4",
    "4632f6fc4a66f6be",
    "466a43f91bd6c63b",
    "4675c118ef30f8a6"
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