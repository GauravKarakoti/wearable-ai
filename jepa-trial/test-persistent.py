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
    "469f55fef5c7c288",
    "46e9d1736a3b9ef2",
    "4711132501b52300",
    "47a559ec148d775c",
    "47e8fc6d0e6fe605",
    "47f45a2c0350ed90",
    "481e71a49d271724",
    "483a6f8c69de7e88",
    "4866d01dd54c7311",
    "4895df868cca4688",
    "49488192c066cb19",
    "4959be3d4b58cf56",
    "495fe780651b0246",
    "4962704615e4fa8a",
    "4a0a16cdba79df5d",
    "4a5c273810448ab6",
    "4a96bbba4873d9a8",
    "4aa4c810fc8c8a91",
    "4ad1f0bdbc6f74dc",
    "4b665c6c345137ea",
    "4ceceb8b2234316b",
    "4d5cedb4f06e60ad",
    "4d98837f12753a68",
    "4e0a7d22563cf57f",
    "4e1bbce234587113",
    "4e4cdc3c3bd525b2",
    "4edf1dd9738170df",
    "4ee6962b7a404ef9",
    "4f895e1bff9ed1a5",
    "4fa2d52355bf6f66",
    "4fb34c7377dead94",
    "5007b14318474903",
    "51b376fd7498147e",
    "520c4e7fab67bc19",
    "520e4e5f969eb453",
    "5294641f57768c0e",
    "52a951fc79d7bdd0",
    "5388720913f4570e",
    "5390f31116aaeebd",
    "53a3fa0a7c4eba36",
    "5405e015928901bc",
    "5422255837b1ad5f",
    "5484af4472c3632d",
    "54a81d361b2a094f",
    "54be0492dc3b1386",
    "5540d5b0bb1a7b9a",
    "5566fb667f28b371",
    "558f0ac0885469ff",
    "55e3e91da83cc21a",
    "5658c4855c99f1bd",
    "566f73525ea7f571",
    "56b9ee6e6098c584",
    "56cf2e891421785b",
    "58559be16ca43743",
    "58dcdf7f4b6b4878",
    "58efeea8bb73f462",
    "58f804c4985d0131",
    "5910dd06ac368119",
    "59160e74f6de369c",
    "5918181b632cc3d4",
    "59a2f8377ce14120",
    "59e9819824aeba69",
    "5a1dc95635ec992d",
    "5a96fde1c24cb409",
    "5ab7030a02f30518",
    "5ab8717d4dd1fe90",
    "5ac7b1152960f1e4",
    "5b8fb3f69d96e95e",
    "5bcb19219734d71f",
    "5c1c35f6a889d91d",
    "5d2e85de42418849",
    "5d7ef5596447050b",
    "5d829775c3467032",
    "5df8d74d926e0fc8",
    "5e16205656482b43",
    "5e1ba77a7f73c84d",
    "5eaed1bd46029a05",
    "5f392677e7824e40",
    "602a4ced3006b80c",
    "611a52406549a9a3",
    "613634df74191426",
    "614db0132c094efa",
    "61876f06c149d4c8",
    "620e719ae9975498",
    "629ab8ccd0a3f554",
    "62b5891eb1a85b09",
    "62c5a948432a1f01",
    "631f3a57440377b2",
    "633c3483dd64f88c",
    "635a2b36939ba236",
    "6478c87cc02c8aa2",
    "6487bd211ab765aa",
    "65a90174fbf6abee",
    "661aed428bcf8b93",
    "665750c3d20adfbc",
    "674a68e938dbd25b",
    "678d22ebca4ad16a",
    "679d05f7c76f25e0",
    "67a66bcb4de5fffa",
    "67d2bd73430ff38e",
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