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
    "680244a66d976e3f",
    "683ab333894b2c06",
    "6874bcc40ab7ace7",
    "68be854a3b8f3d47",
    "68c23dcc96acbfbf",
    "68db3e8d1d963162",
    "68f002c1fab08f95",
    "69afe7c01d4fb660",
    "6aa5674e6c7d3872",
    "6ae5728b361779aa",
    "6b4f03af65f82192",
    "6b8c690b8133259a",
    "6befb649e6eb4570",
    "6c34f96dae9476ce",
    "6cdb8f146c432312",
    "6d5b8909cd6e4538",
    "6db90244402799f4",
    "6dcb7f9fc04a3283",
    "6e12cc003de06176",
    "6e4610f800b157a0",
    "6f0d26d3e3eb2ff1",
    "6fc33b49f0442275",
    "703854b6d93f2b12",
    "70d766f186cbc870",
    "70d8108d8287f08e",
    "719245d9778f41a8",
    "72f6fb2b3fcd5b08",
    "73794817c7d14657",
    "7392bac19ba7eec5",
    "741fcd43a96bd40e",
    "743a4bc4d1c10be9",
    "74edd72c3cbee70f",
    "754bd3fb63570b56",
    "7593bd8121dcfd39",
    "75999a9fb104b1b9",
    "75c6498c432bf077",
    "76bbd3472a7a2dd0",
    "76cd872d8785dae8",
    "77342df1fec5ffff",
    "777459a083922176",
    "77d6d602f07f3bc8",
    "786fc2d06d1a1893",
    "794ad013ff1e18de",
    "7a189ceba6cd2319",
    "7a1d02a893982946",
    "7a5ad35cb65315bb",
    "7a5c4239bc6b4a33",
    "7a8bc6855c83c2eb",
    "7ac4e2e4bc2ecc3d",
    "7b25699f021f7fa4",
    "7b371c444bd5c2d8",
    "7b4dd3a4cbcb9582",
    "7b4fb7f6da738501",
    "7b74a542cd93fefa",
    "7bfeb3af24f17ceb",
    "7c8b228dd61b135f",
    "7caf4511de2b8bee",
    "7cb70948d87b16e8",
    "7cf85be7b281d6aa",
    "7d218c229be9704a",
    "7d5b6c6b5100e141",
    "7de6c8819211d385",
    "7fb5cd178cf07686",
    "80150ed4385f86a9",
    "801add32a19fca1b",
    "807c58c65ef0c3a7",
    "81136397be32b026",
    "81c514e34862c8e2",
    "8314bff1c8dd2833",
    "83459197ce58ee6d",
    "835c766eaa878983",
    "83af08b0c9b7bc0f",
    "83f218905a680d46",
    "8421a42d1b5d314a",
    "842dab1a391dc72c",
    "8489ba53a98bcdbe",
    "84b67b4c525855bb",
    "86897310591a3dc6",
    "86cf704fac7bb2ba",
    "871fb7842cf193f1",
    "8765a7c347d3f82b",
    "88052b7bca179718",
    "8812f05ea4220629",
    "88683ac7ab86baf2",
    "886d2a6075229794",
    "888ab584bea8ddee",
    "88c14ad96aa2ed09",
    "88d3fe74ee770126",
    "88e8cf11f150ba5e",
    "88ee65ccc88209f0",
    "89992c79cb8a29d1",
    "89d686a5c3e5637f",
    "89eed8c271024825",
    "8a2b0052cec40e85",
    "8a3a19f9c55cf5a9",
    "8a6e23fa7a8e172d",
    "8acd2b044cc83d2d",
    "8b02497f95b7204c",
    "8b1c280ae366ad2d",
    "8bbb3df4bc8cfd84",
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