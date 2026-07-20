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
    "b32b516fae8b102c",
    "b38d58abcd059eeb",
    "b3fba3c9b747e519",
    "b437216f70a77454",
    "b5bc3f73ae322ee9",
    "b62094c953346377",
    "b670fb214bff9a2b",
    "b67686a226da2f5a",
    "b6ad8e6dbda2713a",
    "b724c4870790d192",
    "b73ee3f206e0f0c2",
    "b76b21f72125b1ca",
    "b76d10dcd8243a01",
    "b7b62d576af7c64d",
    "b81309c658782384",
    "b863b8980be825e3",
    "b90eb2cedaf6975d",
    "ba88525691af500d",
    "bb165f4c81827b82",
    "bbf5f37b642d5756",
    "bbf6e152b6ee763a",
    "bc1c13c75646fb2f",
    "bc484bf7424a0679",
    "bc6cc236f240d2cd",
    "bcd0a869b9339fe2",
    "bde055e6fe9d2c08",
    "be18a5e734f021c7",
    "be368a6954a25857",
    "bea72d159e1eca4b",
    "bf2a33ef5e329708",
    "c037cfcb6d618e43",
    "c060b9a8c96fcc0f",
    "c07cbdb66d404af3",
    "c10c12038a112a1a",
    "c15faf882b2c3529",
    "c1c0e982b39c5ffb",
    "c21062e32dcbecee",
    "c663d84f69482b36",
    "c6866ef535b21a76",
    "c68d6e65e566b11f",
    "c7403ae0345a15f9",
    "c8661556555ace32",
    "c87158fb587b4beb",
    "c894c42c5e02dec9",
    "c8cb286a8a6583c7",
    "c8fb0b8e48aaba3c",
    "c919300af537a381",
    "c965b61791599ed8",
    "ca6c5a6e706c2094",
    "cb85d5789afca909",
    "cc37909ac018919c",
    "cc843edff14ff1a5",
    "cd3527f83b3036e2",
    "cdc003099b8dd3c5",
    "ce5f716477b5bf44",
    "ce7e498bea4c4036",
    "ce8d213682d87166",
    "cec98dddd7ad2495",
    "cf2bd89a7b5d37d8",
    "cfd872ca200e9529",
    "d00b07b27508e864",
    "d09b1339fe1d32dc",
    "d0ad286d0fd978f7",
    "d0db0bf6b9ab74e5",
    "d12791aa7c6e8d2f",
    "d1a6067a7a75a228",
    "d2a5b3524afba5e0",
    "d48d1679faa0819e",
    "d4ad94f700885cee",
    "d5bbef6bd26a6384",
    "d5f033c63f09b95a",
    "d64ed0d63b455510",
    "d67e1186a658d9bc",
    "d6921118cad55955",
    "d73174bdd7a22aaf",
    "d89f42607516218a",
    "d8c6c71f8aded7f4",
    "d8f1acbcce1e4747",
    "d938c527010d0246",
    "d9db6fb1cdf4bc7d",
    "d9e802102d6400ba",
    "dbafb0e35a78a21d",
    "dbc5348e4a5ed4da",
    "dc47e12181b4f76e",
    "dc93aa20988701ba",
    "dcc8378bca4fb4d8",
    "dcdb8d8cd7889358",
    "dd33e427eabbe593",
    "dd47023168ffb6cf",
    "dd61f1e40f3be0f4",
    "dde4effbbc55e8a0",
    "deacfa3d4686dfa2",
    "df020e5f1eb92edd",
    "df5f62c7c1e941d6",
    "df8fe3a9b2744d7d",
    "e01d486f76bc5d26",
    "e0350daea52c7ddf",
    "e0641c46302f6c4a",
    "e0671b2368ea4cd6",
    "e160e5031b2b7f7b",
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