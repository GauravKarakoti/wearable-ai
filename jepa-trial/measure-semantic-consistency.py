"""
measure-semantic-consistency.py -- does distance in JEPA's pooled embedding
space track semantic distance between clips?

WHAT THIS ACTUALLY MEASURES: for pairs of held-out examples, we compute (a)
cosine distance between their pooled JEPA embeddings, and (b) cosine distance
between their ground-truth answers' text embeddings (using Qwen's own frozen
embedding layer, mean-pooled -- no extra model needed). We then compute the
correlation between (a) and (b) across many pairs.
"""

import os
import glob
import random
import statistics
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")

SAMPLE_SIZE = 150
MAX_PAIRS = 5000
SEED = 42


def load_sample(n):
    paths = glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))
    random.shuffle(paths)
    entries = []
    for p in paths[:n]:
        data = torch.load(p)
        data["video_id"] = os.path.splitext(os.path.basename(p))[0]
        entries.append(data)
    return entries


def jepa_embedding_vector(entry):
    """Mean-pool the 16 pooled JEPA tokens into a single vector for pairwise comparison."""
    return entry["embedding"].mean(dim=0) 


def text_embedding_vector(text, tokenizer, qwen, device):
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        embeds = qwen.get_input_embeddings()(ids)
    return embeds.mean(dim=1).squeeze(0).cpu()


def cosine_distance(a, b):
    a, b = a.float(), b.float()
    sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return 1 - sim


def pearson_correlation(xs, ys):
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    std_x = (sum((x - mean_x) ** 2 for x in xs)) ** 0.5
    std_y = (sum((y - mean_y) ** 2 for y in ys)) ** 0.5
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def spearman_correlation(xs, ys):
    """Rank-based correlation -- more robust to outliers than Pearson here,
    since a handful of very distant pairs could otherwise dominate."""
    def rank(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank_pos, i in enumerate(sorted_idx):
            ranks[i] = rank_pos
        return ranks
    return pearson_correlation(rank(xs), rank(ys))


def main():
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading text backbone: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=torch.float32, device_map="auto")
    qwen.eval()

    print(f"Sampling {SAMPLE_SIZE} videos from the embedding library...")
    entries = load_sample(SAMPLE_SIZE)
    print(f"Loaded {len(entries)} entries.")

    print("Computing per-video JEPA and text embedding vectors...")
    for e in entries:
        e["jepa_vec"] = jepa_embedding_vector(e)
        e["text_vec"] = text_embedding_vector(e["answer"], tokenizer, qwen, device)

    pairs = [(i, j) for i in range(len(entries)) for j in range(i + 1, len(entries))]
    random.shuffle(pairs)
    pairs = pairs[:MAX_PAIRS]
    print(f"Evaluating {len(pairs)} pairs...")

    jepa_distances, text_distances = [], []
    for i, j in pairs:
        jepa_distances.append(cosine_distance(entries[i]["jepa_vec"], entries[j]["jepa_vec"]))
        text_distances.append(cosine_distance(entries[i]["text_vec"], entries[j]["text_vec"]))

    pearson_r = pearson_correlation(jepa_distances, text_distances)
    spearman_r = spearman_correlation(jepa_distances, text_distances)

    print("\n--- SEMANTIC DISTANCE CONSISTENCY ---")
    print(f"Pairs evaluated: {len(pairs)}")
    print(f"Mean JEPA embedding distance: {statistics.mean(jepa_distances):.4f}")
    print(f"Mean answer-text embedding distance: {statistics.mean(text_distances):.4f}")
    print(f"Pearson correlation (JEPA distance vs text distance): {pearson_r:.4f}")
    print(f"Spearman correlation (JEPA distance vs text distance): {spearman_r:.4f}")
    print("\nInterpretation: values near 0 mean JEPA-embedding proximity does NOT reliably")
    print("track semantic (answer-text) similarity. Values approaching 1 mean it does.")
    print("There is no established 'good' threshold for this -- report it descriptively,")
    print("not against some external cutoff.")


if __name__ == "__main__":
    main()