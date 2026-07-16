"""
train-bridge.py -- persistent, incremental bridge training.
Now with automated video downloading from Hugging Face baked into the
embedding step -- no more manually placing .mp4 files next to the script.

WORKFLOW:
  1. Put ~10 new (video_id, question, answer) entries in NEW_BATCH below.
     video_path is no longer needed -- the script downloads each video by
     video_id automatically.
  2. Run this script. It will:
       - download any NEW_BATCH videos not already embedded
       - embed them, cache the (tiny) embedding, then leave the raw .mp4
         on disk (deletion is NOT automatic -- delete manually when ready)
       - load the existing bridge checkpoint (or init fresh if none exists)
       - train on the full library (old + new) for TIME_BUDGET_SECONDS
       - save the updated checkpoint back to BRIDGE_CHECKPOINT_PATH
       - report a quick held-out eval if the library is big enough
  3. Delete downloaded videos yourself once you're done with them, from
     SAVE_DIR (see below) -- this script will not do it for you.
  4. Next session: fill NEW_BATCH with the next 10 videos, repeat.
"""

import os
import glob
import random
import time
import torch
import torch.nn as nn

from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoVideoProcessor, AutoModelForCausalLM, AutoTokenizer
from trial import load_video_frames

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NUM_VIDEO_TOKENS = 16
MAX_NEW_TOKENS = 256
TIME_BUDGET_SECONDS = 9 * 60
LEARNING_RATE = 1e-4
HELD_OUT_COUNT = 2
MIN_LIBRARY_SIZE_FOR_HOLDOUT = 6

HF_REPO_ID = "facebook/wearable-ai"
SAVE_DIR = "."

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
os.makedirs(EMBEDDING_LIBRARY_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

NEW_BATCH = [
    {
        "video_id": "19176681c6d16bdb",
        "question": "What tool did I retrieve from the kitchen drawer, and what did I use it for later in the video?",
        "answer": "I retrieved scissors from the drawer and used them to open the bag of mango chunks.",
    },
    {
        "video_id": "198ded627a6102df",
        "question": "I first saw a sign describing the Combat Information Center Area as the ship's operational 'brain'. Later, I encountered an exhibit about radarmen and the actual radar equipment. According to the sign and the radarman exhibit, what was the primary function of the Combat Information Center, and how did the radarman exhibit describe the working conditions there?",
        "answer": "The sign stated the Combat Information Center was used to identify and track ships and aircraft, and to plan attacks and make critical decisions. The radarman exhibit described dim lighting with a green glow illuminating radarmen hunched over consoles, monitoring radar scopes in the combat information center.",
    },
    {
        "video_id": "19f4457a6f327740",
        "question": "What was the name of the event I attended and the city it was in, based on the signs I saw near the bridge and later near the digital screens?",
        "answer": "Canal Convergence in Scottsdale.",
    },
    {
        "video_id": "1a5db01abab90677",
        "question": "What did I point at with my hand, and when did I next have the water bottle in my hand after that?",
        "answer": "I pointed at a bridge with red railings; I next had the water bottle in my hand when approaching two people with a dog.",
    },
    {
        "video_id": "1a732b061361ef6f",
        "question": "I saw a street name engraved on the pavement as I walked toward the pier early on, and later I noticed the venue's name on the menu at the bar. What were the street name and the venue name?",
        "answer": "JANE STREET and FRYING PAN",
    },
    {
        "video_id": "1a89eaa35120072a",
        "question": "Which pi\u00f1atas had their fringe fully applied before I began decorating the donkey-shaped pi\u00f1ata with brown fringe?",
        "answer": "The bottle and chili pepper pi\u00f1atas.",
    },
    {
        "video_id": "1a96d31d14d53569",
        "question": "What was the total duration and calorie expenditure during the interval when my treadmill speed was above 3.5 mph?",
        "answer": "1 minute 57 seconds and 16 calories",
    },
    {
        "video_id": "1ac045ca12edbb6a",
        "question": "What tool did I use to clean both the shower door track and the drain cover, and where was it at the end of the video?",
        "answer": "The toothbrush, in the bathtub.",
    },
    {
        "video_id": "1aca110c98e1d797",
        "question": "After the dog in the red sweater ran to the far end of the fenced area and returned, where was it when I extended my hand towards it?",
        "answer": "In front of me, near my legs.",
    },
    {
        "video_id": "1ae2ae78bd7cf791",
        "question": "What were the executive producers listed on the first CD case I cleaned, and who is the artist of the last CD case I cleaned?",
        "answer": "Angie Stone & Collin Stanback; Mary J. Blige",
    },
]


def library_path(video_id):
    return os.path.join(EMBEDDING_LIBRARY_DIR, f"{video_id}.pt")


def download_video(video_id):
    """Download one video by id from the HF dataset repo. Returns the local
    path, or None if the download failed (entry is skipped, not fatal)."""
    file_name = f"{video_id}.mp4"
    try:
        video_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"egolongqa/val/{file_name}",
            repo_type="dataset",
            local_dir=SAVE_DIR,
        )
        return video_path
    except Exception as e:
        print(f"  Failed to download {video_id}: {e}")
        return None


def embed_new_batch(device):
    """Download + encode only the videos not already in the embedding library,
    then free V-JEPA2 from memory. Downloaded .mp4s are left on disk --
    delete them yourself when you're done, this function will not."""
    to_embed = [ex for ex in NEW_BATCH if not os.path.exists(library_path(ex["video_id"]))]
    if not to_embed:
        print("All videos in NEW_BATCH are already in the library. Nothing to encode.")
        return

    print(f"\nLoading V-JEPA2: {JEPA_MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID, device_map="auto", attn_implementation="sdpa")
    model.eval()
    num_frames = model.config.frames_per_clip

    for ex in to_embed:
        print(f"--- {ex['video_id']} ---")
        video_path = download_video(ex["video_id"])
        if video_path is None:
            print(f"  Skipping {ex['video_id']} (download failed).")
            continue

        print(f"  Encoding {ex['video_id']} ({video_path}) ...")
        frames = load_video_frames(video_path, num_frames=num_frames)
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            out.transpose(1, 2), NUM_VIDEO_TOKENS
        ).transpose(1, 2).squeeze(0).cpu()

        torch.save(
            {"embedding": pooled, "question": ex["question"],
            "answer": ex["answer"]},
            library_path(ex["video_id"]),
        )
        print(f"  Saved embedding for {ex['video_id']}. Raw video left at {video_path} -- delete manually when ready.")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"\nEmbedded {len([e for e in to_embed if os.path.exists(library_path(e['video_id']))])} new video(s).")


def load_full_library():
    entries = []
    for path in sorted(glob.glob(os.path.join(EMBEDDING_LIBRARY_DIR, "*.pt"))):
        data = torch.load(path)
        data["video_id"] = os.path.splitext(os.path.basename(path))[0]
        entries.append(data)
    return entries


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


def build_training_batch(entry, tokenizer, bridge, qwen, device):
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    projected = bridge(jepa_tokens)

    question_prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
    answer_text = entry["answer"] + tokenizer.eos_token

    q_ids = tokenizer(question_prompt, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    q_embeds = qwen.get_input_embeddings()(q_ids)
    a_embeds = qwen.get_input_embeddings()(a_ids)

    inputs_embeds = torch.cat([projected, q_embeds, a_embeds], dim=1)
    video_len, q_len = projected.shape[1], q_ids.shape[1]
    labels = torch.cat([
        torch.full((1, video_len + q_len), -100, dtype=torch.long, device=device),
        a_ids,
    ], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attention_mask, labels


def evaluate(entry, tokenizer, bridge, qwen, device):
    bridge.eval()
    jepa_tokens = entry["embedding"].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
        projected = bridge(jepa_tokens)

    prompt = f"<|im_start|>user\n{entry['question']}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = qwen.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_ids = qwen.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS)
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    bridge.train()
    return output_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    embed_new_batch(device)
    library = load_full_library()
    print(f"\nFull library size: {len(library)} example(s)")

    held_out, train_set = [], library
    if len(library) >= MIN_LIBRARY_SIZE_FOR_HOLDOUT:
        shuffled = library[:]
        random.shuffle(shuffled)
        held_out = shuffled[:HELD_OUT_COUNT]
        held_out_ids = {e["video_id"] for e in held_out}
        train_set = [e for e in library if e["video_id"] not in held_out_ids]
        print(f"Holding out {len(held_out)} example(s) for a generalization check: "
              f"{[e['video_id'] for e in held_out]}")
    else:
        print("Library still small -- training on everything, no held-out check this run.")

    print(f"\nLoading Qwen2-VL: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    bridge = build_bridge(jepa_hidden=1024, qwen_hidden=qwen_hidden, device=device, dtype=qwen.dtype)
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining on {len(train_set)} example(s) (with replay of prior library), "
          f"time budget {TIME_BUDGET_SECONDS}s ...")
    start = time.time()
    step, epoch = 0, 0
    while time.time() - start < TIME_BUDGET_SECONDS:
        epoch += 1
        random.shuffle(train_set)
        epoch_loss = 0.0
        for entry in train_set:
            inputs_embeds, attention_mask, labels = build_training_batch(entry, tokenizer, bridge, qwen, device)
            outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            if time.time() - start >= TIME_BUDGET_SECONDS:
                break
        print(f"  epoch {epoch}  avg_loss={epoch_loss / len(train_set):.4f}  elapsed={time.time() - start:.0f}s")

    torch.save(bridge.state_dict(), BRIDGE_CHECKPOINT_PATH)
    print(f"\nTraining stopped after {step} steps. Checkpoint saved to {BRIDGE_CHECKPOINT_PATH}")

    if held_out:
        print("\n--- HELD-OUT GENERALIZATION CHECK ---")
        for entry in held_out:
            output_text = evaluate(entry, tokenizer, bridge, qwen, device)
            print(f"\nVideo: {entry['video_id']}")
            print(f"Question: {entry['question']}")
            print(f"Output: {output_text}")
            print(f"Ground truth: {entry['answer']}")

    print(f"\nDownloaded videos are in {SAVE_DIR} -- delete them yourself once you're done with this batch.")


if __name__ == "__main__":
    main()