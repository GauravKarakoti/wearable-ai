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
TIME_BUDGET_SECONDS = 14 * 60
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
        "video_id": "248697710277f829",
        "question": "Who poured water into the French press before I pressed its plunger?",
        "answer": "The person in the denim shirt.",
    },
    {
        "video_id": "24b25f579fb8a8ff",
        "question": "After observing tire tracks and construction tools in the backyard, and later seeing a dump trailer with gravel, what equipment in the final wooded area likely created those tire tracks?",
        "answer": "The yellow skid steer in the wooded area likely created the backyard tire tracks.",
    },
    {
        "video_id": "25a487bbb0c6a04e",
        "question": "After seeing the 'SI Ferry' sign in the terminal, what color was the ferry I boarded?",
        "answer": "Orange",
    },
    {
        "video_id": "25f01498bfb69520",
        "question": "I was near a black iron fence with a water view early on, and later walked along a path with a metal railing by the water. How did I move between these two spots?",
        "answer": "I descended a staircase.",
    },
    {
        "video_id": "262682622af0eb9d",
        "question": "I passed a building with exterior stairs three times during my walk. Between the first and second pass, and between the second and third pass, I saw tennis courts. Were the tennis courts on the same side of the path in both intervals?",
        "answer": "Yes, the tennis courts were on the right side both times.",
    },
    {
        "video_id": "268678599fd83396",
        "question": "I saw two people wearing New York Rangers jerseys inside Macy's; what was the number on the jersey of the person I saw in the fine jewelry and watches section?",
        "answer": "23",
    },
    {
        "video_id": "2753599c9267a81b",
        "question": "When I saw the 'DO NOT WRITE ON SHELF PLEASE' sign for the second time, what product was I holding?",
        "answer": "A purple stationery box with cartoon designs",
    },
    {
        "video_id": "275a57728ae8ceb3",
        "question": "I saw two people walking towards the store entrance before I entered; where did I encounter them again inside the store?",
        "answer": "In the produce section near the tables with sunflower-patterned tablecloths.",
    },
    {
        "video_id": "280f3781277b0b97",
        "question": "What did I relocate from the staircase to a location near a light switch during a break from diamond painting?",
        "answer": "An elf doll",
    },
    {
        "video_id": "283c1d1c42b83e9e",
        "question": "When I first saw the AT&T building near Legends Corner, what color were its spires, and what color were they when I saw it again later near the Hilton hotel?",
        "answer": "Green then blue",
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