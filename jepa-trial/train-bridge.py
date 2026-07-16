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
        "video_id": "14a8ff92adb183c7",
        "question": "What tools did I use to clean up the spilled beans before washing them in the sink?",
        "answer": "A broom and a handheld vacuum",
    },
    {
        "video_id": "156cc97ac25e1078",
        "question": "When I first saw the person in the blue shirt, what were they doing, and after I looked at the fence on the opposite side of the bridge from the river, what were they doing when I saw them again?",
        "answer": "They were leaning on the railing looking at the river; they were walking away from me.",
    },
    {
        "video_id": "156e2e7dc7ea203a",
        "question": "What item was in my cart when I first browsed the candle section and still there when I later examined Elf on the Shelf products?",
        "answer": "The dip mix collection",
    },
    {
        "video_id": "15ac76d47c8753f9",
        "question": "Why didn't the two people I saw on the concrete path later react to the snake I encountered earlier, and where was the snake in relation to the path?",
        "answer": "The snake was in the undergrowth off the concrete path, so the people on the path did not encounter it.",
    },
    {
        "video_id": "15eeddc94cbfe9e3",
        "question": "After passing the mural of Michael K. Williams on the brick building, what prominent monument did I later see in the park I entered?",
        "answer": "Prison Ship Martyrs' Monument",
    },
    {
        "video_id": "166ed6f9ba6e5dea",
        "question": "What piece of equipment, which I later saw on the sidewalk, likely made the tire tracks visible in both the backyard and front lawn?",
        "answer": "The small skid steer loader",
    },
    {
        "video_id": "17628f735391242b",
        "question": "What did the app identify when I first lifted the brick, and what did I do with that brick later?",
        "answer": "Insects; I placed it back in its original position.",
    },
    {
        "video_id": "17ddbc3d41cea677",
        "question": "I first saw the modern building with a diamond-patterned facade from behind a black fence, and later from a bridge over railroad tracks. What did I do between these two views to reach the higher vantage point?",
        "answer": "I rode the elevator.",
    },
    {
        "video_id": "17ffed5d482095b5",
        "question": "What did I take from the red structure I encountered between the Bruer Building and the Moss Building?",
        "answer": "A coloring page",
    },
    {
        "video_id": "1881e98bd8759b6f",
        "question": "Where did I spread the yellow granules both before and after I had to refill the blue bucket?",
        "answer": "Around the curved stone planter in the backyard.",
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