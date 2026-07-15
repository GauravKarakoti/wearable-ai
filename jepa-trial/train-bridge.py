"""
train-bridge.py -- persistent, incremental bridge training.

WORKFLOW:
  1. Put ~10 new (video, question, answer) entries in NEW_BATCH below,
     with the videos physically present at those paths.
  2. Run this script. It will:
       - embed any NEW_BATCH videos not already in the library
       - load the existing bridge checkpoint (or init fresh if none exists)
       - train on the full library (old + new) for TIME_BUDGET_SECONDS
       - save the updated checkpoint back to BRIDGE_CHECKPOINT_PATH
       - report a quick held-out eval if the library is big enough
  3. Delete the raw videos you just embedded (their embeddings are safely
     cached in EMBEDDING_LIBRARY_DIR now) to free disk space.
  4. Next session: fill NEW_BATCH with the next 10 videos, repeat.
"""

import os
import glob
import random
import time
import torch
import torch.nn as nn

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

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library")
BRIDGE_CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "persistent_bridge.pt")
os.makedirs(EMBEDDING_LIBRARY_DIR, exist_ok=True)

NEW_BATCH = [
    {
        "video_id": "00cdb8ca10c069f8",
        "video_path": "00cdb8ca10c069f8.mp4",
        "question": "After I left the grassy trail that ran alongside the flooded marsh, what was the first man-made structure I encountered on the paved section of my walk?",
        "answer": "A concrete bench on the left side of the path.",
    },
    {
        "video_id": "016c4ad8fda54a97",
        "video_path": "016c4ad8fda54a97.mp4",
        "question": "What type of recreational area did I encounter after passing the child on the scooter and descending the stairs?",
        "answer": "A fenced playground with swings.", 
    },
    {
        "video_id": "017ffe52c5b5230c",
        "video_path": "017ffe52c5b5230c.mp4",
        "question": "After I walked past the vending machines, what was I holding when I reached the lake area?",
        "answer": "A gray cup", 
    },
    {
        "video_id": "0207cc5be310f3ba",
        "video_path": "0207cc5be310f3ba.mp4",
        "question": "What scent was the first candle I picked up, and what shape were the discounted candles I examined later in the seasonal section?",
        "answer": "Cashmere Vanilla; pumpkin-shaped.",
    },
    {
        "video_id": "02216818d2099c04",
        "video_path": "02216818d2099c04.mp4",
        "question": "After passing a baseball field and then a soccer field in the park, what building did I see on the street outside?",
        "answer": "Museum of the City of New York", 
    },
    {
        "video_id": "02597f9f4c0ec8cc",
        "video_path": "02597f9f4c0ec8cc.mp4",
        "question": "After I entered the building through the revolving doors, what was the first display I encountered, and which store that I had seen on the exterior did I later see again inside?",
        "answer": "The Hedley Studios classic car display; Patek Philippe", 
    },
    {
        "video_id": "02fb50a660ff1072",
        "video_path": "02fb50a660ff1072.mp4",
        "question": "What did the historical marker I read at the entrance say about the 'wooden' railings I later saw on the stone bridge?",
        "answer": "They were made of concrete using the faux bois (fake wood) technique by Dionicio Rodriguez.", 
    },
    {
        "video_id": "032392866755963e",
        "video_path": "032392866755963e.mp4",
        "question": "When I first arrived, I saw a red car parked alone near the museum building. Later, as I walked back toward the museum along a tree-lined path, I saw the same red car again. What was different about the red car\u2019s situation the second time?",
        "answer": "A black car was parked next to it.", 
    },
    {
        "video_id": "03481185b7c75ab9",
        "video_path": "03481185b7c75ab9.mp4",
        "question": "I first interacted with a product that had a 'Scratch for Scent' sticker and later with a product that had a '$2.50 back' sticker. What were these products?",
        "answer": "Filter Fresh Whole Home Air Freshener and Instant Power Main Line Cleaner.", 
    },
    {
        "video_id": "0391ffc9c8c38c67",
        "video_path": "0391ffc9c8c38c67.mp4",
        "question": "What was the name on the first sign I saw after stepping onto the sidewalk, and what speed limit was indicated on the last sign I saw before stepping off the sidewalk?",
        "answer": "Jesse Bolling Hall; 25", 
    }
]


def library_path(video_id):
    return os.path.join(EMBEDDING_LIBRARY_DIR, f"{video_id}.pt")


def embed_new_batch(device):
    """Encode only the videos not already in the library, then free V-JEPA2 from memory."""
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
        print(f"  Encoding {ex['video_id']} ({ex['video_path']}) ...")
        frames = load_video_frames(ex["video_path"], num_frames=num_frames)
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            out.transpose(1, 2), NUM_VIDEO_TOKENS
        ).transpose(1, 2).squeeze(0).cpu()

        torch.save(
            {"embedding": pooled, "question": ex["question"], "answer": ex["answer"]},
            library_path(ex["video_id"]),
        )

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"Embedded {len(to_embed)} new video(s). You can now delete their raw .mp4 files.")


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

    print(f"\nRaw videos safe to delete now (already embedded): "
          f"{[ex['video_path'] for ex in NEW_BATCH]}")


if __name__ == "__main__":
    main()