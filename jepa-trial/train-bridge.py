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
        "video_id": "03a44264a5df1c9e",
        "video_path": "03a44264a5df1c9e.mp4",
        "question": "What was the man in the black tracksuit with white stripes doing the first time I saw him, and what was he doing the last time I saw him on the courts?",
        "answer": "First, he was sitting on a bench tying his shoe; last, he was playing tennis on the opposite side of the net.",
    },
    {
        "video_id": "03fb89df97cc9908",
        "video_path": "03fb89df97cc9908.mp4",
        "question": "After I left the store and started driving, I used my phone to search for something. What did I search for, and what dashboard warning prompted this search?",
        "answer": "I searched for 'auto parts', prompted by the 'Service Tire Monitor System' warning on the dashboard.",
    },
    {
        "video_id": "056734786b556b0f",
        "video_path": "056734786b556b0f.mp4",
        "question": "What was the design on the station roof sign before I attached the station to the track, and what design replaced it after the station was on the track?",
        "answer": "The roof initially had a sign with a burger, fish, and coffee cup; it was later replaced with a sign showing the numbers 1, 2, 3.",
    },
    {
        "video_id": "059a88dee3d0bddc",
        "video_path": "059a88dee3d0bddc.mp4",
        "question": "Which ingredient that I retrieved from the refrigerator did I chop and add to the pan with ground meat before using the last of the tomato sauce?",
        "answer": "Carrots",
    },
    {
        "video_id": "06615b19ef5373ca",
        "video_path": "06615b19ef5373ca.mp4",
        "question": "Which did I paint first, the house's exterior siding or the adjacent wooden fence, and what visual evidence in the video indicates the order?",
        "answer": "The exterior siding; the fence only shows fresh teal paint in the final frames while the siding is already fully painted teal in earlier frames.",
    },
    {
        "video_id": "069e0b5942b685da",
        "video_path": "069e0b5942b685da.mp4",
        "question": "When I was in the parking lot, I noticed a tree with red leaves in the distance. Later, when I walked past that same tree up close in a grassy area, what was located beneath it?",
        "answer": "A picnic table",
    },
    {
        "video_id": "07092e5a5ee9579c",
        "video_path": "07092e5a5ee9579c.mp4",
        "question": "What was the name of the store with the red awning that I passed before I reached the bus stop at 8 Avenue & 60 Street?",
        "answer": "G.E.DIGITAL PHOTO",
    },
    {
        "video_id": "0789bad7468e72b3",
        "video_path": "0789bad7468e72b3.mp4",
        "question": "After the woman finished examining scarves, what was the first clothing item she looked at, and what was the last clothing item she interacted with in the video?",
        "answer": "A navy blazer and a green sweater.",
    },
    {
        "video_id": "07f54b96296acdff",
        "video_path": "07f54b96296acdff.mp4",
        "question": "Between the time I saw a white van on the opposite side of the road and the time I saw a white car on the opposite side, what construction equipment did I pass on the left side of the road?",
        "answer": "An orange excavator",
    },
    {
        "video_id": "046f557aae4c377b",
        "video_path": "046f557aae4c377b.mp4",
        "question": "What did I point at earlier that the woman later ate with a spoon?",
        "answer": "The yellow soup in the pink bowl.",
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