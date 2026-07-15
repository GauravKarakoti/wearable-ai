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
        "video_id": "08910129d9d6ff34",
        "video_path": "08910129d9d6ff34.mp4",
        "question": "What container did I fill with water in the kitchen that I later carried through the hallway?",
        "answer": "The black watering can.", "mcq_options": "A. The clear water filter pitcher. B. The black watering can. C. The copper pot from the sink. D. The stainless steel kettle.",
    },
    {
        "video_id": "08a129d4d3455f91",
        "video_path": "08a129d4d3455f91.mp4",
        "question": "After using the clear plastic container to rinse the tomato at the sink, what did I later use that same container for?",
        "answer": "To hold and beat the eggs.", "mcq_options": "A. To store the diced tomatoes. B. To collect the garlic and onion peels. C. To hold and beat the eggs. D. To hold the red hot seasoning.",
    },
    {
        "video_id": "09293dae75c227b2",
        "video_path": "09293dae75c227b2.mp4",
        "question": "Why did I open the refrigerator near the end of the video, and what earlier activity prompted this?",
        "answer": "I opened the refrigerator to store the ice trays I had filled with water at the sink earlier.", "mcq_options": "A. I opened the refrigerator to get milk for the iced tea I was preparing with the tea bag and tumbler earlier. B. I opened the refrigerator to put away the electric kettle after it finished boiling water for the tea. C. I opened the refrigerator to store the ice trays I had filled with water at the sink earlier. D. I opened the refrigerator to store the leftover tea from the stainless steel tumbler I had rinsed at the sink.",
    },
    {
        "video_id": "09ea3872eb883ec1",
        "video_path": "09ea3872eb883ec1.mp4",
        "question": "Later in my walk, I saw a tram with an illustration on its side. What was the illustration, and which earlier landmark did it correspond to?",
        "answer": "The tram had an illustration of the White Rabbit, which corresponded to the White Rabbit statue I passed earlier near the circular fountain.", "mcq_options": "A. The tram had an illustration of a dodo bird, which corresponded to the dodo bird sign I saw earlier near the grassy hill. B. The tram had an illustration of a pumpkin, which corresponded to the jack-o'-lantern display I passed earlier near the wooden fence. C. The tram had an illustration of the White Rabbit, which corresponded to the White Rabbit statue I passed earlier near the circular fountain. D. The tram had an illustration of a caucus race, which corresponded to the 'Run your best caucus race!' sign I saw earlier.",
    },
    {
        "video_id": "0a7538816b38423d",
        "video_path": "0a7538816b38423d.mp4",
        "question": "Earlier I saw a building with a sign reading 'Jerry's Rogue River Museum & Gift Shop'. Later, at a different location, I read a sign indicating where that museum is situated relative to my current position. What did the later sign say about the museum's location?",
        "answer": "The sign said the museum was 'Right across the street' from the tour office.", "mcq_options": "A. The sign said the museum was 'In the same building' as the tour office. B. The sign said the museum was 'Right across the street' from the tour office. C. The sign said the museum was 'Next to the dock' where the jet boats are moored. D. The sign said the museum was 'Behind the main parking lot' near the forested hills.",
    },
    {
        "video_id": "0b38cc26c3cf4364",
        "video_path": "0b38cc26c3cf4364.mp4",
        "question": "I moved an object from the floor to the kitchen island during the video. What was the object and where was it first located?",
        "answer": "A blue circular lid; on the wooden floor near the stove.", "mcq_options": "A. A clear plastic bowl; on the wooden floor near the stove. B. A blue circular lid; on the pantry shelf next to the refrigerator. C. A blue circular lid; on the wooden floor near the stove. D. A glass jar with off-white substance; on the wooden floor near the dishwasher.",
    },
    {
        "video_id": "0b941d85cf228741",
        "video_path": "0b941d85cf228741.mp4",
        "question": "What institution was associated with the building visible behind the Christmas tree I saw in the park, and what signage later on the street confirmed this?",
        "answer": "The New York Public Library, confirmed by 'New York Public Library' banners on buildings along the street.", "mcq_options": "A. The Bank of America, confirmed by 'Bank of America Winter Village' banner near the ice rink. B. The Rockefeller Center, confirmed by 'ROCK' signage on a construction-covered building. C. The New York Public Library, confirmed by 'New York Public Library' banners on buildings along the street. D. The CityMD clinic, confirmed by 'CityMD' signage on Madison Avenue.",
    },
    {
        "video_id": "0c2a202bcee2ec87",
        "video_path": "0c2a202bcee2ec87.mp4",
        "question": "After I first saw the Willis Tower and Graceland replicas in Miniland, I later climbed a series of stairs with green treads and blue railings. What was the name of the attraction I reached at the top of those stairs, and what Miniland landmark did I view just before I started climbing?",
        "answer": "Ninjago The Ride; the United States Capitol building.", "mcq_options": "A. LEGO Castle; the Washington Monument. B. Miniland Delights; the Jefferson Memorial. C. Ninjago The Ride; the United States Capitol building. D. LEGO City Water Playground; the Lincoln Memorial.",
    },
    {
        "video_id": "0cf8454cd2d6f863",
        "video_path": "0cf8454cd2d6f863.mp4",
        "question": "Which character did I first see on a large digital screen and later encounter as a physical statue near the stairs?",
        "answer": "Mario", "mcq_options": "A. Link B. Kirby C. Pikachu D. Mario",
    },
    {
        "video_id": "0de357266f10df86",
        "video_path": "0de357266f10df86.mp4",
        "question": "The first time I saw the man in the black jacket with another person, who was he with, and who was he with the last time I saw him with another person?",
        "answer": "First with a woman in black clothing and white sneakers; last with a woman in a gray hoodie and black leggings.", "mcq_options": "A. First with a woman in gray hoodie and black leggings; last with a woman in black clothing and white sneakers. B. First with a woman in black clothing and white sneakers; last with a person with a tattooed arm, black pants, and white sneakers. C. First with a woman in black clothing and white sneakers; last with a woman in a gray hoodie and black leggings. D. First with a woman in black clothing and white sneakers; last with a woman in a light gray hoodie and blue jeans.",
    },
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