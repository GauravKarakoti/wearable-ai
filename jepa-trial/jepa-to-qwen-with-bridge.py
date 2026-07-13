"""
V-JEPA2 -> Qwen2-VL bridge - QUICK training run (~10 min budget).

WHAT THIS IS:
  Trains ONLY the small projector MLP (V-JEPA2's own paper uses exactly this
  design: a 2-layer MLP) that maps V-JEPA2 video embeddings into Qwen2-VL's
  embedding space. Both V-JEPA2 and Qwen2-VL stay completely FROZEN -- only
  the tiny bridge gets gradient updates. That's what makes this fast.
"""

import os
import time
import torch
import torch.nn as nn

from transformers import AutoModel, AutoVideoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from trial import load_video_frames

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
NUM_VIDEO_TOKENS = 16
MAX_NEW_TOKENS = 256
SEED = 42
TIME_BUDGET_SECONDS = 9 * 60
LEARNING_RATE = 1e-4

# The more the better
TRAINING_EXAMPLES = [
    {
        "video": "training.mp4",
        "question": "When I opened the drawer labeled 'MAGNETIC STIR BARS' and later the drawer labeled 'ZETASIZER ESSENTIALS', what discrepancy did I observe between the labels and their contents?",
        "answer": "The 'MAGNETIC STIR BARS' drawer contained Malvern Zetasizer Nano Series boxes, while the 'ZETASIZER ESSENTIALS' drawer held unrelated items like a Trypsin box and instruction manuals."
    },
]

# EVAL_VIDEO = "sample.mp4"
# EVAL_QUESTION = "What was the first product I interacted with and the last product I interacted with in the video?"
# EVAL_GROUND_TRUTH = "A greeting card with a taco and margarita design and a teal short-sleeve shirt."
EVAL_VIDEO = "training.mp4"
EVAL_QUESTION = "When I opened the drawer labeled 'MAGNETIC STIR BARS' and later the drawer labeled 'ZETASIZER ESSENTIALS', what discrepancy did I observe between the labels and their contents?"
EVAL_GROUND_TRUTH = "The 'MAGNETIC STIR BARS' drawer contained Malvern Zetasizer Nano Series boxes, while the 'ZETASIZER ESSENTIALS' drawer held unrelated items like a Trypsin box and instruction manuals."

OUTPUT_DIR = "./trial_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def precompute_jepa_embeddings(video_paths, device):
    """Run V-JEPA2 once per unique video, cache pooled embeddings, then free the model."""
    print(f"\nLoading V-JEPA2: {JEPA_MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID, device_map="auto", attn_implementation="sdpa")
    model.eval()
    num_frames = model.config.frames_per_clip

    cache = {}
    for path in set(video_paths):
        print(f"  Encoding {path} ...")
        frames = load_video_frames(path, num_frames=num_frames)
        inputs = processor(frames, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            out.transpose(1, 2), NUM_VIDEO_TOKENS
        ).transpose(1, 2)
        cache[path] = pooled.squeeze(0).cpu()

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print("V-JEPA2 freed from memory. All video embeddings cached.")
    return cache


def build_training_batch(example, embedding_cache, tokenizer, bridge, qwen, device):
    """One (video, question, answer) example -> inputs_embeds + labels for teacher-forced LM loss."""
    jepa_tokens = embedding_cache[example["video"]].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    projected = bridge(jepa_tokens)

    question_prompt = f"<|im_start|>user\n{example['question']}<|im_end|>\n<|im_start|>assistant\n"
    answer_text = example["answer"] + tokenizer.eos_token

    q_ids = tokenizer(question_prompt, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    q_embeds = qwen.get_input_embeddings()(q_ids)
    a_embeds = qwen.get_input_embeddings()(a_ids)

    inputs_embeds = torch.cat([projected, q_embeds, a_embeds], dim=1)

    video_len, q_len, a_len = projected.shape[1], q_ids.shape[1], a_ids.shape[1]
    labels = torch.cat([
        torch.full((1, video_len + q_len), -100, dtype=torch.long, device=device),
        a_ids,
    ], dim=1)

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
    return inputs_embeds, attention_mask, labels


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    all_videos = [ex["video"] for ex in TRAINING_EXAMPLES] + [EVAL_VIDEO]
    embedding_cache = precompute_jepa_embeddings(all_videos, device)

    print(f"\nLoading Qwen2-VL: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False

    qwen_hidden = qwen.get_input_embeddings().weight.shape[1]
    jepa_hidden = 1024

    bridge = nn.Sequential(
        nn.Linear(jepa_hidden, qwen_hidden),
        nn.GELU(),
        nn.Linear(qwen_hidden, qwen_hidden),
    ).to(device=device, dtype=qwen.dtype)

    optimizer = torch.optim.AdamW(bridge.parameters(), lr=LEARNING_RATE)

    print(f"\nTraining bridge on {len(TRAINING_EXAMPLES)} example(s), time budget {TIME_BUDGET_SECONDS}s ...")
    if len(TRAINING_EXAMPLES) <= 2:
        print("*** WARNING: very few examples -- this might not generalize properly. ***")

    start = time.time()
    step = 0
    epoch = 0
    while time.time() - start < TIME_BUDGET_SECONDS:
        epoch += 1
        epoch_loss = 0.0
        for example in TRAINING_EXAMPLES:
            inputs_embeds, attention_mask, labels = build_training_batch(
                example, embedding_cache, tokenizer, bridge, qwen, device
            )
            outputs = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if time.time() - start >= TIME_BUDGET_SECONDS:
                break

        print(f"  epoch {epoch}  avg_loss={epoch_loss / len(TRAINING_EXAMPLES):.4f}  "
              f"elapsed={time.time() - start:.0f}s")

    print(f"\nTraining stopped after {step} steps, {time.time() - start:.0f}s.")
    torch.save(bridge.state_dict(), os.path.join(OUTPUT_DIR, "trained_bridge.pt"))
    print(f"Saved trained bridge to {OUTPUT_DIR}/trained_bridge.pt")

    # bridge.load_state_dict(
    #     torch.load(
    #         os.path.join(OUTPUT_DIR, "trained_bridge.pt"),
    #         map_location=device,
    #     )
    # )

    print("\n--- EVAL with trained bridge ---")
    bridge.eval()
    jepa_tokens = embedding_cache[EVAL_VIDEO].to(device=device, dtype=qwen.dtype).unsqueeze(0)
    with torch.no_grad():
        projected = bridge(jepa_tokens)

    prompt = f"<|im_start|>user\n{EVAL_QUESTION}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = qwen.get_input_embeddings()(text_ids)
    inputs_embeds = torch.cat([projected, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    with torch.no_grad():
        generated_ids = qwen.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS
        )

        print(generated_ids)
        print(generated_ids.shape)
        print(tokenizer.batch_decode(generated_ids, skip_special_tokens=False))
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"Question: {EVAL_QUESTION}")
    print(f"Trained-bridge output:\n{output_text}")
    print(f"\nGround truth:\n{EVAL_GROUND_TRUTH}")

    with open(os.path.join(OUTPUT_DIR, "jepa-to-qwen-trained-result.txt"), "w") as f:
        f.write(f"Trained on {len(TRAINING_EXAMPLES)} example(s), {step} steps, {time.time() - start:.0f}s\n")
        f.write(f"Question: {EVAL_QUESTION}\n")
        f.write(f"Output: {output_text}\n")
        f.write(f"Ground truth: {EVAL_GROUND_TRUTH}\n")
    print(f"\nSaved eval result to {OUTPUT_DIR}/jepa-to-qwen-trained-result.txt")

if __name__ == "__main__":
    main()