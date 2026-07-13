"""
V-JEPA2 -> Qwen2-VL naive bridge - plumbing sanity check.

WHAT THIS IS:
  A mechanical wiring test. It projects V-JEPA2's video embeddings into
  Qwen2-VL's embedding space with a RANDOMLY INITIALIZED, UNTRAINED linear
  layer, then feeds that + the question straight into Qwen2-VL's generate().
"""

import os
import torch
import torch.nn as nn

from transformers import AutoModel, AutoVideoProcessor, Qwen2VLForConditionalGeneration, AutoTokenizer
from trial import load_video_frames

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VIDEO_SOURCE = "sample.mp4"
QUESTION = "What was the first product I interacted with and the last product I interacted with in the video?"
GROUND_TRUTH_ANSWER = "A greeting card with a taco and margarita design and a teal short-sleeve shirt."
NUM_VIDEO_TOKENS = 16
MAX_NEW_TOKENS = 256
SEED = 42

OUTPUT_DIR = "./trial_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_jepa_embeddings(device):
    print(f"\nLoading V-JEPA2: {JEPA_MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID, device_map="auto", attn_implementation="sdpa")
    model.eval()

    num_frames = model.config.frames_per_clip
    frames = load_video_frames(VIDEO_SOURCE, num_frames=num_frames)
    inputs = processor(frames, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    encoder_out = outputs.last_hidden_state
    print(f"JEPA encoder output shape: {tuple(encoder_out.shape)}")

    pooled = torch.nn.functional.adaptive_avg_pool1d(
        encoder_out.transpose(1, 2), NUM_VIDEO_TOKENS
    ).transpose(1, 2)
    print(f"Pooled JEPA tokens shape: {tuple(pooled.shape)}")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return pooled


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    jepa_tokens = get_jepa_embeddings(device).to(device)

    print(f"\nLoading Qwen2-VL: {QWEN_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qwen = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    qwen.eval()

    qwen_hidden_size = qwen.get_input_embeddings().weight.shape[1]
    jepa_hidden_size = jepa_tokens.shape[-1]
    print(f"JEPA hidden size: {jepa_hidden_size}  |  Qwen hidden size: {qwen_hidden_size}")

    bridge = nn.Linear(jepa_hidden_size, qwen_hidden_size).to(device=device, dtype=qwen.dtype)
    print("\n*** WARNING: bridge layer is randomly initialized and UNTRAINED. ***")
    print("*** Output below is expected to be incoherent. This is a plumbing test only. ***\n")

    projected_video_tokens = bridge(jepa_tokens.to(qwen.dtype))

    prompt = f"<|im_start|>user\n{QUESTION}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    text_embeds = qwen.get_input_embeddings()(text_ids)

    inputs_embeds = torch.cat([projected_video_tokens, text_embeds], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    print(f"Combined inputs_embeds shape: {tuple(inputs_embeds.shape)}")
    print("Running generation from raw inputs_embeds...")

    with torch.no_grad():
        generated_ids = qwen.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("\n--- RESULTS (untrained bridge) ---")
    print(f"Question: {QUESTION}")
    print(f"Qwen output with naive JEPA bridge:\n{output_text}")
    print(f"\nGround truth (for reference, not for the model to have seen):\n{GROUND_TRUTH_ANSWER}")

    with open(os.path.join(OUTPUT_DIR, "jepa-to-qwen-untrained-result.txt"), "w") as f:
        f.write("NOTE: bridge layer is untrained/random. Output is expected to be incoherent.\n")
        f.write(f"Video: {VIDEO_SOURCE}\n")
        f.write(f"Question: {QUESTION}\n")
        f.write(f"Output: {output_text}\n")
        f.write(f"Ground truth: {GROUND_TRUTH_ANSWER}\n")

    torch.save(bridge.state_dict(), os.path.join(OUTPUT_DIR, "untrained_bridge.pt"))
    print(f"\nSaved result + untrained bridge weights to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()