"""
Qwen2-VL - one-video direct QA trial run.

What this does:
  1. Loads Qwen2-VL-2B-Instruct (natively multimodal: video + text -> text).
  2. Feeds it ONE video + ONE question, straight up. No bridge, no JEPA,
     no training needed -- Qwen already understands video natively.
  3. Prints the model's actual generated text answer.
"""

import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from trial import load_video_frames

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VIDEO_SOURCE = "sample.mp4"
QUESTION = "What was the first product I interacted with and the last product I interacted with in the video?"
GROUND_TRUTH_ANSWER = "A greeting card with a taco and margarita design and a teal short-sleeve shirt."
FPS = 0.05    # lower = faster/cheaper, higher = more detail

OUTPUT_DIR = "./trial_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading processor + model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": VIDEO_SOURCE, "fps": FPS},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    frames = load_video_frames(VIDEO_SOURCE, num_frames=8)

    print(type(frames))
    print(frames.shape)
    print(frames.dtype)
    
    inputs = processor(
        text=[text_prompt],
        videos=[frames],
        padding=True,
        return_tensors="pt",
    ).to(device)

    print(inputs.keys())

    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(k, v.shape)

    print(f"\nQuestion: {QUESTION}")
    print("Running generation...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n--- RESULTS ---")
    print(f"Qwen2-VL answer:\n{output_text}")

    if GROUND_TRUTH_ANSWER:
        print(f"\nDataset ground-truth answer:\n{GROUND_TRUTH_ANSWER}")

    with open(os.path.join(OUTPUT_DIR, "qwen-only-result.txt"), "w") as f:
        f.write(f"Video: {VIDEO_SOURCE}\n")
        f.write(f"Question: {QUESTION}\n")
        f.write(f"Qwen answer: {output_text}\n")
        f.write(f"Ground truth: {GROUND_TRUTH_ANSWER}\n")

    print(f"\nSaved result to {OUTPUT_DIR}/qwen-only-result.txt")


if __name__ == "__main__":
    main()