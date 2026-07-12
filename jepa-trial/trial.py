"""
V-JEPA 2 - one-video trial run.

What this does:
  1. Loads the pretrained V-JEPA2 encoder + predictor from Hugging Face.
  2. Loads ONE video (local file or URL).
  3. Runs a forward pass inference.
  4. Prints the output shapes so we can see what V-JEPA2 actually gives us.
  5. Saves the raw embeddings + model config to disk so we can inspect them later.
"""

import os
import numpy as np
import torch
import cv2

from transformers import AutoModel, AutoVideoProcessor

MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"    # swap for a larger one later if needed
VIDEO_SOURCE = "sample.mp4"

OUTPUT_DIR = "./trial_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_video_frames(video_path, num_frames=64):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    indices = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    return np.stack(frames)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\nLoading processor + model: {MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    num_frames = model.config.frames_per_clip
    print(f"Model expects {num_frames} frames per clip")

    frames = load_video_frames(VIDEO_SOURCE, num_frames=num_frames)

    inputs = processor(frames, return_tensors="pt").to(device)

    print("\nRunning forward pass (inference only, no gradients)...")
    with torch.no_grad():
        outputs = model(**inputs)

    encoder_out = outputs.last_hidden_state
    predictor_out = outputs.predictor_output.last_hidden_state

    print("\n--- RESULTS ---")
    print(f"Encoder output shape:   {tuple(encoder_out.shape)}   (batch, num_patches, hidden_size)")
    print(f"Predictor output shape: {tuple(predictor_out.shape)}  (batch, num_patches, pred_hidden_size)")
    print(f"Encoder embedding stats: mean={encoder_out.mean().item():.4f}, std={encoder_out.std().item():.4f}")

    torch.save(
        {
            "encoder_output": encoder_out.cpu(),
            "predictor_output": predictor_out.cpu(),
            "model_id": MODEL_ID,
            "video_source": VIDEO_SOURCE,
            "config": model.config.to_dict(),
        },
        os.path.join(OUTPUT_DIR, "trial_result.pt"),
    )
    print(f"\nSaved embeddings + config to {OUTPUT_DIR}/trial_result.pt")
    print("\nModel weights themselves are cached by huggingface_hub at:")
    print("  ~/.cache/huggingface/hub/  (look for a folder starting with 'models--facebook--vjepa2')")


if __name__ == "__main__":
    main()