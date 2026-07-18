"""
encode.py -- JEPA encoding ONLY, no training. Meant to run alongside
train_bridge.py so you can be encoding the NEXT batch while training chews
through whatever's already in the library.
"""

import os
import json
import torch

from transformers import AutoModel, AutoVideoProcessor
from trial import load_video_frames

JEPA_MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
NUM_VIDEO_TOKENS = 16

MANIFEST_PATH = "encode.jsonl"
HARD_DRIVE_VIDEO_DIR = "D:/wearable_ai_videos_4/egolongqa/val"

DEVICE_OVERRIDE = None

ENCODE_LIMIT = 100

OUTPUT_DIR = "./trial_output"
EMBEDDING_LIBRARY_DIR = os.path.join(OUTPUT_DIR, "embedding_library_2")
os.makedirs(EMBEDDING_LIBRARY_DIR, exist_ok=True)


def library_path(video_id):
    return os.path.join(EMBEDDING_LIBRARY_DIR, f"{video_id}.pt")


def load_manifest():
    rows = []
    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                row["video_id"] = os.path.splitext(row["video_path"])[0]
                rows.append(row)
    return rows


def local_video_path(video_id):
    path = os.path.join(HARD_DRIVE_VIDEO_DIR, f"{video_id}.mp4")
    return path if os.path.exists(path) else None


def save_embedding_atomic(data, video_id):
    final_path = library_path(video_id)
    tmp_path = final_path + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, final_path)


def main():
    device = DEVICE_OVERRIDE or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}"
          f"{' (forced override)' if DEVICE_OVERRIDE else ''}")

    manifest = load_manifest()
    candidates = [
        row for row in manifest
        if not os.path.exists(library_path(row["video_id"])) and local_video_path(row["video_id"])
    ]
    to_encode = candidates[:ENCODE_LIMIT] if ENCODE_LIMIT else candidates
    print(f"{len(candidates)} videos on disk and not yet embedded. Encoding {len(to_encode)} this run.")
    if not to_encode:
        print("Nothing to do.")
        return

    print(f"\nLoading V-JEPA2: {JEPA_MODEL_ID}")
    processor = AutoVideoProcessor.from_pretrained(JEPA_MODEL_ID)
    model = AutoModel.from_pretrained(JEPA_MODEL_ID, device_map={"": device}, attn_implementation="sdpa")
    model.eval()
    num_frames = model.config.frames_per_clip

    encoded, skipped = 0, 0
    for i, row in enumerate(to_encode, 1):
        video_id = row["video_id"]

        if os.path.exists(library_path(video_id)):
            print(f"  [{i}/{len(to_encode)}] {video_id} already embedded elsewhere -- skipping.")
            skipped += 1
            continue

        video_path = local_video_path(video_id)
        print(f"  [{i}/{len(to_encode)}] Encoding {video_id} ...")
        try:
            frames = load_video_frames(video_path, num_frames=num_frames)
            inputs = processor(frames, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                out.transpose(1, 2), NUM_VIDEO_TOKENS
            ).transpose(1, 2).squeeze(0).cpu()

            save_embedding_atomic(
                {"embedding": pooled, "question": row["question"], "answer": row["answer"]},
                video_id,
            )
            encoded += 1
        except Exception as e:
            print(f"    FAILED to encode {video_id}: {e}")

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nDone. Encoded {encoded}, skipped {skipped} (already done by another process), "
          f"{len(to_encode) - encoded - skipped} failed.")
    remaining = len(candidates) - encoded - skipped
    print(f"{remaining} still pending. Re-run to continue.")


if __name__ == "__main__":
    main()