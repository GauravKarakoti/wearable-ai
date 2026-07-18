"""
download_remaining.py -- bulk download ONLY, no encoding, no training.

Reads your full egolongqa manifest (jsonl), downloads every video not
already present on the hard drive, and stops. Run this in chunks of
BATCH_SIZE at a time (or edit MAX_TO_DOWNLOAD to grab everything in one go
if your connection can handle it unattended).
"""

import os
import json
from huggingface_hub import hf_hub_download

HF_REPO_ID = "facebook/wearable-ai"
HF_VIDEO_SUBDIR = "egolongqa/val"

MANIFEST_PATH = "download.jsonl"
HARD_DRIVE_VIDEO_DIR = "D:/wearable_ai_videos_5"

BATCH_SIZE = 100


def load_manifest():
    rows = []
    with open(MANIFEST_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def already_downloaded(video_id):
    return os.path.exists(os.path.join(HARD_DRIVE_VIDEO_DIR, f"{video_id}.mp4"))


def main():
    os.makedirs(HARD_DRIVE_VIDEO_DIR, exist_ok=True)
    manifest = load_manifest()
    print(f"Manifest has {len(manifest)} total videos.")

    pending = [row for row in manifest if not already_downloaded(extract_video_id(row))]
    print(f"{len(pending)} not yet downloaded. Downloading up to {BATCH_SIZE} this run.")

    to_download = pending[:BATCH_SIZE]
    succeeded, failed = 0, []

    for i, row in enumerate(to_download, 1):
        video_id = extract_video_id(row)
        file_name = f"{video_id}.mp4"
        print(f"[{i}/{len(to_download)}] {video_id} ...", end=" ")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{HF_VIDEO_SUBDIR}/{file_name}",
                repo_type="dataset",
                local_dir=HARD_DRIVE_VIDEO_DIR,
            )
            print("OK")
            succeeded += 1
        except Exception as e:
            print(f"FAILED ({e})")
            failed.append(video_id)

    print(f"\nDone. {succeeded} downloaded, {len(failed)} failed.")
    if failed:
        print(f"Failed ids (retry these): {failed}")

    remaining = len(pending) - succeeded
    print(f"{remaining} still pending after this run. Re-run the script to continue.")


def extract_video_id(row):
    return os.path.splitext(row["video_path"])[0]


if __name__ == "__main__":
    main()