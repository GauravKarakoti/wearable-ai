"""
score_predictions.py -- compare predictions.jsonl against ground-truth
MCQ answers and report accuracy, overall and by category.
"""

import json
from collections import defaultdict

PREDICTIONS_PATH = "./trial_output/predictions_3.jsonl"
GROUND_TRUTH_PATH = "../egolongqa/wearable_ai_2026_egolongqa_val_700.jsonl"


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    predictions = load_jsonl(PREDICTIONS_PATH)
    ground_truth = load_jsonl(GROUND_TRUTH_PATH)

    pred_by_path = {p["video_path"]: p["mcq_answer"] for p in predictions}
    gt_by_path = {g["video_path"]: g for g in ground_truth}

    print(f"Predictions file: {len(predictions)} rows")
    print(f"Ground truth file: {len(ground_truth)} rows")

    missing_predictions = [vp for vp in gt_by_path if vp not in pred_by_path]
    extra_predictions = [vp for vp in pred_by_path if vp not in gt_by_path]

    if missing_predictions:
        print(f"\nWARNING: {len(missing_predictions)} ground-truth videos have NO prediction:")
        for vp in missing_predictions[:10]:
            print(f"  {vp}")
        if len(missing_predictions) > 10:
            print(f"  ... and {len(missing_predictions) - 10} more")

    if extra_predictions:
        print(f"\nWARNING: {len(extra_predictions)} predictions have no matching ground-truth row "
              f"(video_path not found in {GROUND_TRUTH_PATH}):")
        for vp in extra_predictions[:10]:
            print(f"  {vp}")
        if len(extra_predictions) > 10:
            print(f"  ... and {len(extra_predictions) - 10} more")

    scored_paths = [vp for vp in gt_by_path if vp in pred_by_path]
    if not scored_paths:
        print("\nNo overlapping video_paths between predictions and ground truth -- nothing to score.")
        return

    correct = 0
    per_category_correct = defaultdict(int)
    per_category_total = defaultdict(int)
    mismatches = []

    for vp in scored_paths:
        gt_row = gt_by_path[vp]
        gt_answer = gt_row["mcq_answer"].strip().upper()
        pred_answer = pred_by_path[vp].strip().upper()
        category = gt_row.get("category", "UNKNOWN")

        is_correct = gt_answer == pred_answer
        per_category_total[category] += 1
        if is_correct:
            correct += 1
            per_category_correct[category] += 1
        else:
            mismatches.append({
                "video_path": vp,
                "question": gt_row.get("question", ""),
                "predicted": pred_answer,
                "correct_answer": gt_answer,
                "category": category,
            })

    total = len(scored_paths)
    accuracy = correct / total

    print(f"\n--- OVERALL ACCURACY ---")
    print(f"Scored: {total} / {len(gt_by_path)} ground-truth videos")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"(Chance baseline for 4-way MCQ: 25%)")

    print(f"\n--- ACCURACY BY CATEGORY ---")
    for category in sorted(per_category_total.keys()):
        cat_total = per_category_total[category]
        cat_correct = per_category_correct[category]
        cat_acc = cat_correct / cat_total
        print(f"  {category:30s}  {cat_correct:4d}/{cat_total:<4d}  {cat_acc*100:6.2f}%")

    if mismatches:
        with open("./trial_output/incorrect_predictions_2B.jsonl", "w") as f:
            for m in mismatches:
                f.write(json.dumps(m) + "\n")
        print(f"\nSaved {len(mismatches)} incorrect predictions to ./incorrect_predictions_2B.jsonl for inspection.")


if __name__ == "__main__":
    main()