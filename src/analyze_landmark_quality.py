import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def frame_valid_mask(arr):
    return np.any(arr != 0, axis=1)


def hand_valid_masks(arr):
    if arr.shape[1] != 126:
        raise ValueError(f"Expected 126 features, got {arr.shape[1]}")

    x = arr.reshape(arr.shape[0], 2, 21, 3)
    left_valid = np.any(x[:, 0] != 0, axis=(1, 2))
    right_valid = np.any(x[:, 1] != 0, axis=(1, 2))
    return left_valid, right_valid


def movement_score(arr):
    valid = frame_valid_mask(arr)
    if valid.sum() < 2:
        return 0.0

    valid_arr = arr[valid]
    deltas = np.abs(valid_arr[1:] - valid_arr[:-1])
    return float(np.mean(deltas))


def analyze_landmark(path):
    path = Path(path)
    result = {
        "landmark_exists": path.exists(),
        "landmark_shape": "",
        "valid_frame_ratio": 0.0,
        "left_hand_ratio": 0.0,
        "right_hand_ratio": 0.0,
        "two_hand_ratio": 0.0,
        "movement_score": 0.0,
        "quality_score": 0.0,
        "quality_good": False,
        "quality_reason": "",
    }

    if not path.exists():
        result["quality_reason"] = "missing_landmark"
        return result

    try:
        arr = np.load(path)
    except Exception as exc:
        result["quality_reason"] = f"load_error:{exc}"
        return result

    result["landmark_shape"] = "x".join(str(v) for v in arr.shape)

    if arr.ndim != 2:
        result["quality_reason"] = "bad_rank"
        return result

    if arr.shape[1] != 126:
        result["quality_reason"] = "bad_feature_dim"
        return result

    valid = frame_valid_mask(arr)
    left_valid, right_valid = hand_valid_masks(arr)
    either_valid = left_valid | right_valid
    two_valid = left_valid & right_valid

    valid_frame_ratio = float(either_valid.mean()) if len(either_valid) else 0.0
    left_ratio = float(left_valid.mean()) if len(left_valid) else 0.0
    right_ratio = float(right_valid.mean()) if len(right_valid) else 0.0
    two_hand_ratio = float(two_valid.mean()) if len(two_valid) else 0.0
    move = movement_score(arr)

    # This score favors videos with visible hands in most frames and some motion.
    # It does not require two hands because many signs are one-handed.
    quality_score = (
        0.65 * valid_frame_ratio
        + 0.20 * max(left_ratio, right_ratio)
        + 0.10 * two_hand_ratio
        + 0.05 * min(move / 0.02, 1.0)
    )

    good = valid_frame_ratio >= 0.60 and max(left_ratio, right_ratio) >= 0.50 and move > 0.0005

    result.update(
        {
            "valid_frame_ratio": valid_frame_ratio,
            "left_hand_ratio": left_ratio,
            "right_hand_ratio": right_ratio,
            "two_hand_ratio": two_hand_ratio,
            "movement_score": move,
            "quality_score": float(quality_score),
            "quality_good": bool(good),
            "quality_reason": "ok" if good else "low_quality",
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Analyze MediaPipe landmark quality for WLASL samples.")
    parser.add_argument("--manifest", required=True, help="Manifest with landmark_path column.")
    parser.add_argument("--output", required=True, help="Output sample-level quality CSV.")
    parser.add_argument(
        "--class-summary-output",
        default=None,
        help="Optional output class summary CSV. Defaults to output stem + _class_summary.csv.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)
    class_summary_path = (
        Path(args.class_summary_output)
        if args.class_summary_output
        else output_path.with_name(output_path.stem + "_class_summary.csv")
    )

    df = pd.read_csv(manifest_path)
    if "landmark_path" not in df.columns:
        raise ValueError("Manifest must contain landmark_path column.")

    rows = []
    for idx, row in df.iterrows():
        info = analyze_landmark(row["landmark_path"])
        rows.append(info)
        if (idx + 1) % 100 == 0:
            print(f"Analyzed {idx + 1}/{len(df)}")

    quality_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(output_path, index=False)

    required = ["gloss", "split", "quality_good", "quality_score"]
    missing = [c for c in required if c not in quality_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for summary: {missing}")

    quality_df["split"] = quality_df["split"].astype(str).str.lower()
    good_df = quality_df[quality_df["quality_good"] == True].copy()

    split_counts = (
        good_df.groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split in ["train", "val", "test"]:
        if split not in split_counts.columns:
            split_counts[split] = 0

    quality_stats = (
        quality_df.groupby("gloss")
        .agg(
            total_rows=("video_id", "size"),
            landmark_exists=("landmark_exists", "sum"),
            quality_good_count=("quality_good", "sum"),
            mean_quality_score=("quality_score", "mean"),
            mean_valid_frame_ratio=("valid_frame_ratio", "mean"),
            mean_movement_score=("movement_score", "mean"),
        )
        .reset_index()
    )

    class_summary = quality_stats.merge(split_counts, on="gloss", how="left").fillna(0)
    class_summary["usable_total"] = class_summary[["train", "val", "test"]].sum(axis=1)
    class_summary = class_summary.sort_values(
        ["usable_total", "train", "mean_quality_score"],
        ascending=False,
    )

    class_summary_path.parent.mkdir(parents=True, exist_ok=True)
    class_summary.to_csv(class_summary_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Landmarks found: {int(quality_df['landmark_exists'].sum())}")
    print(f"Quality-good samples: {int(quality_df['quality_good'].sum())}")
    print(f"Classes: {quality_df['gloss'].nunique()}")
    print(f"Saved sample quality: {output_path}")
    print(f"Saved class summary: {class_summary_path}")
    print("\nTop classes by usable samples:")
    print(class_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()

