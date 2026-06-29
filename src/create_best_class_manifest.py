import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create best-class manifest from sample-level landmark quality analysis."
    )
    parser.add_argument("--quality-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--min-train", type=int, default=8)
    parser.add_argument("--min-val", type=int, default=2)
    parser.add_argument("--min-test", type=int, default=2)
    parser.add_argument("--min-quality-score", type=float, default=0.60)
    parser.add_argument(
        "--keep-low-quality",
        action="store_true",
        help="Use all landmark rows instead of only quality_good rows.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional explicit gloss list to intersect with eligible classes.",
    )
    args = parser.parse_args()

    quality_path = Path(args.quality_csv)
    output_path = Path(args.output)

    df = pd.read_csv(quality_path)

    required = ["gloss", "label_id", "split", "landmark_path", "quality_good", "quality_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["split"] = df["split"].astype(str).str.lower()

    usable = df.copy() if args.keep_low_quality else df[df["quality_good"] == True].copy()
    usable = usable[usable["quality_score"] >= args.min_quality_score].copy()

    counts = (
        usable.groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split in ["train", "val", "test"]:
        if split not in counts.columns:
            counts[split] = 0

    quality_stats = (
        usable.groupby("gloss")
        .agg(
            usable_total=("video_id", "size"),
            mean_quality_score=("quality_score", "mean"),
            mean_valid_frame_ratio=("valid_frame_ratio", "mean"),
            mean_movement_score=("movement_score", "mean"),
        )
        .reset_index()
    )

    eligible = counts.merge(quality_stats, on="gloss", how="inner")
    eligible = eligible[
        (eligible["train"] >= args.min_train)
        & (eligible["val"] >= args.min_val)
        & (eligible["test"] >= args.min_test)
    ].copy()

    if args.classes:
        requested = set(args.classes)
        eligible = eligible[eligible["gloss"].isin(requested)].copy()

    eligible = eligible.sort_values(
        ["usable_total", "train", "mean_quality_score"],
        ascending=False,
    )

    selected = eligible["gloss"].head(args.num_classes).tolist()

    if len(selected) < 2:
        raise ValueError(
            "Fewer than 2 eligible classes found. Lower min-train/min-val/min-test/min-quality-score."
        )

    out = usable[usable["gloss"].isin(selected)].copy()

    label_to_id = {gloss: idx for idx, gloss in enumerate(sorted(selected))}
    out["label_id"] = out["gloss"].map(label_to_id).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels.csv")
    pd.DataFrame(
        [{"label_id": idx, "gloss": gloss} for gloss, idx in label_to_id.items()]
    ).sort_values("label_id").to_csv(labels_path, index=False)

    class_counts_path = output_path.with_name(output_path.stem + "_class_counts.csv")
    (
        out.groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("gloss")
        .to_csv(class_counts_path, index=False)
    )

    selected_summary_path = output_path.with_name(output_path.stem + "_selected_summary.csv")
    eligible[eligible["gloss"].isin(selected)].to_csv(selected_summary_path, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Usable rows after quality filtering: {len(usable)}")
    print(f"Eligible classes: {len(eligible)}")
    print(f"Selected classes: {len(selected)}")
    print(f"Output rows: {len(out)}")
    print("\nSplit counts:")
    print(out["split"].value_counts().to_string())
    print("\nSelected classes:")
    print(selected)
    print(f"\nSaved manifest: {output_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved class counts: {class_counts_path}")
    print(f"Saved selected summary: {selected_summary_path}")


if __name__ == "__main__":
    main()

