import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a higher-quality balanced subset manifest from landmark rows."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-classes", type=int, default=20)
    parser.add_argument("--min-train", type=int, default=5)
    parser.add_argument("--min-val", type=int, default=1)
    parser.add_argument("--min-test", type=int, default=1)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional explicit gloss list. If omitted, classes are chosen by available sample count.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    df = pd.read_csv(manifest_path)

    required = ["video_id", "gloss", "split", "landmark_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["split"] = df["split"].astype(str).str.lower()

    counts = (
        df.groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split in ["train", "val", "test"]:
        if split not in counts.columns:
            counts[split] = 0

    eligible = counts[
        (counts["train"] >= args.min_train)
        & (counts["val"] >= args.min_val)
        & (counts["test"] >= args.min_test)
    ].copy()

    if args.classes:
        requested = set(args.classes)
        eligible = eligible[eligible["gloss"].isin(requested)].copy()

    eligible["total"] = eligible[["train", "val", "test"]].sum(axis=1)
    eligible = eligible.sort_values(["total", "train"], ascending=False)

    selected = eligible["gloss"].head(args.max_classes).tolist()

    if len(selected) < 2:
        raise ValueError(
            "Fewer than 2 eligible classes found. Lower min-train/min-val/min-test or inspect the manifest."
        )

    out = df[df["gloss"].isin(selected)].copy()

    # Re-map label IDs to a compact range for this subset.
    label_to_id = {label: idx for idx, label in enumerate(sorted(selected))}
    out["label_id"] = out["gloss"].map(label_to_id).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels.csv")
    pd.DataFrame(
        [{"label_id": idx, "gloss": label} for label, idx in label_to_id.items()]
    ).sort_values("label_id").to_csv(labels_path, index=False)

    summary_path = output_path.with_name(output_path.stem + "_class_counts.csv")
    (
        out.groupby(["gloss", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .sort_values("gloss")
        .to_csv(summary_path, index=False)
    )

    print(f"Input rows: {len(df)}")
    print(f"Eligible classes: {len(eligible)}")
    print(f"Selected classes: {len(selected)}")
    print(f"Output rows: {len(out)}")
    print("Split counts:")
    print(out["split"].value_counts().to_string())
    print("\nSelected classes:")
    print(selected)
    print(f"\nSaved subset manifest: {output_path}")
    print(f"Saved labels: {labels_path}")
    print(f"Saved class counts: {summary_path}")


if __name__ == "__main__":
    main()
