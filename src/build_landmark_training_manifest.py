import argparse
from pathlib import Path

import pandas as pd


def find_landmark_files(landmarks_dir: Path):
    files = {}
    for path in landmarks_dir.rglob("*.npy"):
        stem = path.stem
        files[stem] = path
        # Also index by numeric/video-id-like tokens inside filename.
        parts = stem.replace("-", "_").split("_")
        for part in parts:
            if part.isdigit():
                files.setdefault(part, path)
    return files


def pick_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Create a training manifest that keeps only rows with extracted landmark .npy files."
    )
    parser.add_argument("--manifest", required=True, help="Downloaded WLASL manifest CSV.")
    parser.add_argument("--landmarks-dir", required=True, help="Folder containing extracted .npy landmark files.")
    parser.add_argument("--output", required=True, help="Output CSV training manifest.")
    parser.add_argument("--min-classes", type=int, default=2)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    landmarks_dir = Path(args.landmarks_dir)
    output_path = Path(args.output)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not landmarks_dir.exists():
        raise FileNotFoundError(f"Landmarks directory not found: {landmarks_dir}")

    df = pd.read_csv(manifest_path)

    video_id_col = pick_col(df, ["video_id", "id", "expected_video_stem"])
    label_col = pick_col(df, ["gloss", "label", "class_name"])
    label_id_col = pick_col(df, ["label_id", "class_id"])
    split_col = pick_col(df, ["split", "subset"])

    if video_id_col is None:
        raise ValueError(f"Could not find video ID column. Columns: {df.columns.tolist()}")
    if label_col is None:
        raise ValueError(f"Could not find label/gloss column. Columns: {df.columns.tolist()}")
    if split_col is None:
        raise ValueError(f"Could not find split column. Columns: {df.columns.tolist()}")

    landmark_lookup = find_landmark_files(landmarks_dir)

    rows = []
    for _, row in df.iterrows():
        video_id = str(row[video_id_col]).strip()
        landmark_path = None

        # Try exact video_id first.
        if video_id in landmark_lookup:
            landmark_path = landmark_lookup[video_id]
        else:
            # Try any landmark filename containing video_id.
            matches = list(landmarks_dir.rglob(f"*{video_id}*.npy"))
            if matches:
                landmark_path = matches[0]

        if landmark_path is None:
            continue

        label = str(row[label_col])
        split = str(row[split_col]).lower()

        record = {
            "video_id": video_id,
            "gloss": label,
            "split": split,
            "landmark_path": str(landmark_path).replace("\\", "/"),
        }

        if label_id_col is not None:
            record["label_id"] = int(row[label_id_col])

        rows.append(record)

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "No rows matched landmark files. Check --landmarks-dir and landmark filenames."
        )

    # Rebuild label IDs from available classes so labels are contiguous.
    classes = sorted(out["gloss"].unique())
    if len(classes) < args.min_classes:
        raise ValueError(f"Only found {len(classes)} classes with landmarks: {classes}")

    label_to_id = {label: idx for idx, label in enumerate(classes)}
    out["label_id"] = out["gloss"].map(label_to_id).astype(int)

    # Keep only standard splits.
    out = out[out["split"].isin(["train", "val", "test"])].copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    labels_path = output_path.with_name(output_path.stem + "_labels.csv")
    pd.DataFrame(
        [{"label_id": idx, "gloss": label} for label, idx in label_to_id.items()]
    ).sort_values("label_id").to_csv(labels_path, index=False)

    print(f"Input manifest rows: {len(df)}")
    print(f"Landmark files found: {len(landmark_lookup)}")
    print(f"Training rows with landmarks: {len(out)}")
    print(f"Classes with landmarks: {len(classes)}")
    print("Split counts:")
    print(out["split"].value_counts().to_string())
    print(f"Saved training manifest: {output_path}")
    print(f"Saved labels: {labels_path}")


if __name__ == "__main__":
    main()
