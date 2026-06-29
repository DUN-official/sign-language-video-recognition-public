import argparse
from pathlib import Path

import pandas as pd
import yaml


def normalize_video_id(video_id) -> str:
    """
    WLASL video IDs are commonly stored as 5-digit filenames, for example:
        618   -> 00618.npy
        623   -> 00623.npy
        69241 -> 69241.npy
    """
    return str(video_id).strip().zfill(5)


def find_landmark_path(landmarks_dir: Path, video_id: str) -> Path | None:
    """
    Search for the landmark .npy file matching a video_id.

    Supports both:
        data/landmarks/wlasl100/00618.npy

    and possible nested structures:
        data/landmarks/wlasl100/class_name/00618.npy
    """
    direct_path = landmarks_dir / f"{video_id}.npy"
    if direct_path.exists():
        return direct_path

    matches = list(landmarks_dir.glob(f"**/{video_id}.npy"))
    if matches:
        return matches[0]

    return None


def cfg_path(cfg: dict, key: str, fallback: str | None = None) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    value = dataset_cfg.get(key, fallback)
    if value is None:
        raise KeyError(f"Missing dataset.{key} in config.")
    return Path(value)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build a training-ready WLASL manifest containing only rows with "
            "extracted landmark .npy files."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset config containing stage-specific manifest paths.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional input manifest override. If omitted, uses "
            "dataset.downloaded_manifest_csv from the config."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output manifest override. If omitted, uses "
            "dataset.landmark_manifest_csv from the config."
        ),
    )
    parser.add_argument(
        "--missing-output",
        type=Path,
        default=None,
        help=(
            "Optional missing-landmarks output override. If omitted, uses "
            "dataset.missing_landmark_manifest_csv from the config."
        ),
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    landmarks_dir = cfg_path(cfg, "landmarks_dir")
    input_manifest = args.manifest or cfg_path(
        cfg,
        "downloaded_manifest_csv",
        fallback="data/processed/wlasl100_manifest_downloaded.csv",
    )
    output_manifest = args.output or cfg_path(
        cfg,
        "landmark_manifest_csv",
        fallback="data/processed/wlasl100_manifest_landmarks.csv",
    )
    missing_output = args.missing_output or cfg_path(
        cfg,
        "missing_landmark_manifest_csv",
        fallback="data/processed/wlasl100_manifest_missing_landmarks.csv",
    )

    if not input_manifest.exists():
        raise FileNotFoundError(
            f"Input manifest not found: {input_manifest}\n"
            "Run verify_wlasl_downloads.py first to create the downloaded manifest."
        )

    if not landmarks_dir.exists():
        raise FileNotFoundError(
            f"Landmarks directory not found: {landmarks_dir}\n"
            "Run extract_landmarks.py before building the landmark manifest."
        )

    df = pd.read_csv(input_manifest)

    if "video_id" not in df.columns:
        raise ValueError(
            "Manifest must contain a 'video_id' column. "
            f"Available columns: {list(df.columns)}"
        )

    landmark_paths = []
    has_landmarks = []

    for raw_video_id in df["video_id"]:
        video_id = normalize_video_id(raw_video_id)
        landmark_path = find_landmark_path(landmarks_dir, video_id)

        if landmark_path is None:
            landmark_paths.append("")
            has_landmarks.append(False)
        else:
            landmark_paths.append(str(landmark_path))
            has_landmarks.append(True)

    output_df = df.copy()
    output_df["landmark_path"] = landmark_paths
    output_df["has_landmarks"] = has_landmarks

    # If a downloaded-manifest status column exists, keep only rows that both
    # have landmarks and were found during download verification.
    if "download_status" in output_df.columns:
        usable_df = output_df[
            (output_df["has_landmarks"]) & (output_df["download_status"].astype(str).str.lower() == "found")
        ].copy()
    else:
        usable_df = output_df[output_df["has_landmarks"]].copy()

    missing_df = output_df[~output_df.index.isin(usable_df.index)].copy()

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    missing_output.parent.mkdir(parents=True, exist_ok=True)

    usable_df.to_csv(output_manifest, index=False)
    missing_df.to_csv(missing_output, index=False)

    print("Landmark manifest build complete.")
    print(f"Input manifest:                 {input_manifest}")
    print(f"Input rows:                     {len(df)}")
    print(f"Rows with usable landmarks:     {len(usable_df)}")
    print(f"Rows missing/unusable:          {len(missing_df)}")
    print(f"Training-ready manifest saved:  {output_manifest}")
    print(f"Missing-landmarks CSV saved:    {missing_output}")


if __name__ == "__main__":
    main()
