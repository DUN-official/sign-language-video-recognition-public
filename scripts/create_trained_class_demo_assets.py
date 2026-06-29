import argparse
import shutil
from pathlib import Path

import pandas as pd


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".webm"]


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip())


def resolve_path(path_text, repo_root):
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return repo_root / path


def find_video_path(row, repo_root):
    if "raw_video_path" in row and pd.notna(row["raw_video_path"]):
        candidate = resolve_path(row["raw_video_path"], repo_root)
        if candidate.exists():
            return candidate

    video_id = str(row["video_id"])
    candidates = []

    if "video_dir" in row and pd.notna(row["video_dir"]):
        base = resolve_path(row["video_dir"], repo_root)
        candidates.extend(base / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS)

    for base in [
        repo_root / "data" / "raw" / "WLASL500" / "videos",
        repo_root / "data" / "raw" / "WLASL100" / "videos",
    ]:
        candidates.extend(base / f"{video_id}{ext}" for ext in VIDEO_EXTENSIONS)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


def select_rows(df, clips_per_class, prefer_splits):
    selected = []

    df = df.copy()
    df["split"] = df["split"].astype(str).str.lower()

    for gloss in sorted(df["gloss"].astype(str).unique()):
        class_df = df[df["gloss"].astype(str) == gloss].copy()
        class_rows = []

        for split in prefer_splits:
            split_df = class_df[class_df["split"] == split]
            for _, row in split_df.iterrows():
                class_rows.append(row)
                if len(class_rows) >= clips_per_class:
                    break
            if len(class_rows) >= clips_per_class:
                break

        if len(class_rows) < clips_per_class:
            used_ids = {str(row["video_id"]) for row in class_rows}
            for _, row in class_df.iterrows():
                if str(row["video_id"]) in used_ids:
                    continue
                class_rows.append(row)
                if len(class_rows) >= clips_per_class:
                    break

        selected.extend(class_rows)

    return pd.DataFrame(selected)


def main():
    parser = argparse.ArgumentParser(
        description="Copy trained-class videos into a curated demo asset folder."
    )
    parser.add_argument("--manifest", required=True, help="Training manifest for selected classes.")
    parser.add_argument("--output-dir", required=True, help="Demo clip output folder.")
    parser.add_argument("--clips-per-class", type=int, default=2)
    parser.add_argument(
        "--prefer-splits",
        nargs="+",
        default=["test", "val", "train"],
        help="Split priority for demo clip selection.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repo root used to resolve relative raw_video_path values.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "manifest-only"],
        default="copy",
        help="copy copies videos into assets; manifest-only writes paths without copying.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = resolve_path(args.manifest, repo_root)
    output_dir = resolve_path(args.output_dir, repo_root)

    df = pd.read_csv(manifest_path)
    required = {"video_id", "gloss", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    selected = select_rows(df, args.clips_per_class, [s.lower() for s in args.prefer_splits])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing_videos = []

    for _, row in selected.iterrows():
        source = find_video_path(row, repo_root)
        if source is None:
            missing_videos.append(
                {
                    "video_id": row["video_id"],
                    "gloss": row["gloss"],
                    "reason": "source_video_not_found",
                }
            )
            continue

        gloss = safe_name(row["gloss"])
        video_id = str(row["video_id"])
        class_dir = output_dir / gloss
        class_dir.mkdir(parents=True, exist_ok=True)

        target = class_dir / f"{video_id}{source.suffix.lower()}"

        if args.copy_mode == "copy" and not target.exists():
            shutil.copy2(source, target)
        elif args.copy_mode == "manifest-only":
            target = source

        try:
            demo_path = target.relative_to(repo_root)
        except ValueError:
            demo_path = target

        rows.append(
            {
                "video_id": video_id,
                "gloss": row["gloss"],
                "label_id": row.get("label_id", ""),
                "split": row.get("split", ""),
                "demo_video_path": str(demo_path).replace("\\", "/"),
                "source_video_path": str(source).replace("\\", "/"),
                "source_manifest": str(manifest_path).replace("\\", "/"),
            }
        )

    demo_manifest = pd.DataFrame(rows)
    demo_manifest_path = output_dir / "demo_manifest.csv"
    demo_manifest.to_csv(demo_manifest_path, index=False)

    class_summary_path = output_dir / "demo_class_summary.csv"
    if not demo_manifest.empty:
        (
            demo_manifest.groupby(["gloss", "split"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
            .sort_values("gloss")
            .to_csv(class_summary_path, index=False)
        )
    else:
        pd.DataFrame(columns=["gloss"]).to_csv(class_summary_path, index=False)

    missing_path = output_dir / "missing_demo_videos.csv"
    pd.DataFrame(missing_videos).to_csv(missing_path, index=False)

    print(f"Input manifest rows: {len(df)}")
    print(f"Selected rows: {len(selected)}")
    print(f"Demo videos available: {len(demo_manifest)}")
    print(f"Missing source videos: {len(missing_videos)}")
    print(f"Saved demo manifest: {demo_manifest_path}")
    print(f"Saved class summary: {class_summary_path}")
    print(f"Saved missing list: {missing_path}")


if __name__ == "__main__":
    main()

