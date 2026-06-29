import argparse
import csv
from pathlib import Path

import yaml


VIDEO_EXTENSIONS = [".mp4", ".mkv", ".webm", ".swf"]


def normalize_video_ids(video_id: str) -> list[str]:
    raw = str(video_id).strip()
    padded = raw.zfill(5)
    if padded == raw:
        return [raw]
    return [raw, padded]


def find_video(video_dir: Path, video_id: str):
    for candidate_id in normalize_video_ids(video_id):
        for ext in VIDEO_EXTENSIONS:
            candidate = video_dir / f"{candidate_id}{ext}"
            if candidate.exists():
                return candidate
    return None


def verify(manifest_path: Path, video_dir: Path, output_manifest: Path):
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    with manifest_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    updated = []
    found = 0
    missing = 0

    for row in rows:
        video_id = row["video_id"]
        video_path = find_video(video_dir, video_id)

        row["resolved_video_path"] = str(video_path) if video_path else ""
        row["download_status"] = "found" if video_path else "missing"

        updated.append(row)

        if video_path:
            found += 1
        else:
            missing += 1

    fieldnames = list(updated[0].keys()) if updated else []
    with output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)

    print(f"Manifest rows: {len(rows)}")
    print(f"Videos found: {found}")
    print(f"Videos missing: {missing}")
    print(f"Updated manifest: {output_manifest}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Verify downloaded WLASL videos and create the downloaded-manifest CSV. "
            "If explicit paths are omitted, paths are read from configs/dataset.yaml."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--output-manifest", type=Path, default=None)
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}

    dataset_cfg = cfg.get("dataset", {})

    manifest_path = args.manifest or Path(
        dataset_cfg.get("original_manifest_csv", "data/processed/wlasl100_manifest.csv")
    )
    video_dir = args.video_dir or Path(
        dataset_cfg.get("raw_video_dir", "data/raw/WLASL100/videos")
    )
    output_manifest = args.output_manifest or Path(
        dataset_cfg.get("downloaded_manifest_csv", dataset_cfg.get("processed_manifest", "data/processed/wlasl100_manifest_downloaded.csv"))
    )

    verify(manifest_path, video_dir, output_manifest)


if __name__ == "__main__":
    main()
