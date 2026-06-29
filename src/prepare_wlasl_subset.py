import argparse
import csv
import json
from pathlib import Path


def build_subset(metadata_path: Path, subset_size: int, output_root: Path, manifest_path: Path):
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    subset_entries = data[:subset_size]

    output_root.mkdir(parents=True, exist_ok=True)
    video_dir = output_root / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    subset_json_path = output_root / f"WLASL{subset_size}.json"
    subset_json_path.write_text(json.dumps(subset_entries, indent=2), encoding="utf-8")

    label_to_id = {entry["gloss"]: idx for idx, entry in enumerate(subset_entries)}
    rows = []

    for entry in subset_entries:
        gloss = entry["gloss"]
        label_id = label_to_id[gloss]

        for inst in entry["instances"]:
            video_id = str(inst["video_id"])
            rows.append(
                {
                    "video_id": video_id,
                    "gloss": gloss,
                    "label": gloss,
                    "label_id": label_id,
                    "split": inst.get("split", ""),
                    "url": inst.get("url", ""),
                    "source": inst.get("source", ""),
                    "signer_id": inst.get("signer_id", ""),
                    "frame_start": inst.get("frame_start", 1),
                    "frame_end": inst.get("frame_end", -1),
                    "fps": inst.get("fps", 25),
                    "expected_video_stem": str(video_dir / video_id),
                    "landmark_path": str(Path("data/landmarks") / f"wlasl{subset_size}" / f"{video_id}.npy"),
                }
            )

    fieldnames = [
        "video_id",
        "gloss",
        "label",
        "label_id",
        "split",
        "url",
        "source",
        "signer_id",
        "frame_start",
        "frame_end",
        "fps",
        "expected_video_stem",
        "landmark_path",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {}
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    print(f"Subset glosses: {len(subset_entries)}")
    print(f"Subset instances: {len(rows)}")
    print(f"Split counts: {split_counts}")
    print(f"Subset JSON: {subset_json_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Video directory: {video_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create WLASL subset JSON and manifest.")
    parser.add_argument("--metadata", type=Path, required=True, help="Path to WLASL_v0.3.json.")
    parser.add_argument("--subset-size", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=Path("data/raw/WLASL100"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/wlasl100_manifest.csv"))
    args = parser.parse_args()

    build_subset(args.metadata, args.subset_size, args.output_root, args.manifest)


if __name__ == "__main__":
    main()
