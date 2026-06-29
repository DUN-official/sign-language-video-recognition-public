import argparse
import csv
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Check how many raw videos have matching extracted landmark .npy files."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Dataset config used by extract_landmarks.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output override.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_cfg = cfg.get("dataset", {})
    logging_cfg = cfg.get("logging", {})

    raw_video_dir = Path(dataset_cfg["raw_video_dir"])
    landmarks_dir = Path(dataset_cfg["landmarks_dir"])

    output_path = args.output or Path(logging_cfg.get("landmark_log_dir", "logs/landmark_extraction")) / "landmark_coverage.csv"

    video_paths = sorted(raw_video_dir.glob("**/*.mp4"))

    rows = []
    found_count = 0
    missing_count = 0

    for video_path in video_paths:
        relative = video_path.relative_to(raw_video_dir).with_suffix(".npy")
        landmark_path = landmarks_dir / relative
        has_landmarks = landmark_path.exists()

        if has_landmarks:
            found_count += 1
        else:
            missing_count += 1

        rows.append(
            {
                "video_path": str(video_path),
                "landmark_path": str(landmark_path),
                "has_landmarks": str(has_landmarks),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "landmark_path", "has_landmarks"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Raw videos:               {len(video_paths)}")
    print(f"Videos with landmarks:    {found_count}")
    print(f"Videos missing landmarks: {missing_count}")
    print(f"Coverage report:          {output_path}")


if __name__ == "__main__":
    main()
