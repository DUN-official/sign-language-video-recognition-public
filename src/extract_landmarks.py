import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from preprocess_video import sample_video_frames
from mediapipe_tasks_landmarks import HandLandmarkExtractor


def write_report(report_rows, report_path: Path):
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "status",
        "video_path",
        "output_path",
        "shape",
        "error",
    ]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def write_failed_list(report_rows, failed_path: Path):
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    failed_video_paths = [
        row["video_path"]
        for row in report_rows
        if row["status"] == "failed"
    ]

    failed_path.write_text(
        "\n".join(failed_video_paths) + ("\n" if failed_video_paths else ""),
        encoding="utf-8",
    )

    return failed_video_paths


def main():
    parser = argparse.ArgumentParser(
        description="Extract hand landmarks from video clips using the newer MediaPipe Tasks API."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional override for the MediaPipe Hand Landmarker .task model file.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional FPS override used to generate timestamps for sampled frames.",
    )
    parser.add_argument(
        "--input-color",
        choices=["rgb", "bgr"],
        default=None,
        help="Optional frame color override. Use bgr only if preprocessing returns raw OpenCV BGR frames.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recreate .npy files even if they already exist.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional override for extraction report output directory.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    dataset_cfg = cfg.get("dataset", {})
    preprocessing_cfg = cfg.get("preprocessing", {})
    mediapipe_cfg = cfg.get("mediapipe", {})
    logging_cfg = cfg.get("logging", {})

    raw_video_dir = Path(dataset_cfg["raw_video_dir"])
    landmarks_dir = Path(dataset_cfg["landmarks_dir"])

    num_frames = int(preprocessing_cfg["num_frames"])
    frame_size = int(preprocessing_cfg["frame_size"])
    max_hands = int(preprocessing_cfg["max_hands"])

    model_path = args.model_path or Path(mediapipe_cfg.get("model_path", "models/hand_landmarker.task"))
    fps = float(args.fps if args.fps is not None else mediapipe_cfg.get("fps", 30.0))
    input_color = args.input_color or mediapipe_cfg.get("input_color", "rgb")
    log_dir = args.log_dir or Path(logging_cfg.get("landmark_log_dir", "logs/landmark_extraction"))

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model file not found: {model_path}\n"
            "Run: py scripts/download_hand_landmarker_model.py\n"
            "Expected output: models/hand_landmarker.task"
        )

    landmarks_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(raw_video_dir.glob("**/*.mp4"))

    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    if not video_paths:
        print(f"No .mp4 files found under {raw_video_dir}. Add WLASL videos before running.")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = log_dir / f"extraction_report_{run_id}.csv"
    failed_path = log_dir / f"failed_videos_{run_id}.txt"
    latest_failed_path = log_dir / "failed_videos_latest.txt"
    latest_report_path = log_dir / "extraction_report_latest.csv"

    print(f"Found {len(video_paths)} videos to process.")
    print(f"Using MediaPipe model: {model_path}")
    print(f"Landmarks directory: {landmarks_dir}")
    print(f"Report will be saved to: {report_path}")

    report_rows = []
    saved_count = 0
    skipped_count = 0
    failed_count = 0

    for video_path in video_paths:
        relative = video_path.relative_to(raw_video_dir).with_suffix(".npy")
        output_path = landmarks_dir / relative

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing {output_path}")
            report_rows.append(
                {
                    "status": "skipped",
                    "video_path": str(video_path),
                    "output_path": str(output_path),
                    "shape": "",
                    "error": "",
                }
            )
            skipped_count += 1
            write_report(report_rows, latest_report_path)
            write_failed_list(report_rows, latest_failed_path)
            continue

        try:
            frames = sample_video_frames(
                video_path,
                num_frames=num_frames,
                frame_size=frame_size,
            )

            # A fresh HandLandmarker per video avoids MediaPipe video timestamp restart errors.
            with HandLandmarkExtractor(
                model_path=model_path,
                max_hands=max_hands,
                fps=fps,
                input_color=input_color,
            ) as extractor:
                sequence = extractor.extract_from_frames(frames)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, sequence)

            shape_text = str(tuple(sequence.shape))
            print(f"Saved {output_path} shape={shape_text}")

            report_rows.append(
                {
                    "status": "saved",
                    "video_path": str(video_path),
                    "output_path": str(output_path),
                    "shape": shape_text,
                    "error": "",
                }
            )
            saved_count += 1

        except Exception as exc:
            error_text = str(exc)
            print(f"FAILED {video_path}: {error_text}")

            report_rows.append(
                {
                    "status": "failed",
                    "video_path": str(video_path),
                    "output_path": str(output_path),
                    "shape": "",
                    "error": error_text,
                }
            )
            failed_count += 1

        # Continuously update latest reports during long full-dataset runs.
        # If the run is interrupted, partial reports are still available.
        write_report(report_rows, latest_report_path)
        write_failed_list(report_rows, latest_failed_path)

    write_report(report_rows, report_path)
    failed_video_paths = write_failed_list(report_rows, failed_path)

    # Also update stable latest files after completion.
    write_report(report_rows, latest_report_path)
    write_failed_list(report_rows, latest_failed_path)

    print()
    print("Landmark extraction complete.")
    print(f"Saved:   {saved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed:  {failed_count}")
    print()
    print(f"Full report:        {report_path}")
    print(f"Latest report:      {latest_report_path}")
    print(f"Failed video list:  {failed_path}")
    print(f"Latest failed list: {latest_failed_path}")

    if failed_video_paths:
        print()
        print("To preview deletion of failed videos, run:")
        print(f"py scripts/delete_failed_videos.py --failed-list {latest_failed_path}")
        print()
        print("To delete failed videos after reviewing the failed list, run:")
        print(f"py scripts/delete_failed_videos.py --failed-list {latest_failed_path} --delete")


if __name__ == "__main__":
    main()
