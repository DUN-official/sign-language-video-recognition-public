import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train temporal sign-language recognition model.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    args = parser.parse_args()

    if not args.config.exists():
        raise FileNotFoundError(f"Train config not found: {args.config}")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset_cfg = cfg["dataset"]

    manifest_path = Path(dataset_cfg["manifest_path"])
    landmarks_dir = Path(dataset_cfg["landmarks_dir"])

    print("Training entry point loaded.")
    print("Landmark directory:", landmarks_dir)
    print("Training manifest:", manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}\n"
            "Run: py src/build_landmark_manifest.py --config configs/dataset.yaml"
        )

    if not landmarks_dir.exists():
        raise FileNotFoundError(
            f"Landmarks directory not found: {landmarks_dir}\n"
            "Run: py src/extract_landmarks.py --config configs/dataset.yaml"
        )

    print("Manifest and landmark directory found.")
    print("Implementation note: add Dataset/DataLoader and training loop here.")


if __name__ == "__main__":
    main()
