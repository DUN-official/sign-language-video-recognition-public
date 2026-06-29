import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Evaluate sign-language recognition model.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pt"))
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest

    if manifest_path is None:
        if not args.train_config.exists():
            raise FileNotFoundError(f"Train config not found: {args.train_config}")
        cfg = yaml.safe_load(args.train_config.read_text(encoding="utf-8"))
        manifest_path = Path(cfg["dataset"]["manifest_path"])

    print("Evaluation entry point loaded.")
    print("Checkpoint:", args.checkpoint)
    print("Manifest:", manifest_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Evaluation manifest not found: {manifest_path}\n"
            "Run: py src/build_landmark_manifest.py --config configs/dataset.yaml"
        )


if __name__ == "__main__":
    main()
