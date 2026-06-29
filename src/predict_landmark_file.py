import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import build_model


def normalize_landmarks(arr):
    arr = arr.astype("float32")
    valid = np.any(arr != 0, axis=1)
    if not valid.any():
        return arr

    valid_values = arr[valid]
    mean = valid_values.mean(axis=0, keepdims=True)
    std = valid_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    arr[valid] = (arr[valid] - mean) / std
    return arr


def main():
    parser = argparse.ArgumentParser(description="Predict one landmark .npy file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmark-file", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    landmark_path = Path(args.landmark_file)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not landmark_path.exists():
        raise FileNotFoundError(f"Landmark file not found: {landmark_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        input_dim=checkpoint.get("input_dim", 126),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_layers=checkpoint.get("num_layers", 2),
        dropout=checkpoint.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = pd.DataFrame(checkpoint["labels"])
    id_to_label = dict(zip(labels["label_id"], labels["gloss"]))

    arr = np.load(landmark_path)
    arr = normalize_landmarks(arr)
    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = probs.topk(min(args.top_k, probs.shape[0]))

    print(f"Prediction for: {landmark_path}")
    for rank, (class_id, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
        print(f"{rank}. {id_to_label.get(class_id, str(class_id))}: {prob:.4f}")


if __name__ == "__main__":
    main()
