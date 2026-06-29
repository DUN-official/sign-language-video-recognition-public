import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from landmark_dataset import LandmarkSequenceDataset
from models import build_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained landmark model.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="results/test_predictions.csv")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    dataset = LandmarkSequenceDataset(
        args.manifest,
        split=args.split,
        max_classes=checkpoint.get("max_classes"),
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

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

    rows = []
    correct = 0
    total = 0
    top5_correct = 0

    offset = 0
    with torch.inference_mode():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            top5 = probs.topk(min(5, probs.shape[1]), dim=1)

            preds = top5.indices[:, 0]
            correct += (preds == y).sum().item()
            top5_correct += top5.indices.eq(y.view(-1, 1)).any(dim=1).sum().item()
            total += y.size(0)

            batch_df = dataset.df.iloc[offset : offset + y.size(0)].reset_index(drop=True)
            offset += y.size(0)

            for i in range(y.size(0)):
                pred_id = int(preds[i].item())
                true_id = int(y[i].item())

                row = {
                    "video_id": batch_df.loc[i, "video_id"],
                    "split": args.split,
                    "true_label_id": true_id,
                    "true_gloss": id_to_label.get(true_id, str(true_id)),
                    "pred_label_id": pred_id,
                    "pred_gloss": id_to_label.get(pred_id, str(pred_id)),
                    "confidence": float(probs[i, pred_id].item()),
                    "correct": pred_id == true_id,
                }

                for rank, class_id in enumerate(top5.indices[i].tolist(), start=1):
                    row[f"top{rank}_label_id"] = int(class_id)
                    row[f"top{rank}_gloss"] = id_to_label.get(int(class_id), str(class_id))
                    row[f"top{rank}_prob"] = float(top5.values[i, rank - 1].item())

                rows.append(row)

    top1_acc = correct / max(total, 1)
    top5_acc = top5_correct / max(total, 1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    metrics_path = output_path.with_name(output_path.stem + "_metrics.csv")
    pd.DataFrame(
        [{"split": args.split, "samples": total, "top1_accuracy": top1_acc, "top5_accuracy": top5_acc}]
    ).to_csv(metrics_path, index=False)

    print(f"Samples: {total}")
    print(f"Top-1 accuracy: {top1_acc:.4f}")
    print(f"Top-5 accuracy: {top5_acc:.4f}")
    print(f"Saved predictions: {output_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
