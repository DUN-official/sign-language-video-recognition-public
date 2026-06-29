import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from landmark_dataset import LandmarkSequenceDataset
from models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_topk(logits, y, k=1):
    k = min(k, logits.shape[1])
    preds = logits.topk(k, dim=1).indices
    return preds.eq(y.view(-1, 1)).any(dim=1).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)

    total_loss = 0.0
    total_count = 0
    total_top1 = 0.0
    total_top5 = 0.0

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += accuracy_topk(logits.detach(), y, k=1) * batch_size
        total_top5 += accuracy_topk(logits.detach(), y, k=5) * batch_size
        total_count += batch_size

    return {
        "loss": total_loss / max(total_count, 1),
        "top1": total_top1 / max(total_count, 1),
        "top5": total_top5 / max(total_count, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train landmark temporal model for WLASL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="models/wlasl100_landmarks")
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "gru"])
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = LandmarkSequenceDataset(args.manifest, split="train", max_classes=args.max_classes)
    val_ds = LandmarkSequenceDataset(args.manifest, split="val", max_classes=args.max_classes)

    num_classes = train_ds.num_classes
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")
    print(f"Classes: {num_classes}")

    train_ds.classes.to_csv(output_dir / "labels.csv", index=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        args.model,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_val_top1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
        }
        history.append(row)

        print(
            f"train loss={row['train_loss']:.4f}, top1={row['train_top1']:.4f}, top5={row['train_top5']:.4f}"
        )
        print(
            f"val   loss={row['val_loss']:.4f}, top1={row['val_top1']:.4f}, top5={row['val_top5']:.4f}"
        )

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "num_classes": num_classes,
                "input_dim": 126,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "max_classes": args.max_classes,
                "labels": train_ds.classes.to_dict(orient="records"),
                "best_val_top1": best_val_top1,
                "epoch": epoch,
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"Saved best model: {output_dir / 'best_model.pt'}")

    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print("\nTraining complete.")
    print(f"Best val top-1: {best_val_top1:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
