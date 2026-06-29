from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandmarkSequenceDataset(Dataset):
    def __init__(self, manifest_csv, split=None, max_classes=None, normalize=True):
        self.manifest_csv = Path(manifest_csv)
        self.normalize = normalize

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        df = pd.read_csv(self.manifest_csv)

        required = ["gloss", "label_id", "split", "landmark_path"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in manifest: {missing}")

        if split is not None:
            df = df[df["split"].astype(str).str.lower() == split.lower()].copy()

        if max_classes is not None:
            keep_ids = sorted(df["label_id"].unique())[: int(max_classes)]
            df = df[df["label_id"].isin(keep_ids)].copy()

            # Remap labels to contiguous ids after filtering.
            old_ids = sorted(df["label_id"].unique())
            remap = {old: new for new, old in enumerate(old_ids)}
            df["label_id"] = df["label_id"].map(remap).astype(int)

        df = df.reset_index(drop=True)

        if df.empty:
            raise ValueError(
                f"No rows available for split={split}, max_classes={max_classes}."
            )

        self.df = df
        self.num_classes = int(df["label_id"].nunique())
        self.classes = (
            df[["label_id", "gloss"]]
            .drop_duplicates()
            .sort_values("label_id")
            .reset_index(drop=True)
        )

    def __len__(self):
        return len(self.df)

    def _normalize_landmarks(self, arr):
        # arr shape: (T, 126). Zero rows mean no hands detected for that frame.
        arr = arr.astype("float32")
        valid = np.any(arr != 0, axis=1)

        if not valid.any():
            return arr

        # Normalize using only detected frames to reduce sensitivity to image position.
        valid_values = arr[valid]
        mean = valid_values.mean(axis=0, keepdims=True)
        std = valid_values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        arr[valid] = (arr[valid] - mean) / std
        return arr

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["landmark_path"])

        if not path.exists():
            raise FileNotFoundError(f"Landmark file not found: {path}")

        arr = np.load(path).astype("float32")

        if arr.ndim != 2:
            raise ValueError(f"Expected 2D landmark array, got {arr.shape}: {path}")

        if arr.shape[1] != 126:
            raise ValueError(f"Expected feature dimension 126, got {arr.shape}: {path}")

        if self.normalize:
            arr = self._normalize_landmarks(arr)

        x = torch.from_numpy(arr)
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)

        return x, y
