from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class LandmarkSequenceDatasetV2(Dataset):
    def __init__(
        self,
        manifest_csv,
        split=None,
        max_classes=None,
        normalize="wrist",
        use_delta=False,
        augment=False,
        noise_std=0.01,
        frame_dropout=0.05,
    ):
        self.manifest_csv = Path(manifest_csv)
        self.normalize = normalize
        self.use_delta = use_delta
        self.augment = augment
        self.noise_std = noise_std
        self.frame_dropout = frame_dropout

        if not self.manifest_csv.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_csv}")

        df = pd.read_csv(self.manifest_csv)

        required = ["gloss", "label_id", "split", "landmark_path"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in manifest: {missing}")

        df["split"] = df["split"].astype(str).str.lower()

        if split is not None:
            df = df[df["split"] == split.lower()].copy()

        if max_classes is not None:
            keep_ids = sorted(df["label_id"].unique())[: int(max_classes)]
            df = df[df["label_id"].isin(keep_ids)].copy()

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

        self.input_dim = 252 if self.use_delta else 126

    def __len__(self):
        return len(self.df)

    def _wrist_normalize(self, arr):
        # arr: (T, 126) -> (T, 2, 21, 3)
        x = arr.reshape(arr.shape[0], 2, 21, 3).copy()

        for t in range(x.shape[0]):
            for h in range(2):
                hand = x[t, h]
                if not np.any(hand != 0):
                    continue

                wrist = hand[0].copy()
                hand = hand - wrist

                # Scale by mean 2D distance from wrist to reduce camera distance effects.
                dist = np.linalg.norm(hand[:, :2], axis=1)
                scale = np.mean(dist[dist > 0]) if np.any(dist > 0) else 1.0
                if scale < 1e-6:
                    scale = 1.0

                x[t, h] = hand / scale

        return x.reshape(arr.shape).astype("float32")

    def _zscore_normalize(self, arr):
        valid = np.any(arr != 0, axis=1)
        if not valid.any():
            return arr.astype("float32")

        valid_values = arr[valid]
        mean = valid_values.mean(axis=0, keepdims=True)
        std = valid_values.std(axis=0, keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        arr = arr.copy()
        arr[valid] = (arr[valid] - mean) / std
        return arr.astype("float32")

    def _normalize(self, arr):
        arr = arr.astype("float32")

        if self.normalize == "none":
            return arr
        if self.normalize == "zscore":
            return self._zscore_normalize(arr)
        if self.normalize == "wrist":
            return self._wrist_normalize(arr)

        raise ValueError("normalize must be one of: none, zscore, wrist")

    def _add_delta(self, arr):
        delta = np.zeros_like(arr)
        delta[1:] = arr[1:] - arr[:-1]
        return np.concatenate([arr, delta], axis=1).astype("float32")

    def _augment(self, arr):
        if not self.augment:
            return arr

        out = arr.copy()

        if self.noise_std > 0:
            valid = np.any(out != 0, axis=1)
            noise = np.random.normal(0, self.noise_std, size=out.shape).astype("float32")
            out[valid] = out[valid] + noise[valid]

        if self.frame_dropout > 0:
            mask = np.random.rand(out.shape[0]) < self.frame_dropout
            out[mask] = 0

        # Small temporal shift.
        if np.random.rand() < 0.5:
            shift = np.random.choice([-2, -1, 1, 2])
            out = np.roll(out, shift=shift, axis=0)

        return out.astype("float32")

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

        arr = self._normalize(arr)
        arr = self._augment(arr)

        if self.use_delta:
            arr = self._add_delta(arr)

        x = torch.from_numpy(arr)
        y = torch.tensor(int(row["label_id"]), dtype=torch.long)

        return x, y
