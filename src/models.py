import torch
import torch.nn as nn


class BiLSTMLandmarkClassifier(nn.Module):
    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        num_layers=2,
        num_classes=100,
        dropout=0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        # x: (batch, frames, features)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


class GRULandmarkClassifier(nn.Module):
    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        num_layers=2,
        num_classes=100,
        dropout=0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.gru(x)
        pooled = out.mean(dim=1)
        return self.classifier(pooled)


def build_model(model_name, num_classes, input_dim=126, hidden_dim=256, num_layers=2, dropout=0.3):
    model_name = model_name.lower()

    if model_name == "bilstm":
        return BiLSTMLandmarkClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    if model_name == "gru":
        return GRULandmarkClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError("Unsupported model. Use: bilstm or gru")
