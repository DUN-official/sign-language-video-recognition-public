import math

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = torch.softmax(self.score(x), dim=1)
        return (x * weights).sum(dim=1)


class BiLSTMAttentionClassifier(nn.Module):
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

        self.pool = AttentionPooling(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        pooled = self.pool(out)
        return self.classifier(pooled)


class TransformerLandmarkClassifier(nn.Module):
    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        num_classes=100,
        dropout=0.2,
        max_frames=64,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.pool = AttentionPooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, T, D)
        t = x.shape[1]
        x = self.input_proj(x)
        x = x + self.pos_embed[:, :t, :]
        x = self.encoder(x)
        x = self.pool(x)
        return self.classifier(x)


class TemporalConvClassifier(nn.Module):
    def __init__(
        self,
        input_dim=126,
        hidden_dim=256,
        num_layers=4,
        num_classes=100,
        dropout=0.3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(num_layers):
            dilation = 2 ** i
            layers.extend(
                [
                    nn.Conv1d(
                        in_dim,
                        hidden_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.mean(dim=2)
        return self.classifier(x)


def build_model(
    model_name,
    num_classes,
    input_dim=126,
    hidden_dim=256,
    num_layers=2,
    dropout=0.3,
    num_heads=4,
):
    model_name = model_name.lower()

    if model_name in {"bilstm_attn", "bilstm_attention"}:
        return BiLSTMAttentionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    if model_name == "transformer":
        return TransformerLandmarkClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

    if model_name in {"tcn", "temporal_conv"}:
        return TemporalConvClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        )

    raise ValueError("Unsupported model. Use: bilstm_attn, transformer, or tcn")
