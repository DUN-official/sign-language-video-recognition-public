import argparse
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from mediapipe_tasks_landmarks import HandLandmarkExtractor
from preprocess_video import sample_video_frames


class ProjectedBiLSTMClassifier(nn.Module):
    """
    Fallback architecture matching checkpoints with keys like:

        input_proj.0.weight
        input_proj.1.weight
        lstm.weight_ih_l0
        lstm.weight_ih_l0_reverse
        classifier.0.weight
        classifier.2.weight

    This is used when the repo's local build_model(...) signature is not called successfully.
    """

    def __init__(
        self,
        raw_input_size: int,
        projected_input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(raw_input_size, projected_input_size),
            nn.LayerNorm(projected_input_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=projected_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, num_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        encoded, _ = self.lstm(x)
        pooled = encoded.mean(dim=1)
        return self.classifier(pooled)


def load_yaml(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_video_id(video_id) -> str:
    """
    Normalize video ID values so:
        626   -> 00626
        00626 -> 00626
    """
    return str(video_id).strip().zfill(5)


def save_upload_to_temp(uploaded_file, suffix: str | None = None) -> Path:
    """
    Save a Streamlit uploaded file to a temporary video file and return its path.

    This keeps app/streamlit_app.py compatible with:
        from predict_video import predict_video, save_upload_to_temp

    Supports Streamlit UploadedFile objects that expose getbuffer(), read(), and name.
    """
    if uploaded_file is None:
        raise ValueError("No uploaded file was provided.")

    if suffix is None:
        uploaded_name = getattr(uploaded_file, "name", "uploaded_video.mp4")
        suffix = Path(uploaded_name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if hasattr(uploaded_file, "getbuffer"):
            tmp.write(uploaded_file.getbuffer())
        else:
            tmp.write(uploaded_file.read())
        return Path(tmp.name)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not find model weights in checkpoint. Expected one of: "
        "'model_state_dict', 'state_dict', 'model', or a raw state_dict."
    )


def strip_module_prefix(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def infer_architecture(state_dict: dict, train_cfg: dict) -> dict:
    model_cfg = dict(train_cfg.get("model", {}))
    dataset_cfg = dict(train_cfg.get("dataset", {}))

    dropout = float(model_cfg.get("dropout", 0.3))

    if "input_proj.0.weight" in state_dict:
        raw_input_size = int(state_dict["input_proj.0.weight"].shape[1])
        projected_input_size = int(state_dict["input_proj.0.weight"].shape[0])
    elif "lstm.weight_ih_l0" in state_dict:
        raw_input_size = int(model_cfg.get("input_size", 126))
        projected_input_size = int(state_dict["lstm.weight_ih_l0"].shape[1])
    else:
        raw_input_size = int(model_cfg.get("input_size", 126))
        projected_input_size = int(model_cfg.get("projected_input_size", raw_input_size))

    if "lstm.weight_ih_l0" in state_dict:
        hidden_size = int(state_dict["lstm.weight_ih_l0"].shape[0] // 4)
    elif "encoder.weight_ih_l0" in state_dict:
        hidden_size = int(state_dict["encoder.weight_ih_l0"].shape[0] // 4)
    else:
        hidden_size = int(model_cfg.get("hidden_size", 256))

    layer_ids = []
    for key in state_dict:
        match = re.match(r"lstm\.weight_ih_l(\d+)$", key)
        if match:
            layer_ids.append(int(match.group(1)))
    if not layer_ids:
        for key in state_dict:
            match = re.match(r"encoder\.weight_ih_l(\d+)$", key)
            if match:
                layer_ids.append(int(match.group(1)))

    num_layers = max(layer_ids) + 1 if layer_ids else int(model_cfg.get("num_layers", 2))
    bidirectional = any("_reverse" in key for key in state_dict)

    if "classifier.2.weight" in state_dict:
        num_classes = int(state_dict["classifier.2.weight"].shape[0])
    elif "classifier.weight" in state_dict:
        num_classes = int(state_dict["classifier.weight"].shape[0])
    else:
        candidate_keys = [
            key
            for key, value in state_dict.items()
            if key.endswith(".weight") and torch.is_tensor(value) and value.ndim == 2
        ]
        num_classes = int(state_dict[candidate_keys[-1]].shape[0]) if candidate_keys else int(dataset_cfg.get("num_classes", 100))

    return {
        "raw_input_size": raw_input_size,
        "projected_input_size": projected_input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "bidirectional": bidirectional,
        "dropout": dropout,
    }


def try_local_build_model(arch: dict):
    try:
        from models import build_model
    except Exception:
        return None

    attempts = [
        lambda: build_model("bilstm", arch["num_classes"]),
        lambda: build_model("lstm", arch["num_classes"]),
        lambda: build_model("landmark_bilstm", arch["num_classes"]),
        lambda: build_model(
            "bilstm",
            arch["num_classes"],
            input_size=arch["raw_input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch["dropout"],
        ),
    ]

    for attempt in attempts:
        try:
            return attempt()
        except Exception:
            pass

    return None


def build_prediction_model(state_dict: dict, train_cfg: dict):
    arch = infer_architecture(state_dict, train_cfg)

    model = try_local_build_model(arch)

    if model is not None:
        try:
            model.load_state_dict(state_dict, strict=True)
            print("Loaded checkpoint with local models.build_model(...).")
            return model, arch
        except Exception:
            pass

    model = ProjectedBiLSTMClassifier(
        raw_input_size=arch["raw_input_size"],
        projected_input_size=arch["projected_input_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        num_classes=arch["num_classes"],
        bidirectional=arch["bidirectional"],
        dropout=arch["dropout"],
    )

    model.load_state_dict(state_dict, strict=True)
    print("Loaded checkpoint with internal ProjectedBiLSTMClassifier.")
    return model, arch


def checkpoint_label_names(checkpoint, num_classes: int):
    if not isinstance(checkpoint, dict):
        return None

    for key in ("idx_to_label", "index_to_label", "id_to_label"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            labels = []
            for i in range(num_classes):
                label = value.get(i, value.get(str(i), f"class_{i}"))
                labels.append(label)
            return labels

    for key in ("classes", "class_names", "labels", "idx_to_class"):
        value = checkpoint.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= num_classes:
            return list(value[:num_classes])

    label_encoder = checkpoint.get("label_encoder")
    if hasattr(label_encoder, "classes_"):
        classes = list(label_encoder.classes_)
        if len(classes) >= num_classes:
            return classes[:num_classes]

    return None


def manifest_label_names(train_cfg: dict, dataset_cfg: dict, num_classes: int):
    manifest_path = resolve_manifest_path(train_cfg, dataset_cfg)

    if manifest_path is None or not manifest_path.exists():
        return None

    try:
        import pandas as pd

        df = pd.read_csv(manifest_path)

        if "label_id" in df.columns and "gloss" in df.columns:
            labels_df = (
                df[["label_id", "gloss"]]
                .drop_duplicates()
                .sort_values("label_id")
            )
            labels = []
            for _, row in labels_df.iterrows():
                labels.append({"label_id": int(row["label_id"]), "gloss": str(row["gloss"])})
            if len(labels) >= num_classes:
                return labels[:num_classes]

        for col in ("gloss", "label", "class_name", "class"):
            if col in df.columns:
                labels = sorted(str(x) for x in df[col].dropna().unique())
                if len(labels) >= num_classes:
                    return labels[:num_classes]

    except Exception:
        return None

    return None


def get_label_names(checkpoint, train_cfg: dict, dataset_cfg: dict, num_classes: int):
    labels = checkpoint_label_names(checkpoint, num_classes)
    if labels is not None:
        return labels

    labels = manifest_label_names(train_cfg, dataset_cfg, num_classes)
    if labels is not None:
        return labels

    return [f"class_{i}" for i in range(num_classes)]


def resolve_dataset_config(train_cfg: dict, explicit_dataset_config: Path | None):
    if explicit_dataset_config is not None:
        return explicit_dataset_config

    dataset_cfg = train_cfg.get("dataset", {})
    config_path = dataset_cfg.get("config_path")

    if config_path:
        return Path(config_path)

    return Path("configs/dataset.yaml")


def resolve_manifest_path(train_cfg: dict, dataset_cfg: dict) -> Path | None:
    """
    Find the manifest most likely to contain true labels.
    Preference order:
      1. train config manifest_path / landmark_manifest_csv
      2. dataset.yaml landmark_manifest_csv
      3. dataset.yaml downloaded_manifest_csv
      4. dataset.yaml processed_manifest
      5. dataset.yaml original_manifest_csv
    """
    train_dataset_cfg = train_cfg.get("dataset", {})
    config_dataset_cfg = dataset_cfg.get("dataset", {})

    candidates = [
        train_dataset_cfg.get("manifest_path"),
        train_dataset_cfg.get("landmark_manifest_csv"),
        config_dataset_cfg.get("landmark_manifest_csv"),
        config_dataset_cfg.get("downloaded_manifest_csv"),
        config_dataset_cfg.get("processed_manifest"),
        config_dataset_cfg.get("original_manifest_csv"),
    ]

    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path

    return None


def get_true_label_for_video(video_path: Path, train_cfg: dict, dataset_cfg: dict):
    """
    Look up the true label/gloss for the input video using the manifest.

    This supports common manifest columns:
      video_id, gloss, label_id, label, split
    """
    manifest_path = resolve_manifest_path(train_cfg, dataset_cfg)

    if manifest_path is None:
        return None

    try:
        import pandas as pd

        df = pd.read_csv(manifest_path)
    except Exception:
        return None

    video_id = normalize_video_id(video_path.stem)

    match = None

    if "video_id" in df.columns:
        normalized_ids = df["video_id"].apply(normalize_video_id)
        matches = df[normalized_ids == video_id]
        if not matches.empty:
            match = matches.iloc[0]

    if match is None:
        # Fallback if a manifest stores video paths instead of video IDs.
        for col in ("video_path", "path", "filepath", "file_path"):
            if col in df.columns:
                matches = df[df[col].astype(str).apply(lambda p: Path(p).stem == video_path.stem)]
                if not matches.empty:
                    match = matches.iloc[0]
                    break

    if match is None:
        return {
            "video_id": video_id,
            "manifest_path": str(manifest_path),
            "found": False,
        }

    result = {
        "video_id": video_id,
        "manifest_path": str(manifest_path),
        "found": True,
    }

    for col in ("gloss", "label", "class_name", "class"):
        if col in match.index:
            result["gloss"] = str(match[col])
            break

    if "label_id" in match.index:
        try:
            result["label_id"] = int(match["label_id"])
        except Exception:
            result["label_id"] = str(match["label_id"])

    if "split" in match.index:
        result["split"] = str(match["split"])

    return result


def video_to_landmark_sequence(
    video_path: Path,
    dataset_cfg: dict,
    num_frames_override: int | None = None,
    frame_size_override: int | None = None,
):
    preprocessing_cfg = dataset_cfg.get("preprocessing", {})
    mediapipe_cfg = dataset_cfg.get("mediapipe", {})

    num_frames = int(num_frames_override or preprocessing_cfg.get("num_frames", 32))
    frame_size = int(frame_size_override or preprocessing_cfg.get("frame_size", 224))
    max_hands = int(preprocessing_cfg.get("max_hands", 2))

    model_path = Path(mediapipe_cfg.get("model_path", "models/hand_landmarker.task"))
    fps = float(mediapipe_cfg.get("fps", 30.0))
    input_color = mediapipe_cfg.get("input_color", "rgb")

    if not model_path.exists():
        raise FileNotFoundError(
            f"MediaPipe model file not found: {model_path}\n"
            "Run: py scripts/download_hand_landmarker_model.py"
        )

    frames = sample_video_frames(
        video_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )

    with HandLandmarkExtractor(
        model_path=model_path,
        max_hands=max_hands,
        fps=fps,
        input_color=input_color,
    ) as extractor:
        return extractor.extract_from_frames(frames)


def format_label(label):
    if isinstance(label, dict):
        if "gloss" in label:
            return str(label["gloss"])
        return str(label)
    return str(label)


def labels_match(predicted_label, true_label_info):
    if not true_label_info or not true_label_info.get("found"):
        return False

    predicted_text = format_label(predicted_label).strip().lower()
    true_gloss = str(true_label_info.get("gloss", "")).strip().lower()

    if true_gloss and predicted_text == true_gloss:
        return True

    true_label_id = true_label_info.get("label_id")
    if isinstance(predicted_label, dict) and true_label_id is not None:
        return str(predicted_label.get("label_id")) == str(true_label_id)

    return False


def predict_video(
    video_path: Path,
    checkpoint_path: Path,
    train_config_path: Path | int | None = Path("configs/train.yaml"),
    dataset_config_path: Path | None = None,
    top_k: int = 5,
    num_frames: int | None = None,
    frame_size: int | None = None,
    include_true_label: bool = False,
):
    # Backward compatibility:
    # Some older app code may call predict_video(video, checkpoint, 5),
    # where the third positional argument is top_k rather than train_config_path.
    if isinstance(train_config_path, int):
        top_k = train_config_path
        train_config_path = Path("configs/train.yaml")

    if train_config_path is None:
        train_config_path = Path("configs/train.yaml")

    video_path = Path(video_path)
    checkpoint_path = Path(checkpoint_path)
    train_config_path = Path(train_config_path)

    if dataset_config_path is not None:
        dataset_config_path = Path(dataset_config_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    train_cfg = load_yaml(train_config_path)
    resolved_dataset_config = resolve_dataset_config(train_cfg, dataset_config_path)
    dataset_cfg = load_yaml(resolved_dataset_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))

    model, arch = build_prediction_model(state_dict, train_cfg)
    model.eval()

    true_label_info = get_true_label_for_video(video_path, train_cfg, dataset_cfg)

    landmarks = video_to_landmark_sequence(
        video_path=video_path,
        dataset_cfg=dataset_cfg,
        num_frames_override=num_frames,
        frame_size_override=frame_size,
    )

    x = torch.from_numpy(landmarks).float().unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    num_classes = int(arch["num_classes"])
    label_names = get_label_names(checkpoint, train_cfg, dataset_cfg, num_classes)

    k = min(int(top_k), len(probabilities))
    top_probs, top_indices = torch.topk(probabilities, k=k)

    predictions = []
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), start=1):
        label = label_names[idx] if idx < len(label_names) else f"class_{idx}"

        predictions.append(
            {
                "rank": rank,
                "class_index": int(idx),
                "label_id": int(label.get("label_id", idx)) if isinstance(label, dict) else int(idx),

                # Main CLI/internal fields.
                "label": label,
                "probability": float(prob),

                # Backward-compatible Streamlit fields expected by app/streamlit_app.py.
                "gloss": format_label(label),
                "confidence": float(prob),

                "is_true_label": labels_match(label, true_label_info),
            }
        )

    if include_true_label:
        return predictions, landmarks, true_label_info

    return predictions, landmarks


def main():
    parser = argparse.ArgumentParser(
        description="Predict a WLASL sign class from one video using MediaPipe Tasks landmarks and a trained PyTorch checkpoint."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--frame-size", type=int, default=None)
    parser.add_argument(
        "--save-landmarks",
        type=Path,
        default=None,
        help="Optional path to save the extracted landmark sequence used for prediction.",
    )
    args = parser.parse_args()

    predictions, landmarks, true_label_info = predict_video(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        train_config_path=args.train_config,
        dataset_config_path=args.dataset_config,
        top_k=args.top_k,
        num_frames=args.num_frames,
        frame_size=args.frame_size,
        include_true_label=True,
    )

    if args.save_landmarks is not None:
        args.save_landmarks.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_landmarks, landmarks)
        print(f"Saved extracted landmarks: {args.save_landmarks} shape={landmarks.shape}")
        print()

    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Landmark shape: {landmarks.shape}")

    if true_label_info and true_label_info.get("found"):
        true_label_text = true_label_info.get("gloss", "UNKNOWN")
        label_id_text = true_label_info.get("label_id", "UNKNOWN")
        split_text = true_label_info.get("split", "UNKNOWN")
        print(f"True label: {true_label_text} (label_id={label_id_text}, split={split_text})")
    elif true_label_info:
        print(f"True label: not found in manifest for video_id={true_label_info.get('video_id')}")
        print(f"Manifest checked: {true_label_info.get('manifest_path')}")
    else:
        print("True label: unavailable, no readable manifest found")

    print()
    print("Top predictions:")

    for item in predictions:
        marker = " <-- TRUE LABEL" if item["is_true_label"] else ""
        print(
            f"{item['rank']}. {format_label(item['label'])} "
            f"(class_index={item['class_index']}, probability={item['probability']:.4f})"
            f"{marker}"
        )


if __name__ == "__main__":
    main()
