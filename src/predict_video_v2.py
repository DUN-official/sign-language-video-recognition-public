import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from models_v2 import build_model
from preprocess_video import sample_video_frames


def load_yaml(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_mediapipe_hands_module():
    """Return the classic MediaPipe Hands solution.

    The MediaPipe Tasks HandLandmarker API can fail to load native shared
    libraries on Streamlit Cloud. The classic Hands solution is enough for this
    app because the model only needs 2 hands x 21 landmarks x 3 coordinates.
    """
    try:
        import mediapipe as mp

        return mp.solutions.hands
    except AttributeError:
        from mediapipe.python.solutions import hands

        return hands


def normalize_video_id(video_id) -> str:
    return str(video_id).strip().zfill(5)


def resolve_dataset_config(train_config_path: Path | None, dataset_config_path: Path | None):
    if dataset_config_path is not None:
        return Path(dataset_config_path)

    train_cfg = load_yaml(train_config_path or Path("configs/train.yaml"))
    config_path = train_cfg.get("dataset", {}).get("config_path")
    if config_path:
        return Path(config_path)

    return Path("configs/dataset.yaml")


def resolve_manifest_path(train_cfg: dict, dataset_cfg: dict) -> Path | None:
    train_dataset_cfg = train_cfg.get("dataset", {})
    config_dataset_cfg = dataset_cfg.get("dataset", {})

    candidates = [
        train_dataset_cfg.get("manifest_path"),
        train_dataset_cfg.get("landmark_manifest_csv"),
        config_dataset_cfg.get("landmark_manifest_csv"),
        config_dataset_cfg.get("downloaded_manifest_csv"),
        config_dataset_cfg.get("processed_manifest"),
        config_dataset_cfg.get("original_manifest_csv"),
        "data/processed/wlasl100_manifest_landmarks.csv",
        "data/processed/wlasl100_manifest_downloaded.csv",
        "data/processed/wlasl100_manifest.csv",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists():
            return path

    return None


def get_true_label_for_video(video_path: Path, train_cfg: dict, dataset_cfg: dict):
    manifest_path = resolve_manifest_path(train_cfg, dataset_cfg)

    if manifest_path is None:
        return None

    try:
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
        for col in ("raw_video_path", "video_path", "path", "filepath", "file_path"):
            if col in df.columns:
                matches = df[df[col].astype(str).apply(lambda p: Path(p).stem == video_path.stem)]
                if not matches.empty:
                    match = matches.iloc[0]
                    break

    if match is None:
        return {
            "found": False,
            "video_id": video_id,
            "manifest_path": str(manifest_path),
        }

    result = {
        "found": True,
        "video_id": video_id,
        "manifest_path": str(manifest_path),
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


def labels_match(predicted_label, true_label_info):
    if not true_label_info or not true_label_info.get("found"):
        return False

    predicted_text = str(predicted_label).strip().lower()
    true_gloss = str(true_label_info.get("gloss", "")).strip().lower()

    if true_gloss and predicted_text == true_gloss:
        return True

    return False


def video_to_landmark_sequence(
    video_path: Path,
    dataset_cfg: dict,
    num_frames: int = 32,
    frame_size: int | None = None,
):
    preprocessing_cfg = dataset_cfg.get("preprocessing", {})

    frame_size = int(frame_size or preprocessing_cfg.get("frame_size", 224))
    max_hands = int(preprocessing_cfg.get("max_hands", 2))

    frames = sample_video_frames(
        video_path,
        num_frames=num_frames,
        frame_size=frame_size,
    )

    mp_hands = get_mediapipe_hands_module()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    sequence = []

    try:
        for frame in frames:
            result = hands.process(frame)
            frame_landmarks = np.zeros((max_hands, 21, 3), dtype=np.float32)

            if result.multi_hand_landmarks:
                for hand_idx, hand in enumerate(result.multi_hand_landmarks[:max_hands]):
                    for point_idx, point in enumerate(hand.landmark):
                        frame_landmarks[hand_idx, point_idx] = [
                            point.x,
                            point.y,
                            point.z,
                        ]

            sequence.append(frame_landmarks.reshape(-1))
    finally:
        hands.close()

    return np.stack(sequence).astype("float32")


def wrist_normalize(arr):
    x = arr.reshape(arr.shape[0], 2, 21, 3).copy()

    for t in range(x.shape[0]):
        for h in range(2):
            hand = x[t, h]
            if not np.any(hand != 0):
                continue

            wrist = hand[0].copy()
            hand = hand - wrist

            dist = np.linalg.norm(hand[:, :2], axis=1)
            scale = np.mean(dist[dist > 0]) if np.any(dist > 0) else 1.0
            if scale < 1e-6:
                scale = 1.0

            x[t, h] = hand / scale

    return x.reshape(arr.shape).astype("float32")


def zscore_normalize(arr):
    arr = arr.astype("float32")
    valid = np.any(arr != 0, axis=1)

    if not valid.any():
        return arr

    valid_values = arr[valid]
    mean = valid_values.mean(axis=0, keepdims=True)
    std = valid_values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    arr = arr.copy()
    arr[valid] = (arr[valid] - mean) / std
    return arr


def normalize_landmarks(arr, mode):
    if mode == "none":
        return arr.astype("float32")
    if mode == "zscore":
        return zscore_normalize(arr)
    if mode == "wrist":
        return wrist_normalize(arr)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def add_delta(arr):
    delta = np.zeros_like(arr)
    delta[1:] = arr[1:] - arr[:-1]
    return np.concatenate([arr, delta], axis=1).astype("float32")


def extract_landmarks_from_video(
    video_path,
    num_frames=32,
    normalize="wrist",
    use_delta=False,
    dataset_config_path=None,
    frame_size=None,
):
    dataset_config = resolve_dataset_config(
        train_config_path=Path("configs/train.yaml"),
        dataset_config_path=dataset_config_path,
    )
    dataset_cfg = load_yaml(dataset_config)

    features = video_to_landmark_sequence(
        video_path=Path(video_path),
        dataset_cfg=dataset_cfg,
        num_frames=num_frames,
        frame_size=frame_size,
    )

    features = normalize_landmarks(features, normalize)

    if use_delta:
        features = add_delta(features)

    return features


def load_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = build_model(
        checkpoint["model_name"],
        num_classes=checkpoint["num_classes"],
        input_dim=checkpoint.get("input_dim", 126),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_layers=checkpoint.get("num_layers", 2),
        num_heads=checkpoint.get("num_heads", 4),
        dropout=checkpoint.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    labels = pd.DataFrame(checkpoint["labels"])
    id_to_label = dict(zip(labels["label_id"], labels["gloss"]))

    return model, id_to_label, device, checkpoint


def predict_video(
    video_path,
    checkpoint_path,
    train_config_path=None,
    top_k=5,
    num_frames=32,
    dataset_config_path=None,
    frame_size=None,
    include_true_label=False,
):
    video_path = Path(video_path)
    train_config = Path(train_config_path) if train_config_path is not None else Path("configs/train.yaml")
    train_cfg = load_yaml(train_config)
    dataset_config = resolve_dataset_config(train_config, dataset_config_path)
    dataset_cfg = load_yaml(dataset_config)

    model, id_to_label, device, checkpoint = load_checkpoint(checkpoint_path)
    true_label_info = get_true_label_for_video(video_path, train_cfg, dataset_cfg)

    landmarks = extract_landmarks_from_video(
        video_path,
        num_frames=num_frames,
        normalize=checkpoint.get("normalize", "wrist"),
        use_delta=checkpoint.get("use_delta", False),
        dataset_config_path=dataset_config_path,
        frame_size=frame_size,
    )

    x = torch.from_numpy(landmarks).float().unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        top = probs.topk(min(top_k, probs.shape[0]))

    predictions = []
    for rank, (class_id, prob) in enumerate(zip(top.indices.tolist(), top.values.tolist()), start=1):
        label_id = int(class_id)
        gloss = id_to_label.get(label_id, str(label_id))
        predictions.append(
            {
                "rank": rank,
                "class_index": label_id,
                "label_id": label_id,
                "label": gloss,
                "gloss": gloss,
                "probability": float(prob),
                "confidence": float(prob),
                "is_true_label": labels_match(gloss, true_label_info),
            }
        )

    if include_true_label:
        return predictions, landmarks, true_label_info

    return predictions, landmarks


def save_upload_to_temp(uploaded_file, suffix=".mp4"):
    if suffix is None:
        suffix = Path(getattr(uploaded_file, "name", "uploaded_video.mp4")).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        if hasattr(uploaded_file, "getbuffer"):
            tmp.write(uploaded_file.getbuffer())
        else:
            tmp.write(uploaded_file.read())
        return Path(tmp.name)


def main():
    parser = argparse.ArgumentParser(
        description="Predict an ASL sign from a video file using a V2 checkpoint and MediaPipe Tasks landmarks."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--dataset-config", type=Path, default=None)
    parser.add_argument("--frame-size", type=int, default=None)
    args = parser.parse_args()

    predictions, landmarks, true_label_info = predict_video(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        num_frames=args.num_frames,
        dataset_config_path=args.dataset_config,
        frame_size=args.frame_size,
        include_true_label=True,
    )

    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Landmark shape: {landmarks.shape}")

    if true_label_info and true_label_info.get("found"):
        print(
            "True label: "
            f"{true_label_info.get('gloss', 'UNKNOWN')} "
            f"(label_id={true_label_info.get('label_id', 'UNKNOWN')}, "
            f"split={true_label_info.get('split', 'UNKNOWN')})"
        )
    elif true_label_info:
        print(f"True label: not found in manifest for video_id={true_label_info.get('video_id')}")
        print(f"Manifest checked: {true_label_info.get('manifest_path')}")
    else:
        print("True label: unavailable, no readable manifest found")

    print("Top predictions:")
    for pred in predictions:
        marker = " <-- TRUE LABEL" if pred["is_true_label"] else ""
        print(
            f"{pred['rank']}. {pred['gloss']} "
            f"(label_id={pred['label_id']}, confidence={pred['confidence']:.4f})"
            f"{marker}"
        )


if __name__ == "__main__":
    main()
