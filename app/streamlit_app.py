import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predict_video_v2 import predict_video, save_upload_to_temp


def load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_demo_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    required = {"gloss", "video_id", "demo_video_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Demo manifest missing required columns: {sorted(missing)}")

    return df


def normalize_video_id(video_id) -> str:
    return str(video_id).strip().zfill(5)


def resolve_manifest_path(train_config_path: Path, dataset_config_path: Path) -> Path | None:
    train_cfg = load_yaml(train_config_path)
    dataset_cfg = load_yaml(dataset_config_path)
    dataset500_cfg = load_yaml(ROOT_DIR / "configs" / "dataset_wlasl500.yaml")

    train_dataset = train_cfg.get("dataset", {})
    dataset = dataset_cfg.get("dataset", {})
    dataset500 = dataset500_cfg.get("dataset", {})

    candidates = [
        train_dataset.get("manifest_path"),
        train_dataset.get("landmark_manifest_csv"),
        dataset500.get("landmark_manifest_csv"),
        dataset500.get("landmark_manifest"),
        dataset500.get("downloaded_manifest_csv"),
        dataset500.get("processed_manifest"),
        dataset500.get("original_manifest_csv"),
        dataset.get("landmark_manifest_csv"),
        dataset.get("downloaded_manifest_csv"),
        dataset.get("processed_manifest"),
        dataset.get("original_manifest_csv"),
    ]

    for candidate in candidates:
        if candidate:
            path = ROOT_DIR / candidate if not Path(candidate).is_absolute() else Path(candidate)
            if path.exists():
                return path

    return None


def lookup_true_label_from_manifest(uploaded_filename: str, manifest_path: Path | None):
    """
    If the uploaded file is a dataset video such as 00626.mp4, look up its true label
    in the manifest. For arbitrary user videos, this will usually return not found.
    """
    if manifest_path is None or not manifest_path.exists():
        return {
            "found": False,
            "reason": "No readable manifest found.",
        }

    video_id = normalize_video_id(Path(uploaded_filename).stem)

    try:
        df = pd.read_csv(manifest_path)
    except Exception as exc:
        return {
            "found": False,
            "reason": f"Could not read manifest: {exc}",
            "manifest_path": str(manifest_path),
        }

    if "video_id" not in df.columns:
        return {
            "found": False,
            "reason": "Manifest has no video_id column.",
            "manifest_path": str(manifest_path),
        }

    normalized_ids = df["video_id"].apply(normalize_video_id)
    matches = df[normalized_ids == video_id]

    if matches.empty:
        return {
            "found": False,
            "video_id": video_id,
            "reason": "Uploaded filename does not match a dataset video_id.",
            "manifest_path": str(manifest_path),
        }

    row = matches.iloc[0]

    true_info = {
        "found": True,
        "video_id": video_id,
        "manifest_path": str(manifest_path),
    }

    if "gloss" in row.index:
        true_info["gloss"] = str(row["gloss"])

    if "label_id" in row.index:
        try:
            true_info["label_id"] = int(row["label_id"])
        except Exception:
            true_info["label_id"] = str(row["label_id"])

    if "split" in row.index:
        true_info["split"] = str(row["split"])

    return true_info


def normalize_prediction_rows(predictions):
    """
    Make prediction dictionaries compatible with the app even if predict_video.py
    uses slightly different key names internally.
    """
    rows = []

    for i, item in enumerate(predictions, start=1):
        row = dict(item)

        if "gloss" not in row:
            label = row.get("label", row.get("class_name", row.get("class", "")))
            if isinstance(label, dict):
                row["gloss"] = str(label.get("gloss", label))
                if "label_id" not in row and "label_id" in label:
                    row["label_id"] = label["label_id"]
            else:
                row["gloss"] = str(label)

        if "confidence" not in row:
            row["confidence"] = float(row.get("probability", row.get("score", 0.0)))

        if "label_id" not in row:
            row["label_id"] = int(row.get("class_index", row.get("index", i - 1)))

        row["rank"] = int(row.get("rank", i))
        row["confidence_pct"] = round(float(row["confidence"]) * 100, 2)

        rows.append(row)

    return rows


def call_predict_video(video_path, checkpoint_path, top_k, num_frames):
    """
    Supports both older and newer predict_video.py return formats:
      - predictions, landmarks
      - predictions, landmarks, true_label_info
    """
    try:
        result = predict_video(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            train_config_path=ROOT_DIR / "configs" / "train.yaml",
            dataset_config_path=ROOT_DIR / "configs" / "dataset.yaml",
            top_k=top_k,
            num_frames=num_frames,
        )
    except TypeError:
        result = predict_video(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            top_k=top_k,
            num_frames=num_frames,
        )

    if isinstance(result, tuple) and len(result) == 3:
        predictions, landmarks, true_label_info = result
    elif isinstance(result, tuple) and len(result) == 2:
        predictions, landmarks = result
        true_label_info = None
    else:
        raise RuntimeError("predict_video() returned an unexpected result format.")

    return predictions, landmarks, true_label_info


def true_label_in_top_k(true_gloss: str | None, prediction_rows):
    if not true_gloss:
        return False

    true_gloss = true_gloss.strip().lower()

    for row in prediction_rows:
        if str(row.get("gloss", "")).strip().lower() == true_gloss:
            return True

    return False


st.set_page_config(
    page_title="ASL Video Sign Recognition",
    page_icon="🤟",
    layout="wide",
)


st.sidebar.header("Settings")

checkpoint_path = st.sidebar.text_input(
    "Model checkpoint",
    value="models/final/wlasl500_best6_v3_finetuned/best_model.pt",
)

top_k = st.sidebar.slider("Top-k predictions", min_value=1, max_value=10, value=5)
num_frames = st.sidebar.slider("Sampled frames", min_value=8, max_value=64, value=64, step=8)

demo_manifest_text = st.sidebar.text_input(
    "Demo manifest",
    value="assets/demo_clips/final/wlasl500_best6/demo_manifest.csv",
)

st.sidebar.subheader("Model status")
st.sidebar.write(
    "Current model is an early landmark-based prototype. "
    "Low confidence predictions should be treated as uncertain."
)

st.title("ASL Video Sign Recognition")

st.write(
    "Upload a short isolated ASL sign video or choose a curated trained-class demo clip. "
    "The app extracts hand landmarks and predicts the most likely WLASL gloss."
)

source_mode = st.radio(
    "Video source",
    ["Curated trained-class demo clip", "Upload custom video"],
    horizontal=True,
)

demo_manifest_path = ROOT_DIR / demo_manifest_text
temp_video_path = None
input_display_name = ""
manual_true_label = ""
manifest_true_label = None

if source_mode == "Curated trained-class demo clip":
    try:
        demo_df = load_demo_manifest(demo_manifest_path)
    except Exception as exc:
        st.error("Could not read demo manifest.")
        st.exception(exc)
        st.stop()

    if demo_df.empty:
        st.warning(
            "No curated demo manifest found yet. Create one with "
            "`scripts/create_trained_class_demo_assets.py`, or switch to custom upload."
        )
        st.stop()

    glosses = sorted(demo_df["gloss"].astype(str).unique())
    selected_gloss = st.selectbox("Demo class", glosses)

    class_df = demo_df[demo_df["gloss"].astype(str) == selected_gloss].copy()
    class_df["display"] = class_df.apply(
        lambda row: f"{row['video_id']} | split={row.get('split', '')} | {Path(row['demo_video_path']).name}",
        axis=1,
    )
    selected_display = st.selectbox("Demo video", class_df["display"].tolist())
    selected_row = class_df[class_df["display"] == selected_display].iloc[0]

    temp_video_path = ROOT_DIR / str(selected_row["demo_video_path"])
    input_display_name = Path(temp_video_path).name

    if not temp_video_path.exists():
        st.error(f"Demo video not found: {temp_video_path}")
        st.stop()

    manifest_true_label = {
        "found": True,
        "video_id": normalize_video_id(selected_row["video_id"]),
        "gloss": str(selected_row["gloss"]),
        "label_id": selected_row.get("label_id", "UNKNOWN"),
        "split": selected_row.get("split", "UNKNOWN"),
        "manifest_path": str(demo_manifest_path),
    }

    st.video(str(temp_video_path))

else:
    uploaded_file = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    manual_true_label = st.text_input(
        "Optional expected/true gloss for custom uploads",
        value="",
        help=(
            "For WLASL dataset videos, the app can infer the true label from the manifest "
            "if the filename matches a video_id such as 00626.mp4. For custom uploads, type "
            "the expected gloss manually if you want a comparison."
        ),
    )

    if uploaded_file is None:
        st.info("Upload a short video to run prediction.")
        st.stop()

    temp_video_path = Path(save_upload_to_temp(uploaded_file))
    input_display_name = uploaded_file.name
    st.video(str(temp_video_path))

manifest_path = resolve_manifest_path(
    train_config_path=ROOT_DIR / "configs" / "train.yaml",
    dataset_config_path=ROOT_DIR / "configs" / "dataset.yaml",
)

if manifest_true_label is None:
    manifest_true_label = lookup_true_label_from_manifest(
        uploaded_filename=input_display_name,
        manifest_path=manifest_path,
    )

if st.button("Run prediction"):
    st.subheader("Prediction")

    checkpoint = ROOT_DIR / checkpoint_path

    with st.spinner("Extracting landmarks and running model..."):
        predictions, landmarks, predict_true_label_info = call_predict_video(
            video_path=temp_video_path,
            checkpoint_path=checkpoint,
            top_k=top_k,
            num_frames=num_frames,
        )

    prediction_rows = normalize_prediction_rows(predictions)

    true_info = manifest_true_label
    if predict_true_label_info and predict_true_label_info.get("found"):
        true_info = predict_true_label_info

    manual_label = manual_true_label.strip()

    top_pred = prediction_rows[0]

    st.metric(
        label="Top prediction",
        value=top_pred["gloss"],
        delta=f"{top_pred['confidence']:.2%} confidence",
    )

    if top_pred["confidence"] < 0.50:
        st.warning(
            "Low-confidence prediction. Treat this as a prototype result, not a reliable interpretation."
        )

    st.subheader("Ground truth / expected label")

    if true_info and true_info.get("found"):
        true_gloss = true_info.get("gloss", "UNKNOWN")
        true_label_id = true_info.get("label_id", "UNKNOWN")
        split = true_info.get("split", "UNKNOWN")

        st.success(
            f"Dataset true label: **{true_gloss}** "
            f"(label_id={true_label_id}, split={split})"
        )

        if true_label_in_top_k(true_gloss, prediction_rows):
            st.success("The true label appears in the top-k predictions.")
        else:
            st.error("The true label does not appear in the top-k predictions.")

        with st.expander("Ground-truth source"):
            st.write(f"Matched input `{input_display_name}` against a known label source.")
            st.write(f"Manifest: `{true_info.get('manifest_path')}`")

    elif manual_label:
        st.info(f"Manual expected label: **{manual_label}**")

        if true_label_in_top_k(manual_label, prediction_rows):
            st.success("The manually provided label appears in the top-k predictions.")
        else:
            st.warning("The manually provided label does not appear in the top-k predictions.")

    else:
        st.info(
            "True label unavailable. This is expected for custom uploads unless the filename "
            "matches a WLASL video_id in the manifest or you manually enter the expected gloss."
        )

        if true_info and true_info.get("reason"):
            with st.expander("Why no true label was found"):
                st.write(true_info["reason"])
                if true_info.get("manifest_path"):
                    st.write(f"Manifest checked: `{true_info['manifest_path']}`")

    st.subheader("Top-k predictions")

    df = pd.DataFrame(prediction_rows)
    display_cols = ["rank", "label_id", "gloss", "confidence_pct"]

    available_cols = [col for col in display_cols if col in df.columns]
    #st.dataframe(df[available_cols], use_container_width=True)
    st.dataframe(df[available_cols], use_container_width=True)

    st.subheader("Landmark extraction check")
    st.write(f"Extracted landmark tensor shape: `{landmarks.shape}`")

    valid_shapes = [(num_frames, 126), (num_frames, 252)]

    if tuple(landmarks.shape) in valid_shapes:
        st.success("Landmark tensor shape is valid for the current model input.")
    else:
        st.warning(
            "Landmark tensor shape differs from expected shape. "
            "Check preprocessing settings and model input size."
        )
