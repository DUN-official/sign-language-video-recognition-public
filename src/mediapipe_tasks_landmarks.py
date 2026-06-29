from pathlib import Path

import numpy as np


def empty_landmark_vector(
    max_hands: int = 2,
    landmarks_per_hand: int = 21,
    coordinates: int = 3,
) -> np.ndarray:
    """
    Returns a zero-filled landmark tensor.

    Output shape before flattening:
        (max_hands, 21, 3)

    Output shape after flattening for max_hands=2:
        126 values per frame
    """
    return np.zeros((max_hands, landmarks_per_hand, coordinates), dtype=np.float32)


def _to_uint8_rgb(frame: np.ndarray, input_color: str = "rgb") -> np.ndarray:
    """
    Convert a frame to uint8 RGB format for MediaPipe Tasks.

    MediaPipe Tasks expects SRGB/RGB image data. If your preprocessing function
    already converts OpenCV BGR frames to RGB, keep input_color='rgb'. If your
    frames come directly from cv2.VideoCapture, use input_color='bgr'.
    """
    arr = np.asarray(frame)

    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected frame shape (H, W, 3), got {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if input_color == "bgr":
        arr = arr[..., ::-1]

    return np.ascontiguousarray(arr)


class HandLandmarkExtractor:
    """
    MediaPipe Tasks-based hand landmark extractor.

    This replaces the older API pattern:
        mp.solutions.hands.Hands(...)

    with the newer Tasks API pattern:
        vision.HandLandmarker.create_from_options(...)
        landmarker.detect_for_video(...)

    For each sampled frame, this returns a flattened vector of:
        max_hands * 21 landmarks * 3 coordinates

    For max_hands=2, each frame becomes 126 values.
    """

    def __init__(
        self,
        model_path: Path | str,
        max_hands: int = 2,
        fps: float = 30.0,
        input_color: str = "rgb",
        min_hand_detection_confidence: float = 0.5,
        min_hand_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.model_path = Path(model_path)
        self.max_hands = int(max_hands)
        self.fps = float(fps)
        self.input_color = input_color

        self.min_hand_detection_confidence = float(min_hand_detection_confidence)
        self.min_hand_presence_confidence = float(min_hand_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)

        self._mp = None
        self._landmarker = None

    def __enter__(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Hand Landmarker model not found: {self.model_path}"
            )

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_hand_detection_confidence,
            min_hand_presence_confidence=self.min_hand_presence_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

        self._mp = mp
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._landmarker is not None:
            self._landmarker.close()

        self._mp = None
        self._landmarker = None

    def extract_from_frames(self, frames) -> np.ndarray:
        if self._landmarker is None or self._mp is None:
            raise RuntimeError("Use HandLandmarkExtractor as a context manager with 'with ... as extractor:'.")

        sequence = []

        for frame_idx, frame in enumerate(frames):
            rgb_frame = _to_uint8_rgb(frame, input_color=self.input_color)

            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            timestamp_ms = int(round(frame_idx * 1000.0 / self.fps))
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_landmarks = empty_landmark_vector(max_hands=self.max_hands)

            if result.hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks[: self.max_hands]):
                    for point_idx, point in enumerate(hand_landmarks):
                        frame_landmarks[hand_idx, point_idx] = [
                            point.x,
                            point.y,
                            point.z,
                        ]

            sequence.append(frame_landmarks.reshape(-1))

        return np.stack(sequence).astype(np.float32)
