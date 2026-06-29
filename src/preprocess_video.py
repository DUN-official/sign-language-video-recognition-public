from pathlib import Path

import cv2
import numpy as np


def sample_video_frames(video_path: Path, num_frames: int = 32, frame_size: int = 224):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames found in video: {video_path}")

    indices = np.linspace(0, len(frames) - 1, num=num_frames).astype(int)
    return [frames[i] for i in indices]
