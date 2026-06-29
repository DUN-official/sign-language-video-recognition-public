from pathlib import Path
from urllib.request import urlretrieve


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

OUTPUT_PATH = Path("models/hand_landmarker.task")


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if OUTPUT_PATH.exists():
        print(f"Model already exists: {OUTPUT_PATH}")
        return

    print(f"Downloading MediaPipe Hand Landmarker model to {OUTPUT_PATH}")
    print(MODEL_URL)
    urlretrieve(MODEL_URL, OUTPUT_PATH)
    print("Download complete.")


if __name__ == "__main__":
    main()
