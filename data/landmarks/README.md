# Landmark Data Folder

This folder is intentionally left empty in the clean portfolio version of the repository.

Landmark `.npy` files are generated from the downloaded videos using MediaPipe. They are not included because they can be large and are reproducible from the scripts.

To regenerate landmarks, first download and verify the WLASL videos, then run:

```powershell
py src/extract_landmarks.py --config configs/dataset_wlasl500.yaml
```

After extraction, generate the final landmark training manifest:

```powershell
py src/build_landmark_manifest.py --config configs/dataset_wlasl500.yaml
```

Expected local structure after extraction:

```text
data/landmarks/wlasl500/
data/processed/wlasl500_manifest_landmarks.csv
data/processed/wlasl500_manifest_missing_landmarks.csv
```

See:

```text
README.md
docs/DATASET_WORKFLOW.md
```

Do not commit generated landmark arrays to GitHub unless you intentionally create a small sample set.

