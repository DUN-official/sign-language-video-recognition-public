# Dataset Workflow

The dataset pipeline uses fixed manifest paths in the YAML configs so the user does not need to manually switch files between stages.

## Key Manifests

For WLASL500:

```text
original_manifest_csv: data/processed/wlasl500_manifest.csv
downloaded_manifest_csv: data/processed/wlasl500_manifest_downloaded.csv
landmark_manifest_csv: data/processed/wlasl500_manifest_landmarks.csv
missing_landmark_manifest_csv: data/processed/wlasl500_manifest_missing_landmarks.csv
```

## Rebuild Sequence

Place WLASL metadata at:

```text
data/raw/WLASL/WLASL_v0.3.json
```

Create WLASL500 subset:

```powershell
py src/prepare_wlasl_subset.py --metadata data/raw/WLASL/WLASL_v0.3.json --subset-size 500 --output-root data/raw/WLASL500 --manifest data/processed/wlasl500_manifest.csv
```

Download videos:

```powershell
py src/download_wlasl_subset.py --index data/raw/WLASL500/WLASL500.json --output-dir data/raw/WLASL500/videos
```

Verify downloads:

```powershell
py src/verify_wlasl_downloads.py --config configs/dataset_wlasl500.yaml
```

Extract landmarks:

```powershell
py src/extract_landmarks.py --config configs/dataset_wlasl500.yaml
```

Delete failed/corrupted videos if needed:

```powershell
py scripts/delete_failed_videos.py --failed-list logs/landmark_extraction_wlasl500/failed_videos_latest.txt --delete
```

Build final landmark manifest:

```powershell
py src/build_landmark_manifest.py --config configs/dataset_wlasl500.yaml
```

Analyze landmark quality:

```powershell
py src/analyze_landmark_quality.py --manifest data/processed/wlasl500_manifest_landmarks.csv --output data/processed/wlasl500_landmark_quality.csv
```

Create final best6 manifest:

```powershell
py src/create_best_class_manifest.py --quality-csv data/processed/wlasl500_landmark_quality.csv --output data/processed/wlasl500_best6_manifest.csv --num-classes 6 --min-train 5 --min-val 1 --min-test 1 --min-quality-score 0.55
```

