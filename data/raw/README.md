# Raw Data Folder

This folder is intentionally left empty in the clean portfolio version of the repository.

Raw WLASL metadata and downloaded videos are not included because they are large, generated locally, and may have dataset/license restrictions.

To rebuild this folder, follow the dataset setup instructions in:

```text
README.md
docs/DATASET_WORKFLOW.md
```

Expected local structure after setup:

```text
data/raw/WLASL/WLASL_v0.3.json
data/raw/WLASL500/WLASL500.json
data/raw/WLASL500/videos/
```

Typical rebuild commands:

```powershell
py src/prepare_wlasl_subset.py --metadata data/raw/WLASL/WLASL_v0.3.json --subset-size 500 --output-root data/raw/WLASL500 --manifest data/processed/wlasl500_manifest.csv

py src/download_wlasl_subset.py --index data/raw/WLASL500/WLASL500.json --output-dir data/raw/WLASL500/videos

py src/verify_wlasl_downloads.py --config configs/dataset_wlasl500.yaml
```

Do not commit full raw video folders to GitHub.

