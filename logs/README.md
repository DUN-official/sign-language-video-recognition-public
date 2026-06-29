# Logs Folder

This folder is intentionally left empty 

Logs are generated when running dataset verification, landmark extraction, and cleanup scripts. They are useful for debugging local runs but are not needed for reviewing the final demo.

Example generated log files:

```text
logs/landmark_extraction_wlasl500/failed_videos_latest.txt
logs/landmark_extraction_wlasl500/extraction_report_latest.csv
```

Typical workflow:

```powershell
py src/extract_landmarks.py --config configs/dataset_wlasl500.yaml

py scripts/delete_failed_videos.py --failed-list logs/landmark_extraction_wlasl500/failed_videos_latest.txt --delete

py src/verify_wlasl_downloads.py --config configs/dataset_wlasl500.yaml
```

See:

```text
README.md
docs/DATASET_WORKFLOW.md
docs/LIMITATIONS.md
```

Do not commit run-specific logs unless they are intentionally included as small examples.

