# Streamlit Demo

Run:

```powershell
py -m streamlit run app/streamlit_app.py
```

Default settings:

```text
Model checkpoint: models/final/wlasl500_best6_v3_finetuned/best_model.pt
Demo manifest: assets/demo_clips/final/wlasl500_best6/demo_manifest.csv
Sampled frames: 64
Top-k predictions: 5
```

## Demo Modes

### Curated Trained-Class Demo Clip

Use this for the most reliable demonstration. The app loads clips from the exact classes used by the final model.

### Upload Custom Video

Use this for exploratory testing. The model will still predict, but results may be weak if the sign is outside the trained classes.

## Interpreting Output

The app shows:

```text
top prediction
confidence
dataset true label if available
whether true label appears in top-k
top-k predictions table
landmark tensor shape
```

For the final model, valid model input is usually:

```text
(64, 252)
```

This means 64 frames and 252 features: 126 hand-landmark features plus 126 motion-delta features.

