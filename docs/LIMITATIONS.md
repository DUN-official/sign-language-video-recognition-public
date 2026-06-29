# Limitations

This repository is a working prototype, not a production ASL translator.

## Dataset Limitations

- WLASL video links are not all available. Some are private, removed, expired, or corrupted.
- After download and landmark extraction, many classes have very few usable samples.
- Some labels appear noisy or inconsistent.
- Some clips include extra movement, setup motion, repeated signs, or multiple signs while being labelled as one class.
- Test sets can be very small, so per-class accuracy can swing sharply from one example.

## Video Quality Limitations

- Videos vary in resolution, aspect ratio, lighting, compression, and framing.
- Signers appear at different distances from the camera.
- Some hands leave the frame or become occluded.
- Some videos are short or contain idle frames.

## Feature Limitations

The current model uses hand landmarks. This is useful, but incomplete for ASL.

Some signs depend on:

```text
face expression
body/torso pose
arm trajectory
hand position relative to face/chest
two-hand interaction
motion speed and direction
```

Hand landmarks alone can miss important context.

## Model Limitations

- The model can be overconfident when wrong.
- Visually similar signs can be confused.
- Classes with fewer samples are harder to learn.
- The final selected model is reliable mainly for curated trained-class demonstrations.

## Honest Interpretation

The strongest contribution of this project is the full pipeline:

```text
dataset preparation -> download verification -> landmark extraction -> quality filtering -> temporal modeling -> Streamlit demo
```

The current final model demonstrates that the pipeline works, while also showing why stronger data and richer features are needed for robust ASL recognition.

