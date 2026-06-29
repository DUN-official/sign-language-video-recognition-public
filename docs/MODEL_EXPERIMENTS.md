# Model Experiments

This project went through several model stages. Earlier experiments are retained for transparency, while the final selected demo model is stored separately under `models/final/`.

## Experiment Summary

| Stage | Model / Data | Result | Main Lesson |
|---|---|---:|---|
| Baseline | WLASL100 early temporal model | 6.2% top-1 | Full 100-class task was too sparse/noisy for the first model |
| V2 | WLASL100 transformer | 28.0% top-1 | Better temporal model and normalization improved broad performance |
| Quality20 | WLASL100 quality-controlled subset | 50.0% top-1 | Curated subsets improve learnability but still had weak classes |
| V3 broader | WLASL500-derived best50 | 28.8% top-1 | More classes increased difficulty; useful as a scalability benchmark |
| V3 final | WLASL500-derived best6 fine-tuned | 50.0% top-1, 100.0% top-5 | Final demo model works best on curated trained-class clips |

## Why The Final Model Uses Best6

The WLASL500 candidate pool was filtered by landmark quality, split coverage, and sample availability. Under stricter quality rules, only a small number of classes had enough usable train/validation/test examples.

The final best6 classes are:

```text
computer, drink, enjoy, language, sunday, visit
```

This is not presented as a full ASL translator. It is a quality-controlled demonstration of the end-to-end pipeline.

## Why Some Models Struggled

- Many classes had only a few usable samples after video download and landmark extraction.
- Some test classes had only one example, making per-class accuracy unstable.
- Videos varied in resolution, framing, background, and signer style.
- Some labels were noisy or visually ambiguous.
- Hand landmarks alone do not capture all information needed for ASL.
- Some signs require face/body/pose context or location relative to the head/chest.
- MediaPipe sometimes missed hands or produced sparse landmarks.

## Recommended Next Improvements

- Add pose/face/body landmarks with MediaPipe Holistic or Tasks.
- Improve video trimming to remove idle/start/end frames.
- Use a larger verified dataset with stronger per-class coverage.
- Train with signer-aware validation if metadata supports it.
- Add calibration or confidence thresholding.
- Try pretrained video models or sign-language-specific architectures.

