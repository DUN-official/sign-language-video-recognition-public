import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Analyze prediction CSV results.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="results/analysis")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)

    required = ["true_gloss", "pred_gloss", "correct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["correct"] = df["correct"].astype(bool)

    overall = {
        "samples": len(df),
        "top1_accuracy": float(df["correct"].mean()) if len(df) else 0.0,
        "classes_true": int(df["true_gloss"].nunique()),
        "classes_predicted": int(df["pred_gloss"].nunique()),
    }

    per_class = (
        df.groupby("true_gloss")
        .agg(
            samples=("correct", "size"),
            correct=("correct", "sum"),
            top1_accuracy=("correct", "mean"),
        )
        .reset_index()
        .sort_values(["top1_accuracy", "samples"], ascending=[True, False])
    )

    confusion_pairs = (
        df[df["correct"] == False]
        .groupby(["true_gloss", "pred_gloss"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    prediction_bias = (
        df.groupby("pred_gloss")
        .size()
        .reset_index(name="times_predicted")
        .sort_values("times_predicted", ascending=False)
    )

    pd.DataFrame([overall]).to_csv(output_dir / "overall_summary.csv", index=False)
    per_class.to_csv(output_dir / "per_class_accuracy.csv", index=False)
    confusion_pairs.to_csv(output_dir / "top_confusions.csv", index=False)
    prediction_bias.to_csv(output_dir / "prediction_bias.csv", index=False)

    print("Overall:")
    print(pd.DataFrame([overall]).to_string(index=False))
    print("\nWorst classes:")
    print(per_class.head(15).to_string(index=False))
    print("\nMost common wrong prediction pairs:")
    print(confusion_pairs.head(15).to_string(index=False))
    print("\nMost frequently predicted classes:")
    print(prediction_bias.head(15).to_string(index=False))
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
