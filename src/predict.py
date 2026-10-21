from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.joblib"


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score churn risk for a customer CSV.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(args.input_csv)
    probabilities = model.predict_proba(df)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    out = df.copy()
    out["churn_probability"] = probabilities.round(4)
    out["predicted_churn"] = pd.Series(predictions).map({0: "No", 1: "Yes"})
    out["risk_band"] = [risk_band(p) for p in probabilities]
    out.to_csv(args.output, index=False)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
