from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_model_scores_dataset():
    model = joblib.load(ROOT / "models" / "churn_model.joblib")
    df = pd.read_csv(ROOT / "data" / "customer_churn.csv").head(10)
    features = df.drop(columns=["customer_id", "churn"])
    probs = model.predict_proba(features)[:, 1]
    assert len(probs) == 10
    assert ((probs >= 0) & (probs <= 1)).all()
