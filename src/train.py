from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "customer_churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

TARGET = "churn"
ID_COL = "customer_id"
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "avg_monthly_gb",
    "satisfaction_score",
]
CATEGORICAL_FEATURES = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "phone_service",
    "multiple_lines",
    "internet_service",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "contract_type",
    "paperless_billing",
    "payment_method",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def evaluate(name: str, pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    pred = pipeline.predict(x_test)
    prob = pipeline.predict_proba(x_test)[:, 1]
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob),
    }


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    out = pd.DataFrame({"feature": feature_names, "importance": values})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


def save_eda(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    churn_rate = df[TARGET].value_counts(normalize=True).reindex(["No", "Yes"]) * 100
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(churn_rate.index, churn_rate.values)
    ax.set_title("Customer Churn Distribution")
    ax.set_ylabel("Percent of customers")
    ax.set_ylim(0, 100)
    for i, v in enumerate(churn_rate.values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_distribution.png", dpi=160)
    plt.close(fig)

    contract_churn = (
        df.assign(churn_flag=df[TARGET].eq("Yes").astype(int))
        .groupby("contract_type", observed=False)["churn_flag"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(contract_churn.index, contract_churn.values)
    ax.set_title("Churn Rate by Contract Type")
    ax.set_ylabel("Churn rate (%)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    sat_churn = (
        df.assign(churn_flag=df[TARGET].eq("Yes").astype(int))
        .groupby("satisfaction_score", observed=False)["churn_flag"]
        .mean()
        * 100
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sat_churn.index, sat_churn.values, marker="o")
    ax.set_title("Churn Rate by Satisfaction Score")
    ax.set_xlabel("Satisfaction score")
    ax.set_ylabel("Churn rate (%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    fig.tight_layout()
    fig.savefig(FIG_DIR / "churn_by_satisfaction.png", dpi=160)
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}. Run src/generate_data.py first.")

    df = pd.read_csv(DATA_PATH)
    save_eda(df)

    x = df[FEATURES].copy()
    y = df[TARGET].map({"No": 0, "Yes": 1})
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.045,
            max_depth=2,
            random_state=42,
        ),
    }

    results = []
    fitted = {}
    for name, model in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", model)])
        pipe.fit(x_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, x_test, y_test))

    comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.loc[0, "model"]
    best = fitted[best_name]
    joblib.dump(best, MODEL_DIR / "churn_model.joblib")

    importance = extract_feature_importance(best)
    importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)

    # ROC curve
    best_prob = best.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, best_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{best_name} (AUC={roc_auc_score(y_test, best_prob):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)

    # Confusion matrix
    best_pred = best.predict(x_test)
    cm = confusion_matrix(y_test, best_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No churn", "Churn"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center", color="white" if val > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    # Top features plot
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance"])
    ax.set_title(f"Top Churn Drivers — {best_name}")
    ax.set_xlabel("Feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "top_features.png", dpi=160)
    plt.close(fig)

    best_metrics = comparison.iloc[0].to_dict()
    best_metrics["dataset_rows"] = int(len(df))
    best_metrics["churn_rate"] = float(y.mean())
    best_metrics["train_rows"] = int(len(x_train))
    best_metrics["test_rows"] = int(len(x_test))
    best_metrics["confusion_matrix"] = cm.tolist()
    best_metrics["classification_report"] = classification_report(y_test, best_pred, output_dict=True)

    with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)

    schema = {
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "best_model": best_name,
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("\nModel comparison:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved best model: {best_name}")
    print(f"ROC-AUC: {best_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
