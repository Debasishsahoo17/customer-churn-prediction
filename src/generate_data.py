from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_customer_churn_data(n_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic telecom churn dataset.

    The target is probabilistically generated from business-like churn drivers,
    so the project remains fully runnable without downloading external data.
    """
    rng = np.random.default_rng(seed)

    tenure = np.clip(rng.gamma(shape=2.2, scale=15, size=n_rows), 0, 72).round().astype(int)
    gender = rng.choice(["Female", "Male"], n_rows)
    senior = rng.choice(["No", "Yes"], n_rows, p=[0.84, 0.16])
    partner = rng.choice(["No", "Yes"], n_rows, p=[0.52, 0.48])
    dependents = np.where(
        partner == "Yes",
        rng.choice(["No", "Yes"], n_rows, p=[0.58, 0.42]),
        rng.choice(["No", "Yes"], n_rows, p=[0.86, 0.14]),
    )

    phone_service = rng.choice(["No", "Yes"], n_rows, p=[0.08, 0.92])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        rng.choice(["No", "Yes"], n_rows, p=[0.52, 0.48]),
    )

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.48, 0.18])

    def internet_addon(yes_prob: float) -> np.ndarray:
        values = rng.choice(["No", "Yes"], n_rows, p=[1 - yes_prob, yes_prob])
        return np.where(internet_service == "No", "No internet service", values)

    online_security = internet_addon(0.36)
    online_backup = internet_addon(0.42)
    device_protection = internet_addon(0.41)
    tech_support = internet_addon(0.34)
    streaming_tv = internet_addon(0.50)
    streaming_movies = internet_addon(0.49)

    # Longer-tenure customers are more likely to move to longer contracts.
    month_prob = np.clip(0.78 - tenure / 140, 0.30, 0.78)
    one_year_prob = np.clip(0.13 + tenure / 260, 0.13, 0.34)
    contract = []
    for i in range(n_rows):
        p_month = month_prob[i]
        p_one = one_year_prob[i]
        p_two = max(0.02, 1 - p_month - p_one)
        probs = np.array([p_month, p_one, p_two])
        probs = probs / probs.sum()
        contract.append(rng.choice(["Month-to-month", "One year", "Two year"], p=probs))
    contract = np.array(contract)

    paperless = rng.choice(["No", "Yes"], n_rows, p=[0.39, 0.61])
    payment_method = rng.choice(
        ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
        n_rows,
        p=[0.22, 0.23, 0.35, 0.20],
    )

    base_charge = np.where(internet_service == "Fiber optic", 68, np.where(internet_service == "DSL", 45, 20))
    base_charge += np.where(phone_service == "Yes", 12, 0)
    addon_count = sum(
        (arr == "Yes").astype(int)
        for arr in [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
    )
    monthly_charges = np.clip(base_charge + addon_count * 5.2 + rng.normal(0, 7, n_rows), 18, 125).round(2)

    total_charges = (monthly_charges * np.maximum(tenure, 1) * rng.normal(1.0, 0.035, n_rows)).clip(18).round(2)
    avg_monthly_gb = np.where(
        internet_service == "No",
        0,
        np.clip(rng.gamma(3.0, 28, n_rows) + np.where(internet_service == "Fiber optic", 45, 0), 2, 500),
    ).round(1)

    # Satisfaction and support tickets are strong but imperfect operational signals.
    satisfaction = np.clip(np.rint(rng.normal(3.7, 0.95, n_rows)), 1, 5).astype(int)
    support_lambda = np.clip(1.2 + (4 - satisfaction) * 0.65 + (internet_service == "Fiber optic") * 0.35, 0.2, 5.0)
    support_tickets = np.clip(rng.poisson(support_lambda), 0, 12)

    # Churn propensity: coefficients chosen to mimic common telecom retention patterns.
    logit = (
        -1.75
        + 1.15 * (contract == "Month-to-month")
        - 0.55 * (contract == "One year")
        - 1.25 * (contract == "Two year")
        + 0.65 * (internet_service == "Fiber optic")
        + 0.48 * (payment_method == "Electronic check")
        + 0.32 * (paperless == "Yes")
        + 0.40 * (senior == "Yes")
        - 0.42 * (partner == "Yes")
        - 0.34 * (dependents == "Yes")
        - 0.75 * (online_security == "Yes")
        - 0.68 * (tech_support == "Yes")
        + 0.030 * (monthly_charges - 65)
        - 0.035 * tenure
        + 0.42 * support_tickets
        - 0.72 * (satisfaction - 3)
        + rng.normal(0, 0.55, n_rows)
    )
    churn_prob = sigmoid(logit)
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(1, n_rows + 1)],
            "gender": gender,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "tenure_months": tenure,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "contract_type": contract,
            "paperless_billing": paperless,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "num_support_tickets": support_tickets,
            "avg_monthly_gb": avg_monthly_gb,
            "satisfaction_score": satisfaction,
            "churn": churn,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/customer_churn.csv"))
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customer_churn_data(args.rows, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")
    print(f"Churn rate: {(df['churn'].eq('Yes').mean() * 100):.1f}%")


if __name__ == "__main__":
    main()
