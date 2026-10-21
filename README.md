# Customer Churn Prediction

An end-to-end **data analytics + machine learning** project that predicts which telecom customers are likely to churn and turns the model output into business-friendly retention insights.

## What this project includes

- Realistic telecom customer dataset (6,000 records)
- Exploratory data analysis (EDA)
- Automated preprocessing for numerical and categorical features
- Model comparison: Logistic Regression, Random Forest, Gradient Boosting
- Evaluation with Accuracy, Precision, Recall, F1 and ROC-AUC
- Feature-importance analysis to identify churn drivers
- Single-customer churn probability scoring
- Batch CSV scoring
- Interactive Streamlit dashboard
- Saved production-ready scikit-learn pipeline
- Smoke test with pytest

> The included data is synthetic and intentionally generated inside the repository so the project runs immediately without requiring an external download. The same pipeline can be adapted to the IBM/Kaggle Telco Customer Churn dataset by aligning column names.

## Project structure

```text
customer-churn-prediction/
├── app.py
├── data/
│   └── customer_churn.csv
├── models/
│   ├── churn_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── reports/
│   ├── feature_importance.csv
│   ├── metrics.json
│   ├── model_comparison.csv
│   ├── PROJECT_REPORT.md
│   └── figures/
├── src/
│   ├── generate_data.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_smoke.py
├── requirements.txt
└── README.md
```

## Run locally

### 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```

### 2. Regenerate the dataset (optional)

```bash
python src/generate_data.py
```

### 3. Train and evaluate models

```bash
python src/train.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

### 5. Run tests

```bash
pytest -q
```

## Batch prediction

Prepare a CSV with the model input columns and run:

```bash
python src/predict.py your_customers.csv --output predictions.csv
```

The output contains:

- `churn_probability`
- `predicted_churn`
- `risk_band` (Low / Medium / High)

## Business use case

A telecom company can use the model to rank customers by churn probability and focus retention resources on the highest-risk accounts. The analytical layer also highlights customer characteristics associated with churn, such as contract type, satisfaction, support activity and tenure.

## Tech stack

Python, Pandas, NumPy, scikit-learn, Matplotlib, Streamlit, Joblib, Pytest.
