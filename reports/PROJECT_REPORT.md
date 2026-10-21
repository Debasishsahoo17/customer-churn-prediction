# Customer Churn Prediction — Project Report

## Executive summary

This project builds an end-to-end machine learning workflow for predicting telecom customer churn. The included dataset contains **6,000 customers** with an observed churn rate of **21.7%**. Three classification models were compared and **Logistic Regression** produced the highest test ROC-AUC of **0.873**.

The selected model reaches **82.3% recall** on churners, meaning it identifies most customers who actually churn in the held-out test sample. This makes the model suitable as a retention-prioritization tool, where missing a genuinely high-risk customer may be more costly than contacting some false positives.

## Business problem

Customer churn directly reduces recurring revenue and increases acquisition pressure. The goal is to estimate each customer's probability of churn so a retention team can:

1. Rank customers by churn risk.
2. Focus proactive outreach on high-risk accounts.
3. Identify common churn drivers.
4. Measure which customer segments need retention interventions.

## Dataset

The project ships with a synthetic but realistic telecom dataset so it is fully reproducible without external downloads. It contains customer demographics, service usage, contract information, billing data, support activity, satisfaction, and a binary churn outcome.

- Rows: **6,000**
- Features used for modeling: **22**
- Target: `churn`
- Churn rate: **21.7%**
- Train/test split: **80/20**, stratified by churn

## Model comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.786 | 0.504 | 0.823 | 0.625 | 0.873 |
| Gradient Boosting | 0.822 | 0.660 | 0.365 | 0.470 | 0.867 |
| Random Forest | 0.823 | 0.602 | 0.535 | 0.566 | 0.859 |


## Selected model

**Logistic Regression** was selected because it achieved the best ROC-AUC on unseen test data. The model uses class balancing so that churners are not overwhelmed by the larger non-churn class.

### Test performance

- Accuracy: **78.6%**
- Precision (churn): **50.4%**
- Recall (churn): **82.3%**
- F1 score (churn): **0.625**
- ROC-AUC: **0.873**

Confusion matrix on the 1,200-customer test set:

- True negatives: **729**
- False positives: **211**
- False negatives: **46**
- True positives: **214**

## Major churn drivers

The strongest model signals are listed below. For logistic regression, the report uses absolute coefficient magnitude as feature importance.

1. `contract_type_Month-to-month` — importance 1.164
2. `contract_type_Two year` — importance 0.859
3. `satisfaction_score` — importance 0.662
4. `tenure_months` — importance 0.611
5. `num_support_tickets` — importance 0.603
6. `monthly_charges` — importance 0.582
7. `online_security_Yes` — importance 0.485
8. `contract_type_One year` — importance 0.482
9. `tech_support_Yes` — importance 0.390
10. `payment_method_Electronic check` — importance 0.362


Business interpretation: short or month-to-month contracts, low satisfaction, low tenure, frequent support issues, higher charges, missing support/security services, and electronic-check payment behavior are useful signals for churn-risk prioritization in this synthetic scenario.

## Recommended retention workflow

A business could score active customers weekly and divide them into three operational bands:

- **High risk (70%+)**: proactive retention call, service review, targeted offer.
- **Medium risk (40–69%)**: personalized email/in-app offer or satisfaction survey.
- **Low risk (<40%)**: regular engagement and monitoring.

The probability threshold should ultimately be tuned using the real financial cost of retention offers, false positives, and lost customers.

## Deliverables

- Reproducible dataset generator
- Data preprocessing pipeline
- Three-model comparison
- Saved trained model
- Feature-importance report
- Evaluation charts
- Command-line batch scoring
- Streamlit prediction dashboard
- Notebook walkthrough
- Automated smoke test

## Limitations

The bundled dataset is synthetic, so the reported performance demonstrates the workflow rather than real-world business performance. Before production use, retrain on real customer history, validate on an out-of-time test set, monitor data drift, evaluate fairness, and calibrate intervention thresholds against retention economics.
