# 📉 Customer Churn Prediction Dashboard

A machine learning web app that predicts whether a telecom customer is likely to churn, built with XGBoost and deployed with Streamlit + FastAPI.

**🔗 Live Demo:** [https://churn-prediction-6iyp6zbysqzo6yju8biqdk.streamlit.app/](https://churn-prediction-6iyp6zbysqzo6yju8biqdk.streamlit.app/)

---

## What it does

Given a telecom customer's details — contract type, tenure, monthly charges, services subscribed — the app predicts the likelihood of that customer leaving. It also explains *why* the model made its prediction using SHAP values, and supports batch prediction via CSV upload.

---

## Features

- **Single Prediction** — fill in customer details and get an instant churn probability with risk level (Low / Medium / High)
- **Model Insights** — SHAP summary and feature importance plots showing which factors drive churn the most
- **Batch Prediction** — upload a CSV of multiple customers and download predictions as a CSV

---

## Model Performance

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.83 |
| F1 Score (churn class) | 0.59 |
| Accuracy | 77% |

> Trained on the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,032 customers, 20 features.

The dataset is imbalanced (73% stayed, 27% churned). SMOTE was applied to the training set to balance classes before training.

---

## Key Finding

**Month-to-month contract type** was the strongest predictor of churn, followed by tenure and monthly charges. Customers with short tenure, high monthly charges, fiber optic internet, and no security services were flagged as highest risk — consistent with real-world telecom churn patterns.

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, scikit-learn |
| Modelling | XGBoost, imbalanced-learn (SMOTE) |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit, Plotly |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
churn-prediction/
├── data/
│   └── churn.csv                 # Telco dataset
├── models/
│   ├── xgb_model.pkl             # Trained XGBoost model
│   └── feature_names.pkl         # Feature column names for inference
├── notebooks/
│   └── exploration.ipynb         # EDA notebook
├── plots/
│   ├── shap_summary.png          # SHAP dot plot
│   └── shap_bar.png              # Feature importance bar chart
├── src/
│   ├── preprocess.py             # Data cleaning, encoding, SMOTE
│   ├── model.py                  # XGBoost training and evaluation
│   └── explain.py                # SHAP value generation
├── app.py                        # Streamlit dashboard
├── api.py                        # FastAPI REST endpoint
└── requirements.txt
```

---

## API Usage

The FastAPI endpoint accepts a POST request with customer data and returns a churn prediction.

**Endpoint:** `POST /predict`

**Example request:**
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 151.65
}
```

**Example response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.8868,
  "risk_level": "High"
}
```

---

## Limitations

- SMOTE inflates cross-validation scores — the honest test set F1 is 0.59, not the CV score of 0.83
- Model was trained on a US telecom dataset and may not generalise to other markets without retraining
- A fine-tuned transformer or stacked ensemble would likely improve F1 on the minority class

