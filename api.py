import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

app = FastAPI(
    title="Churn Prediction API",
    description="Predicts whether a telecom customer will churn",
    version="1.0"
)

# Input schema for API
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: CustomerData):
    # Convert input to dataframe
    input_df = pd.DataFrame([customer.dict()])

    # Apply same encoding as training
    from sklearn.preprocessing import LabelEncoder
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    le = LabelEncoder()
    for col in binary_cols:
        input_df[col] = le.fit_transform(input_df[col])

    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=multi_cols)

    
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Predicting
    prob = model.predict_proba(input_df)[0][1]
    prediction = int(prob >= 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(float(prob), 4),
        "risk_level": "High" if prob >= 0.7 else "Medium" if prob >= 0.4 else "Low"
    }