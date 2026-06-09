import pandas as pd
import numpy as np
import pickle
import os
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score
)
from src.preprocess import load_and_clean, encode_features, split_and_balance


def train(data_path='data/churn.csv'):
    # Preprocess
    df = load_and_clean(data_path)
    df = encode_features(df)
    X_train, X_test, y_train, y_test = split_and_balance(df)

    # Training XGBoost
    print("\nTraining XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n── Evaluation Results ──")
    print(f"ROC-AUC Score:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score:       {f1_score(y_test, y_pred):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    os.makedirs('models', exist_ok=True)
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(X_train.columns.tolist(), f)

    print("\nModel saved to models/xgb_model.pkl")
    return model, X_test, y_test

if __name__ == '__main__':
    train()