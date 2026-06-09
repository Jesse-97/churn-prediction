import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.preprocess import load_and_clean, encode_features, split_and_balance


def generate_shap_plots(data_path='data/churn.csv'):
    # Loading model
    with open('models/xgb_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocessing data
    df = load_and_clean(data_path)
    df = encode_features(df)
    _, X_test, _, _ = split_and_balance(df)

    print("Generating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    os.makedirs('plots', exist_ok=True)

    # Plot 1 — Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved plots/shap_summary.png")

    # Plot 2 — Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_bar.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("Saved plots/shap_bar.png")


if __name__ == '__main__':
    generate_shap_plots()