import setup
import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image
from src.preprocess import load_and_clean, encode_features
from sklearn.preprocessing import LabelEncoder

# Loading model
with open('models/xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

st.set_page_config(
    page_title='Churn Prediction Dashboard',
    page_icon='📉',
    layout='wide'
)

st.title('Customer Churn Prediction Dashboard')
st.markdown('Predict whether a telecom customer is likely to churn using XGBoost.')

# ── Tabs ──
tab1, tab2, tab3 = st.tabs(['🔍 Predict', '📊 Model Insights', '📁 Batch Predict'])

# ── Tab 1: Single Prediction ──
with tab1:
    st.subheader('Enter Customer Details')

    col1, col2, col3 = st.columns(3)

    with col1:
        gender         = st.selectbox('Gender', ['Male', 'Female'])
        senior         = st.selectbox('Senior Citizen', [0, 1])
        partner        = st.selectbox('Partner', ['Yes', 'No'])
        dependents     = st.selectbox('Dependents', ['Yes', 'No'])
        tenure         = st.slider('Tenure (months)', 0, 72, 12)
        phone          = st.selectbox('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['Yes', 'No', 'No phone service'])

    with col2:
        internet       = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        security       = st.selectbox('Online Security', ['Yes', 'No', 'No internet service'])
        backup         = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service'])
        device         = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service'])
        tech           = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service'])
        tv             = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service'])
        movies         = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service'])

    with col3:
        contract       = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless      = st.selectbox('Paperless Billing', ['Yes', 'No'])
        payment        = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
        monthly        = st.number_input('Monthly Charges ($)', 0.0, 200.0, 65.0)
        total          = st.number_input('Total Charges ($)', 0.0, 10000.0, 1000.0)

    if st.button('Predict Churn', type='primary'):
        input_dict = {
            'gender': gender, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multiple_lines, 'InternetService': internet,
            'OnlineSecurity': security, 'OnlineBackup': backup,
            'DeviceProtection': device, 'TechSupport': tech,
            'StreamingTV': tv, 'StreamingMovies': movies,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment, 'MonthlyCharges': monthly,
            'TotalCharges': total
        }

        input_df = pd.DataFrame([input_dict])

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

        prob = model.predict_proba(input_df)[0][1]
        prediction = int(prob >= 0.5)
        risk = "🔴 High Risk" if prob >= 0.7 else "🟡 Medium Risk" if prob >= 0.4 else "🟢 Low Risk"

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric('Churn Prediction', 'Will Churn' if prediction == 1 else 'Will Stay')
        c2.metric('Churn Probability', f"{prob:.1%}")
        c3.metric('Risk Level', risk)

        fig = px.bar(
            x=['Stay', 'Churn'],
            y=[1 - prob, prob],
            color=['Stay', 'Churn'],
            color_discrete_map={'Stay': '#2ECC71', 'Churn': '#E74C3C'},
            title='Prediction Probability'
        )
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Model Insights ──
with tab2:
    st.subheader('SHAP Feature Importance')
    st.markdown('These plots show which features drive churn predictions the most.')

    col_a, col_b = st.columns(2)
    with col_a:
        st.image('plots/shap_summary.png', caption='SHAP Summary Plot', use_container_width=True)
    with col_b:
        st.image('plots/shap_bar.png', caption='Feature Importance (Bar)', use_container_width=True)

# ── Tab 3: Batch Predict ──
with tab3:
    st.subheader('Batch Prediction — Upload CSV')
    st.markdown('Upload a CSV file with the same columns as the training data.')

    uploaded = st.file_uploader('Upload CSV', type=['csv'])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded {len(batch_df)} rows")

        if 'customerID' in batch_df.columns:
            ids = batch_df['customerID']
            batch_df.drop(columns=['customerID'], inplace=True)
        else:
            ids = pd.Series(range(len(batch_df)))

        le = LabelEncoder()
        binary_cols = [
            'gender', 'Partner', 'Dependents', 'PhoneService',
            'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies'
        ]
        for col in binary_cols:
            if col in batch_df.columns:
                batch_df[col] = le.fit_transform(batch_df[col])

        multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
        batch_df = pd.get_dummies(batch_df, columns=multi_cols)
        batch_df = batch_df.reindex(columns=feature_names, fill_value=0)

        probs = model.predict_proba(batch_df)[:, 1]
        results = pd.DataFrame({
            'CustomerID': ids,
            'Churn Probability': probs.round(4),
            'Prediction': ['Churn' if p >= 0.5 else 'Stay' for p in probs],
            'Risk Level': ['High' if p >= 0.7 else 'Medium' if p >= 0.4 else 'Low' for p in probs]
        })

        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False)
        st.download_button('Download Results', csv, 'churn_predictions.csv', 'text/csv')