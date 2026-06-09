import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_clean(path):
    df = pd.read_csv(path)
    
    # Dropping customerID
    df.drop(columns=['customerID'], inplace=True)

    # Fixing totalcharges 
    df = df[df['TotalCharges'].str.strip() != '']
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

    # Converting churn to binary value
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    return df

def encode_features(df):
    df = df.copy()

    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]

    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    multi_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=multi_cols)

    return df

def split_and_balance(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    #Applying SMOTE to balance the training data
    print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {dict(zip(['No','Yes'], [sum(y_train==0), sum(y_train==1)]))}")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    df = load_and_clean('data/churn.csv')
    print(f"Cleaned shape: {df.shape}")
    df = encode_features(df)
    print(f"Encoded shape: {df.shape}")
    X_train, X_test, y_train, y_test = split_and_balance(df)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")