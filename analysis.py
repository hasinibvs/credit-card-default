import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

DATA_PATH = 'data.xls'
TARGET_COL = 'default payment next month'

def load_data(path):
    """Loads data from Excel or CSV."""
    print(f"Loading data from {path}...")
    try:
        if path.endswith('.xls') or path.endswith('.xlsx'):
            df = pd.read_excel(path, header=1)
        else:
            df = pd.read_csv(path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: File not found. Please download the UCI dataset.")
        return None

def preprocess_data(df):
    """Cleans and splits the data."""
    print("Preprocessing data...")
    
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    X = df.drop(TARGET_COL, axis=1)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

def handle_imbalance(X_train, y_train):
    """
    Applies SMOTE to handle class imbalance.
    This creates synthetic samples of the minority class (defaulters).
    """
    print("Handling class imbalance using SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"Original dataset shape: {y_train.value_counts().to_dict()}")
    print(f"Resampled dataset shape: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled

def train_model(X_train, y_train):
    """Trains a Random Forest Classifier."""
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    return rf

def evaluate_model(model, X_test, y_test):
    """Generates evaluation metrics."""
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png')
    print("Confusion Matrix saved as 'confusion_matrix.png'")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    roc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_score:.4f}")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    
    if df is not None:
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
        
        X_train_smote, y_train_smote = handle_imbalance(X_train, y_train)
        
        model = train_model(X_train_smote, y_train_smote)
        
        evaluate_model(model, X_test, y_test)
        
        print("\nAnalysis Complete.")