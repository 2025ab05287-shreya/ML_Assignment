import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

st.set_page_config(page_title="Adult Income Classification", layout="wide")

st.title("Adult Income Classification - ML Models")
st.write("Upload test dataset and select a model for evaluation.")

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic.pkl"),
    "Decision Tree": joblib.load("models/decision_tree.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes.pkl"),
    "Random Forest": joblib.load("models/random_forest.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

scaler = joblib.load("models/scaler.pkl")
trained_columns = joblib.load("models/columns.pkl")

model_name = st.selectbox("Select Model", list(models.keys()))
model = models[model_name]

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.write(data.head())

    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)

    data["income"] = data["income"].map({
        "<=50K": 0,
        ">50K": 1
    })

    X = data.drop("income", axis=1)
    y = data["income"]

    X = pd.get_dummies(X, drop_first=True)

    for col in trained_columns:
        if col not in X:
            X[col] = 0

    X = X[trained_columns]

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = "Not Available"

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "AUC", "MCC"],
        "Value": [accuracy, precision, recall, f1, auc, mcc]
    })

    st.table(metrics)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)
