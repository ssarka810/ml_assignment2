import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Title
st.title("Bank Marketing Classification App")

st.write("Upload a CSV file and select a model to make predictions.")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Load models
models = {
    "Logistic Regression": joblib.load("model/logistic.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

# File uploader
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, sep=';')

    st.write("Uploaded Data Preview:")
    st.dataframe(df.head())

    # Encode target if present
    if 'y' in df.columns:
        df['y'] = df['y'].map({'yes': 1, 'no': 0})
        y_true = df['y']
        X = df.drop('y', axis=1)
    else:
        y_true = None
        X = df

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Align columns with training data
    training_columns = joblib.load("model/columns.pkl")
    X = X.reindex(columns=training_columns, fill_value=0)

    # Model selection
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]

    # Scaling for specific models
    if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_processed = scaler.transform(X)
    else:
        X_processed = X

    # Predict
    predictions = model.predict(X_processed)

    st.write("Predictions:")
    st.write(predictions)

    if y_true is not None:
        st.subheader("Classification Report")
        report = classification_report(y_true, predictions, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, predictions)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)
