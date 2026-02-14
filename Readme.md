# Bank Marketing Classification – Machine Learning Assignment 2

## 1. Problem Statement

The objective of this project is to build and compare multiple classification models to predict whether a client will subscribe to a term deposit based on demographic information and previous marketing campaign data.

The target variable is:
- `y` → Indicates whether the client subscribed to a term deposit (yes/no)

This is a binary classification problem.

---

## 2. Dataset Description

- Dataset Name: Bank Marketing Dataset
- Source: UCI Machine Learning Repository / Kaggle
- Total Instances: 41,188
- Total Features: 20 input features + 1 target variable
- Type: Binary Classification
- Data Type: Mix of categorical and numerical features

The dataset contains information such as:
- Age
- Job
- Marital status
- Education
- Loan details
- Contact communication type
- Campaign-related information
- Economic indicators

The goal is to predict whether a client will subscribe to a term deposit.

---

## 3. Models Implemented

The following six classification models were implemented and evaluated:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (GaussianNB)  
5. Random Forest (Ensemble Model)  
6. XGBoost (Ensemble Model)  

All models were trained and evaluated using the same dataset and preprocessing steps.

---

## 4. Evaluation Metrics

The following evaluation metrics were calculated for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---

## 5. Model Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.9163 | 0.9424 | 0.7100 | 0.4353 | 0.5397 | 0.5147 |
| Decision Tree | 0.8928 | 0.7416 | 0.5232 | 0.5463 | 0.5345 | 0.4741 |
| KNN | 0.9022 | 0.8322 | 0.6178 | 0.3448 | 0.4426 | 0.4138 |
| Naive Bayes | 0.7527 | 0.8493 | 0.2902 | 0.8265 | 0.4296 | 0.3860 |
| Random Forest | 0.9153 | 0.9453 | 0.6701 | 0.4881 | 0.5648 | 0.5272 |
| XGBoost | 0.9226 | 0.9546 | 0.6923 | 0.5625 | 0.6207 | 0.5819 |

---

## 6. Observations

### Logistic Regression
Logistic Regression achieved high accuracy and strong AUC. It showed high precision but relatively lower recall, meaning it was conservative in predicting positive cases. Overall, it provided stable and reliable performance.

### Decision Tree
Decision Tree achieved moderate performance. It had higher recall compared to Logistic Regression but lower AUC and precision, indicating more false positive predictions.

### K-Nearest Neighbors (KNN)
KNN showed moderate accuracy but lower recall and MCC. Due to high dimensionality after one-hot encoding, its performance was lower compared to ensemble methods.

### Naive Bayes
Naive Bayes achieved very high recall but very low precision. It predicted many positive cases, resulting in more false positives and lower overall accuracy. This behavior is typical due to its independence assumptions.

### Random Forest
Random Forest improved recall and F1 score compared to Logistic Regression. It showed better balance between precision and recall and achieved strong AUC, making it one of the top-performing models.

### XGBoost
XGBoost achieved the highest Accuracy, AUC, F1 Score, and MCC among all models. It demonstrated the best overall performance and provided balanced predictions. Therefore, XGBoost is selected as the best-performing model for this dataset.

---

## 7. Streamlit Web Application

An interactive Streamlit web application was developed with the following features:

- CSV dataset upload option
- Model selection dropdown
- Prediction generation
- Display of classification report
- Confusion matrix visualization

The application allows users to test different models on uploaded data and compare performance interactively.

---

## 8. Project Structure

assignment2/
│-- app.py
│-- requirements.txt
│-- README.md
│-- bank_marketing_models.ipynb
│-- model/
│ ├── logistic.pkl
│ ├── decision_tree.pkl
│ ├── knn.pkl
│ ├── naive_bayes.pkl
│ ├── random_forest.pkl
│ ├── xgboost.pkl
│ ├── scaler.pkl
│ ├── columns.pkl


---

## 9. Deployment

The application is deployed on Streamlit Community Cloud.

Live App Link: (Add your deployed Streamlit link here)

GitHub Repository Link: (Add your GitHub repo link here)