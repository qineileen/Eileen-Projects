# Gradient Boosting Model — Home Credit Default Risk
This repository implements a supervised machine learning pipeline for predicting loan default risk using the Home Credit Default Risk dataset from Kaggle. The task is a binary classification problem, and the primary evaluation metric is AUC.

## Project Overview
This project trains models on real-world tabular data and covers the full machine learning workflow, including:

- Feature engineering
- Missing value handling
- Categorical encoding
- Model validation
- Gradient boosting models
- Model interpretability

Gradient boosting models (LightGBM, XGBoost, CatBoost) are emphasized due to their strong performance on structured data.

## Dataset
Kaggle dataset link:
https://www.kaggle.com/c/home-credit-default-risk/data

* The data is not uploaded to github because the size over limit (100M)

Main files include:

application_train.csv

application_test.csv

bureau.csv

previous_application.csv

installments_payments.csv

credit_card_balance.csv

POS_CASH_balance.csv

Repository Structure
text
gradient_boosting_project/
├── data/
├── notebooks/
├── src/
├── output/
├── reports/
├── requirements.txt
└── README.md

## Methodology Summary
- Load and inspect application_train.csv
- Perform EDA: target imbalance, missing values, distributions
- Preprocess data: missing values, encoding, leakage removal
- Train baseline model (Logistic Regression or LightGBM)
- Improve model with cross-validation, feature engineering, hyperparameter tuning
- Interpret model using feature importance and SHAP

## Expected Outputs
- Missing value summary
- Target distribution plot
- Correlation heatmap
- Baseline and cross-validation AUC
- ROC and PR curves
- Confusion matrix
- Feature importance and SHAP plots
- Model comparison table

