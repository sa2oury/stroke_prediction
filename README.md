🩺 Heart Stroke Prediction Web App

A Machine Learning–powered Streamlit web application that predicts the likelihood of a person having a heart stroke based on health-related data.

This project demonstrates how data preprocessing, feature engineering, and model comparison can be used to support early stroke risk prediction in healthcare analytics.

🚀 Features

✅ Interactive Web Interface — built with Streamlit for easy, no-code predictions.
✅ Dropdown Inputs — users select from predefined options (no manual typing).
✅ Automatic Data Cleaning — irrelevant columns like id are removed.
✅ Model Training & Evaluation — trains and compares RandomForest & XGBoost models.
✅ Performance Metrics — displays accuracy, precision, recall, and F1-score.
✅ Instant Predictions — outputs whether a person is at risk of heart stroke (0 or 1).

📊 Model Performance
Model	Accuracy	Precision	Recall	F1 Score
RandomForest	0.951	0.500	0.020	0.038
XGBoost	0.939	0.227	0.100	0.139

Note: The dataset is imbalanced — future improvements will focus on increasing recall using resampling techniques (e.g. SMOTE).

🧠 Technologies Used

Python 3.9+

Streamlit

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib
