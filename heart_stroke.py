import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
)

# Try importing XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Stroke Prediction App", layout="wide")

# ----------------------------- Load Dataset -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    # Ensure gender only contains Male/Female
    if 'gender' in df.columns:
        df = df[df['gender'].isin(['Male', 'Female'])]
    return df

df = load_data()

# ----------------------------- Helper Functions -----------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

    return preprocessor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return y_pred, acc, prec, rec, f1, fpr, tpr, roc_auc

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    st.pyplot(fig)

# ----------------------------- Tabs -----------------------------
tabs = st.tabs(["🏠 Home", "📊 Data Overview", "⚙️ Train Model", "📈 Evaluate", "🩺 Predict"])

# ----------------------------- Home -----------------------------
with tabs[0]:
    st.title("Stroke Prediction App 🩺")
    st.markdown("""
    This app predicts the likelihood of a person having a stroke based on health indicators.
    - Uses the healthcare stroke dataset.
    - Removes the ID column automatically.
    - Supports RandomForest and XGBoost models.
    - Predicts stroke (1) or no stroke (0).
    - Gender limited to Male/Female only.
    """)

# ----------------------------- Data Overview -----------------------------
with tabs[1]:
    st.header("Dataset Overview")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe(include='all'))

    st.subheader("Missing Values")
    st.write(df.isna().sum())

    st.subheader("Class Balance")
    st.bar_chart(df['stroke'].value_counts())

# ----------------------------- Train Model -----------------------------
with tabs[2]:
    st.header("Train Model")

    X = df.drop(columns=['stroke'])
    y = df['stroke']

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random Seed", value=42)

    preprocessor = build_preprocessor(X)

    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=int(random_state)))
    ])

    if XGB_AVAILABLE:
        xgb_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', xgb.XGBClassifier(eval_metric='logloss', random_state=int(random_state)))
        ])

    if st.button("Train Models"):
        with st.spinner('Training RandomForest model...'):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=int(random_state))
            rf_model.fit(X_train, y_train)
            st.session_state['rf_model'] = rf_model
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.success('RandomForest trained!')

        if XGB_AVAILABLE:
            with st.spinner('Training XGBoost model...'):
                xgb_model.fit(X_train, y_train)
                st.session_state['xgb_model'] = xgb_model
                st.success('XGBoost trained!')

# ----------------------------- Evaluate -----------------------------
with tabs[3]:
    st.header("Evaluate Models")

    if 'rf_model' not in st.session_state:
        st.info("Please train a model first.")
    else:
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']

        model_options = ['RandomForest'] + (['XGBoost'] if XGB_AVAILABLE else [])
        selected_model = st.selectbox("Select Model to Evaluate", model_options)

        model = st.session_state['rf_model'] if selected_model == 'RandomForest' else st.session_state['xgb_model']

        y_pred, acc, prec, rec, f1, fpr, tpr, roc_auc = evaluate_model(model, X_test, y_test)

        st.subheader(f"{selected_model} Performance Metrics")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(cm)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        st.pyplot(fig)

# ----------------------------- Predict -----------------------------
with tabs[4]:
    st.header("Make Predictions")

    if 'rf_model' not in st.session_state:
        st.info("Please train a model first.")
    else:
        model_choice = st.selectbox("Select Model for Prediction", ['RandomForest'] + (['XGBoost'] if XGB_AVAILABLE else []))
        model = st.session_state['rf_model'] if model_choice == 'RandomForest' else st.session_state['xgb_model']

        st.subheader("Single Patient Prediction")
        input_data = {}
        
        for col in df.drop(columns=['stroke']).columns:
            if df[col].dtype == 'object':
                if col == 'gender':
                    options = ['Male', 'Female']
                else:
                    options = df[col].dropna().unique().tolist()
                input_data[col] = st.selectbox(f"{col}", options)
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                mean_val = float(df[col].mean())
                input_data[col] = st.slider(f"{col}", min_val, max_val, mean_val)

        if st.button("Predict Stroke Risk"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.write(f"### Prediction: {'Stroke (1)' if prediction == 1 else 'No Stroke (0)'}")
            st.write(f"### Probability: {prob:.3f}")

        st.subheader("Batch Prediction (Upload CSV)")
        uploaded_file = st.file_uploader("Upload a CSV for batch predictions", type=['csv'])
        if uploaded_file:
            batch_df = pd.read_csv(uploaded_file)
            preds = model.predict(batch_df)
            probs = model.predict_proba(batch_df)[:, 1]
            batch_df['stroke_prediction'] = preds
            batch_df['stroke_probability'] = probs
            st.write(batch_df.head())

            st.download_button("Download Predictions", data=batch_df.to_csv(index=False).encode('utf-8'), file_name='stroke_predictions.csv')