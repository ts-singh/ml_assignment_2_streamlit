import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.data import load_dataset
#Page Setup and color starts here
st.markdown(
    """
    <style>

    /* ===== Main Background ===== */
    .stApp {
        background: linear-gradient(135deg, #F4F7FB 0%, #E9F1FF 100%);
        font-family: "Segoe UI", sans-serif;
    }

    /* ===== Title Styling ===== */
    h1, h2, h3 {
        color: #1F3C88;
        font-weight: 700;
    }

    /* ===== Sidebar ===== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F3C88 0%, #2952A3 100%);
    }

    /* Sidebar headings */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: white !important;
    }

    /* ===== FIX: Dropdown visibility ===== */

    /* Selectbox main area */
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
        background: white;
        color: black !important;
        border-radius: 8px;
    }

    /* Selected value text */
    section[data-testid="stSidebar"] .stSelectbox span {
        color: black !important;
    }

    /* Dropdown menu list */
    div[role="listbox"] {
        background: white !important;
        color: black !important;
    }

    /* Dropdown options */
    div[role="option"] {
        color: black !important;
    }

    /* ===== Buttons ===== */
    .stButton>button {
        background: linear-gradient(90deg, #1F3C88, #4A90E2);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }

    /* ===== Cards (containers) ===== */
    div[data-testid="stMetric"],
    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        background: white;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.08);
    }

    /* ===== Input Widgets (Main area) ===== */
    .stSelectbox, .stNumberInput, .stTextInput {
        background: white;
        border-radius: 10px;
    }

    /* ===== Progress Bar ===== */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #1F3C88, #4A90E2);
    }

    /* ===== Footer hide ===== */
    footer {visibility: hidden;}

    </style>
    """,
    unsafe_allow_html=True
)
#Page setup and color ends here


st.set_page_config(page_title='ML Assignment 2 - Classification Models', layout='wide')


@st.cache_data
def load_metrics_table():
    p = Path('model') / 'model_comparison_metrics.csv'
    return pd.read_csv(p)


@st.cache_data
def load_artifacts():
    with open(Path('model') / 'artifacts.json', 'r', encoding='utf-8') as f:
        return json.load(f)


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(model_file)


def plot_confusion_matrix(cm, labels=('benign', 'malignant')):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


def compute_user_metrics(y_true, y_pred, y_score=None):
    from sklearn.metrics import (
        accuracy_score, roc_auc_score, precision_score, recall_score,
        f1_score, matthews_corrcoef, confusion_matrix, classification_report
    )
    m = {
        'Accuracy': float(accuracy_score(y_true, y_pred)),
        'Precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'F1': float(f1_score(y_true, y_pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None:
        m['AUC'] = float(roc_auc_score(y_true, y_score))
    else:
        m['AUC'] = np.nan
    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=['benign', 'malignant'], zero_division=0)
    return m, cm, rep


st.title('My ML Assignment 2 - End-to-End Classification & Streamlit')
st.write('Six trained classification models based on the Breast Cancer Wisconsin (Diagnostic) dataset are integrated into this application for two-class prediction.')

metrics_df = load_metrics_table()
artifacts = load_artifacts()

with st.sidebar:
    st.header('Drop Down: Select model')
    model_name = st.selectbox('Select model From Drop Down', metrics_df['ML Model Name'].tolist())
    show_holdout = st.checkbox('Show Default evaluation ', value=True)
    st.divider()
    st.subheader('Upload test CSV (optional)')
    
    st.caption(
    "Tip: Upload only test data. Include a target column if you want evaluation.\n"
    "Target should be 1=malignant, 0=benign OR strings malignant/benign."
    )

    uploaded = st.file_uploader('Upload CSV', type=['csv'])

# Selected model
info = artifacts[model_name]
model = load_model(info['model_file'])

col1, col2 = st.columns([1.25, 1])

with col1:
    st.subheader('Model comparison table (holdout test set)')
    st.dataframe(metrics_df.style.format({
        'Accuracy': '{:.4f}', 'AUC': '{:.4f}', 'Precision': '{:.4f}',
        'Recall': '{:.4f}', 'F1': '{:.4f}', 'MCC': '{:.4f}'
    }), use_container_width=True)

    st.subheader(f'Selected model: {model_name}')
    m = info['metrics']
    st.metric('Accuracy', f"{m['Accuracy']:.4f}")
    st.metric('AUC', f"{m['AUC']:.4f}")
    st.metric('Precision', f"{m['Precision']:.4f}")
    st.metric('Recall', f"{m['Recall']:.4f}")
    st.metric('F1', f"{m['F1']:.4f}")
    st.metric('MCC', f"{m['MCC']:.4f}")

with col2:
    if show_holdout:
        st.subheader('Confusion matrix (holdout)')
        cm = np.array(info['confusion_matrix'])
        st.pyplot(plot_confusion_matrix(cm))

        st.subheader('Classification report (holdout)')
        st.code(info['classification_report'])

st.divider()
st.subheader('Try the model on your uploaded test data')

# Provide sample template
df_full = load_dataset(as_frame=True)
feature_cols = [c for c in df_full.columns if c not in ('target', 'target_label')]

st.write('Download a sample test CSV template (features only):')
sample_features = df_full[feature_cols].sample(25, random_state=7)
st.download_button('Download sample_test_template.csv', sample_features.to_csv(index=False), file_name='sample_test_template.csv', mime='text/csv')

if uploaded is None:
    st.info('Upload a CSV to run predictions, or use the template above.')
else:
    user_df = pd.read_csv(uploaded)
    st.write('Preview of uploaded data:')
    st.dataframe(user_df.head(), use_container_width=True)

    # Determine if target column exists
    possible_targets = [c for c in user_df.columns if c.lower() in ('target', 'label', 'diagnosis', 'class')]
    target_col = None
    if possible_targets:
        target_col = st.selectbox('Select target column (optional)', ['<None>'] + possible_targets)
        if target_col == '<None>':
            target_col = None

    # Ensure feature columns match
    missing = [c for c in feature_cols if c not in user_df.columns]
    if missing:
        st.error(f"Missing required feature columns: {missing[:8]}{'...' if len(missing)>8 else ''}")
    else:
        X_user = user_df[feature_cols]
        y_pred = model.predict(X_user)
        y_score = None
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_user)[:, 1]
        preds = pd.DataFrame({
            'prediction': y_pred,
            'prediction_label': pd.Series(y_pred).map({1: 'malignant', 0: 'benign'})
        })
        if y_score is not None:
            preds['malignant_probability'] = y_score

        st.write('Predictions:')
        st.dataframe(pd.concat([user_df.reset_index(drop=True), preds], axis=1).head(50), use_container_width=True)
        st.download_button('Download predictions CSV', pd.concat([user_df.reset_index(drop=True), preds], axis=1).to_csv(index=False),
                            file_name='predictions.csv', mime='text/csv')

        # Optional evaluation if target provided
        if target_col is not None:
            y_true_raw = user_df[target_col]
            if y_true_raw.dtype == object:
                y_true = y_true_raw.astype(str).str.lower().map({'malignant': 1, 'm': 1, 'benign': 0, 'b': 0})
            else:
                y_true = y_true_raw
            y_true = pd.to_numeric(y_true, errors='coerce')
            if y_true.isna().any():
                st.warning('Some target values could not be parsed to 0/1. Please clean the target column.')
            else:
                user_metrics, cm_u, rep_u = compute_user_metrics(y_true.values.astype(int), y_pred.astype(int), y_score)
                st.subheader('Evaluation on uploaded data')
                st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in user_metrics.items()})
                st.pyplot(plot_confusion_matrix(cm_u))
                st.code(rep_u)
