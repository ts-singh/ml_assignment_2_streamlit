"""Data loading utilities.

Dataset: Breast Cancer Wisconsin (Diagnostic) (UCI).
We load it via scikit-learn's built-in loader (a copy of the UCI dataset).
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd


def load_dataset(as_frame: bool = True) -> pd.DataFrame:
    """Return a DataFrame with features + target.

    Target is mapped to binary where 1 = malignant, 0 = benign.
    """
    ds = load_breast_cancer(as_frame=as_frame)
    df = ds.frame.copy()
    # scikit-learn encodes target 0=malignant, 1=benign.
    df['target'] = (df['target'] == 0).astype(int)
    df['target_label'] = df['target'].map({1: 'malignant', 0: 'benign'})
    return df


def get_feature_target(df: pd.DataFrame, target_col: str = 'target'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
