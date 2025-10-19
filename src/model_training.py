from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from .data_preprocessing import read_superstore_csv
from .feature_engineering import add_basic_features
from .model_utils import save_model

DEFAULT_MODEL_PATH = "models/model_profit_clf.joblib"

def get_feature_columns(df: pd.DataFrame) -> Tuple[list, list]:
    num_features = [c for c in ["Sales", "Quantity", "Discount", "Days_to_Ship"] if c in df.columns]
    cat_features = [c for c in ["Ship Mode", "Segment", "Category", "Sub-Category", "Region"] if c in df.columns]
    return num_features, cat_features

def train_profit_classifier(csv_path: str, model_out: str = DEFAULT_MODEL_PATH, test_size: float = 0.2, random_state: int = 42) -> Dict:
    # Load & FE
    df = read_superstore_csv(csv_path)
    df = add_basic_features(df)

    # Drop rows without target
    df = df.dropna(subset=["Profitable"])

    # Features/Target
    num_features, cat_features = get_feature_columns(df)
    feature_cols = num_features + cat_features
    X = df[feature_cols].copy()
    y = df["Profitable"].astype(int)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Preprocess
    preproc = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="passthrough"
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])

    # Fit
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0)
    }

    # Save model
    save_model(pipe, model_out)

    return {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_path": model_out
    }
