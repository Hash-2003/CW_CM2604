from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import fbeta_score
from xgboost import XGBClassifier

from preprocess import build_tree_preprocessor


def build_xgb_model(
    X_train: pd.DataFrame,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    scale_pos_weight: float = 1.0,
    random_state: int = 2025,
) -> Pipeline:
    """
    Builds the XGBoost pipeline including preprocessing.
    """

    preprocessor = build_tree_preprocessor(X_train)

    xgb = XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", xgb),
        ]
    )

    return model


def find_best_threshold(y_true, y_prob):
    """
    Finds the threshold that maximizes the F2-score,
    prioritizing Recall for churn prediction.
    """
    best_threshold = 0.5
    best_f2 = 0.0

    for t in np.arange(0.20, 0.60, 0.01):
        y_pred = (y_prob >= t).astype(int)

        # Calculate F2-score instead of F1
        f2 = fbeta_score(y_true, y_pred, beta=2.0)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    return best_threshold, best_f2


def run_xgb_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    scale_pos_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, float]:

    print("Building XGBoost model...")

    model = build_xgb_model(
        X_train=X_train,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
    )

    print("Training model...")
    model.fit(X_train, y_train)

    print("Finding best threshold on validation set...")
    y_val_prob = model.predict_proba(X_val)[:, 1]

    best_threshold, best_val_f2 = find_best_threshold(y_val, y_val_prob)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation F2: {best_val_f2:.4f}")

    print("Generating predictions on test set...")
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    return y_test.to_numpy(), y_test_pred, y_test_prob, model, best_threshold