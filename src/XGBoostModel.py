from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import build_tree_preprocessor


def build_xgb_model(
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    min_child_weight: int = 1,
    scale_pos_weight: float = 1.0,
    gamma: float = 0.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 2025,
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f2 = 0.0

    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob >= t).astype(int)
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
    n_estimators=500,
    learning_rate=0.03,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, object, float]:

    print("Building preprocessor...")
    preprocessor = build_tree_preprocessor(X_train)

    print("Transforming data...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    print("Building XGBoost model...")
    xgb = build_xgb_model(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        scale_pos_weight=scale_pos_weight,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
    )

    print("Training model...")
    xgb.fit(
        X_train_t,
        y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False
    )

    print("Finding best threshold on validation set...")
    y_val_prob = xgb.predict_proba(X_val_t)[:, 1]
    best_threshold, best_val_f2 = find_best_threshold(y_val, y_val_prob)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation F2: {best_val_f2:.4f}")

    print("Generating predictions on test set...")
    y_test_prob = xgb.predict_proba(X_test_t)[:, 1]
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", xgb),
        ]
    )

    return y_test.to_numpy(), y_test_pred, y_test_prob, model, best_threshold