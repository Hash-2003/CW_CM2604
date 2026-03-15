from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_tree_preprocessor,
)


def build_decision_tree_model(
    X_train: pd.DataFrame,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
    random_state: int = 2025,
) -> Pipeline:
    preprocessor = build_tree_preprocessor(X_train)

    tree_clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("tree", tree_clf),
        ]
    )

    return model


def find_best_threshold(y_true, y_prob):
    best_threshold = 0.5
    best_f2 = 0.0

    for t in np.arange(0.20, 0.60, 0.01):
        y_pred = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_true, y_pred, beta=2.0)

        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    return best_threshold, best_f2


def run_tree_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Pipeline, float]:

    print("Building Decision Tree model...")
    model = build_decision_tree_model(
        X_train=X_train,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
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
