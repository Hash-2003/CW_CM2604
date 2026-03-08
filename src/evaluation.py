from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from DecisionTree import run_tree_experiment
from NeuralNetwork import run_nn_experiment
from preprocess import(
    load_and_prepare_data,
    split_data
)

def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "Model",
    verbose: bool = True,
) -> Dict[str, Any]:

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": None,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = None

    if verbose:
        print(f"\n=== {model_name} Metrics ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-score : {f1:.4f}")
        if metrics["roc_auc"] is not None:
            print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
        else:
            print("ROC-AUC  : N/A (no probabilities or only one class)")

    return metrics


def print_comparison(tree_metrics: Dict[str, Any], nn_metrics: Dict[str, Any]) -> None:
    print("\n================ Model Comparison ================")
    print(f"{'Metric':<10} {'Decision Tree':>15} {'Neural Network':>18}")
    print("-" * 45)

    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("roc_auc", "ROC-AUC"),
    ]:
        tree_val = tree_metrics.get(key)
        nn_val = nn_metrics.get(key)

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and v is not None else "N/A"

        print(f"{label:<10} {fmt(tree_val):>15} {fmt(nn_val):>18}")

    print("=================================================\n")


def main():
    X, y = load_and_prepare_data()

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Running Decision Tree experiment for evaluation...")
    y_test_tree, y_pred_tree, y_prob_tree, _ = run_tree_experiment(
        X_train,
        X_test,
        y_train,
        y_test,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=10,
        class_weight="balanced",
    )

    print("Running Neural Network experiment for evaluation...")
    y_test_nn, y_pred_nn, y_prob_nn, _ = run_nn_experiment(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        hidden_units1=32,
        hidden_units2=16,
        learning_rate=0.0005,
        batch_size=32,
        dropout_rate=0.1,
        l2_reg=0.0,
    )

    if len(y_test_tree) != len(y_test_nn):
        print("Warning: tree and NN test sets have different lengths.")
        print(f"Tree test size: {len(y_test_tree)}, NN test size: {len(y_test_nn)}")

    tree_metrics = evaluate_classification(
        y_true=y_test_tree,
        y_pred=y_pred_tree,
        y_prob=y_prob_tree,
        model_name="Decision Tree",
        verbose=True,
    )

    nn_metrics = evaluate_classification(
        y_true=y_test_nn,
        y_pred=y_pred_nn,
        y_prob=y_prob_nn,
        model_name="Neural Network",
        verbose=True,
    )

    print_comparison(tree_metrics, nn_metrics)


if __name__ == "__main__":
    main()
