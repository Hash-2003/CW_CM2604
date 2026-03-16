from typing import Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve, auc,
)

from DecisionTree import run_tree_experiment
from NeuralNetwork import run_nn_experiment
from preprocess import (
    load_and_prepare_data,
    split_data
)
from RandomForest import run_random_forest_experiment
from XGBoostModel import run_xgb_experiment
from sklearn.metrics import precision_recall_curve

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

def plot_tree_feature_importance(model, top_n: int = 10):
    preprocessor = model.named_steps["preprocess"]
    tree = model.named_steps["tree"]

    feature_names = preprocessor.get_feature_names_out()
    importances = tree.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    importance_df = importance_df[importance_df["importance"] > 0]
    importance_df = importance_df.sort_values(by="importance", ascending=False)

    print("\nTop Decision Tree Features:")
    print(importance_df.head(top_n))

    plt.figure(figsize=(8, 5))
    plt.barh(
        importance_df["feature"].head(top_n)[::-1],
        importance_df["importance"].head(top_n)[::-1]
    )
    plt.title("Top Decision Tree Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_conf_matrix(y_true, y_pred, title: str):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Churn", "Churn"]
    )
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_true, y_prob, title):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(6,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def print_comparison(tree_metrics, rf_metrics, nn_metrics, xgb_metrics):

    print("\n==================== Model Comparison ====================")
    print(
        f"{'Metric':<10} {'Decision Tree':>15} {'Random Forest':>15} "
        f"{'Neural Network':>18} {'XGBoost':>12}"
    )
    print("-" * 80)

    for key, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("roc_auc", "ROC-AUC"),
    ]:
        tree_val = tree_metrics.get(key)
        rf_val = rf_metrics.get(key)
        nn_val = nn_metrics.get(key)
        xgb_val = xgb_metrics.get(key)

        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) and v is not None else "N/A"

        print(
            f"{label:<10} {fmt(tree_val):>15} {fmt(rf_val):>15} "
            f"{fmt(nn_val):>18} {fmt(xgb_val):>12}"
        )

    print("==========================================================\n")

def plot_roc_curve(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X, y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("Running Decision Tree experiment for evaluation...")
    y_test_tree, y_pred_tree, y_prob_tree, tree_model, tree_threshold = run_tree_experiment(
        X_train,
        X_val,  # <--- Passing X_val
        X_test,
        y_train,
        y_val,  # <--- Passing y_val
        y_test,
        max_depth=5,
        min_samples_split=50,
        min_samples_leaf=10,
        class_weight="balanced",
    )

    print("Running Random Forest experiment for evaluation...")
    y_test_rf, y_pred_rf, y_prob_rf, rf_model, rf_threshold = run_random_forest_experiment(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
    )

    print("Running Neural Network experiment for evaluation...")
    y_test_nn, y_pred_nn, y_prob_nn, nn_model, nn_threshold, history = run_nn_experiment(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        epochs=40,
        batch_size=64,
        learning_rate=0.0005,
        units1=64,
        units2=32,
        dropout_rate=0.15,
        l2_reg=0.0005,
        threshold_metric="f2",
    )

    print("Running XGBoost experiment for evaluation...")

    imbalance_ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
    print(f"Calculated scale_pos_weight for XGBoost: {imbalance_ratio:.4f}")

    y_test_xgb, y_pred_xgb, y_prob_xgb, xgb_model, xgb_threshold = run_xgb_experiment(
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
        scale_pos_weight=imbalance_ratio,
    )

    tree_metrics = evaluate_classification(
        y_true=y_test_tree,
        y_pred=y_pred_tree,
        y_prob=y_prob_tree,
        model_name="Decision Tree",
        verbose=True,
    )

    rf_metrics = evaluate_classification(
        y_true=y_test_rf,
        y_pred=y_pred_rf,
        y_prob=y_prob_rf,
        model_name="Random Forest",
        verbose=True,
    )

    xgb_metrics = evaluate_classification(
        y_true=y_test_xgb,
        y_pred=y_pred_xgb,
        y_prob=y_prob_xgb,
        model_name="XGBoost",
        verbose=True,
    )

    nn_metrics = evaluate_classification(
        y_true=y_test_nn,
        y_pred=y_pred_nn,
        y_prob=y_prob_nn,
        model_name="Neural Network",
        verbose=True,
    )

    print_comparison(tree_metrics, rf_metrics, nn_metrics, xgb_metrics)

    # confusion matrices
    plot_conf_matrix(y_test_tree, y_pred_tree, "Decision Tree Confusion Matrix")
    plot_conf_matrix(y_test_rf, y_pred_rf, "Random Forest Confusion Matrix")
    plot_conf_matrix(y_test_nn, y_pred_nn, "Neural Network Confusion Matrix")
    plot_conf_matrix(y_test_xgb, y_pred_xgb, "XGBoost Confusion Matrix")

    # tree feature importance
    plot_tree_feature_importance(tree_model, top_n=10)

    # NN training curves - Loss
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Neural Network Training Curve (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # NN training curves - Accuracy
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Neural Network Training Curve (Accuracy)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # NN training curves - ROC AUC
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["roc_auc"], label="Train ROC-AUC")
    plt.plot(history.history["val_roc_auc"], label="Validation ROC-AUC")
    plt.title("Neural Network Training Curve (ROC-AUC)")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ROC curves
    plot_roc_curve(y_test_tree, y_prob_tree, "Decision Tree ROC Curve")
    plot_roc_curve(y_test_rf, y_prob_rf, "Random Forest ROC Curve")
    plot_roc_curve(y_test_nn, y_prob_nn, "Neural Network ROC Curve")
    plot_roc_curve(y_test_xgb, y_prob_xgb, "XGBoost ROC Curve")

    plot_pr_curve(y_test_nn, y_prob_nn, "Neural Network PR Curve")
    plot_pr_curve(y_test_rf, y_prob_rf, "Random Forest PR Curve")
    plot_pr_curve(y_test_xgb, y_prob_xgb, "XGBoost PR Curve")


if __name__ == "__main__":
    main()