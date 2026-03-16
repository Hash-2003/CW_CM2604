from typing import Tuple
import os
import random

SEED = 2025
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from preprocess import build_nn_preprocessor

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass


def build_nn_model(
    input_dim: int,
    units1: int = 64,
    units2: int = 32,
    learning_rate: float = 5e-4,
    dropout_rate: float = 0.1,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    initializer = tf.keras.initializers.GlorotUniform(seed=SEED)
    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(units1, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizer),
        Dropout(dropout_rate, seed=SEED),

        Dense(units2, activation="relu", kernel_initializer=initializer, kernel_regularizer=regularizer),
        Dropout(dropout_rate, seed=SEED),

        Dense(1, activation="sigmoid", kernel_initializer=initializer),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
        ],
    )

    return model


def find_best_threshold(y_true, y_prob, metric: str = "f1"):
    best_threshold = 0.5
    best_score = 0.0

    for t in np.arange(0.30, 0.60, 0.01):
        y_pred = (y_prob >= t).astype(int)

        if metric == "f2":
            score = fbeta_score(y_true, y_pred, beta=2.0)
        else:
            score = f1_score(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


def run_nn_experiment(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train,
    y_val,
    y_test,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    units1: int = 64,
    units2: int = 32,
    dropout_rate: float = 0.1,
    l2_reg: float = 1e-4,
    threshold_metric: str = "f1",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple, float, object]:

    print("Building NN preprocessor...")
    preprocessor = build_nn_preprocessor(X_train)

    print("Transforming data...")
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    if hasattr(X_train_t, "toarray"):
        X_train_t = X_train_t.toarray()
        X_val_t = X_val_t.toarray()
        X_test_t = X_test_t.toarray()

    print("Building Neural Network...")
    model = build_nn_model(
        input_dim=X_train_t.shape[1],
        units1=units1,
        units2=units2,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
    )

    print("Computing class weights...")
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight={
            0: 1.0,
            1: 2.5
        },
        classes=classes,
        y=y_train,
    )
    class_weight = {cls: weight for cls, weight in zip(classes, weights)}
    print("Class weights:", class_weight)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        X_train_t,
        y_train,
        validation_data=(X_val_t, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    print(f"Finding best threshold on validation set using {threshold_metric.upper()}...")
    y_val_prob = model.predict(X_val_t, verbose=0).ravel()
    best_threshold, best_val_score = find_best_threshold(y_val, y_val_prob, metric=threshold_metric)

    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Best validation {threshold_metric.upper()}: {best_val_score:.4f}")

    print("Generating predictions on test set...")
    y_test_prob = model.predict(X_test_t, verbose=0).ravel()
    y_test_pred = (y_test_prob >= best_threshold).astype(int)

    y_test_array = y_test.to_numpy() if hasattr(y_test, "to_numpy") else np.array(y_test)

    return (
        y_test_array,
        y_test_pred,
        y_test_prob,
        (preprocessor, model),
        best_threshold,
        history,
    )