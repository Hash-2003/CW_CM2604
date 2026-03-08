from typing import Tuple
import os
import random
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import (
    build_nn_preprocessor,
)

SEED = 2025

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_nn_model(
    input_dim: int,
    hidden_units1: int = 32,
    hidden_units2: int = 16,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
) -> keras.Model:
    """
    Feed-forward neural network (MLP)

    Input -> Dense -> Dense -> Dropout -> Dense(1, sigmoid)
    """
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(hidden_units1, activation="relu", kernel_regularizer=regularizer),
            layers.Dense(hidden_units2, activation="relu", kernel_regularizer=regularizer),
            layers.Dropout(dropout_rate),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def run_nn_experiment(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    epochs: int = 20,
    batch_size: int = 32,
    hidden_units1: int = 32,
    hidden_units2: int = 16,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    l2_reg: float = 0.0,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, keras.Model]:

    print("Building NN preprocessing pipeline...")
    preprocessor = build_nn_preprocessor(X_train)

    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    print("Transforming train, validation, and test data...")
    X_train_nn = preprocessor.transform(X_train)
    X_val_nn = preprocessor.transform(X_val)
    X_test_nn = preprocessor.transform(X_test)

    if hasattr(X_train_nn, "toarray"):
        X_train_nn = X_train_nn.toarray()
        X_val_nn = X_val_nn.toarray()
        X_test_nn = X_test_nn.toarray()
    else:
        X_train_nn = np.asarray(X_train_nn)
        X_val_nn = np.asarray(X_val_nn)
        X_test_nn = np.asarray(X_test_nn)

    input_dim = X_train_nn.shape[1]
    print(f"Input dimension after preprocessing: {input_dim}")

    print("Building Neural Network model...")
    model = build_nn_model(
        input_dim=input_dim,
        hidden_units1=hidden_units1,
        hidden_units2=hidden_units2,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        l2_reg=l2_reg,
    )

    print("Training Neural Network model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train,
    )

    class_weight = {
        0: class_weights_array[0],
        1: class_weights_array[1],
    }

    print("Using class weights:", class_weight)

    model.fit(
        X_train_nn,
        y_train,
        validation_data=(X_val_nn, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    print("Generating predictions on test set...")
    y_prob = model.predict(X_test_nn).ravel()
    y_pred_binary = (y_prob >= threshold).astype(int)

    return y_test.to_numpy(), y_pred_binary, y_prob, model