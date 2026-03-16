import numpy as np
import tensorflow as tf
import os
import random
from tensorflow import keras
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_nn_preprocessor,
)

SEED = 2025

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_model(input_dim, units1, units2, lr, dropout_rate, l2_reg):
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(units1, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(units2, activation="relu", kernel_regularizer=regularizer),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="roc_auc")]
    )
    return model


def tune_neural_network():
    X, y = load_and_prepare_data()
    X_train, X_val, _, y_train, y_val, _ = split_data(X, y)

    preprocessor = build_nn_preprocessor(X_train)
    preprocessor.fit(X_train)

    X_train_nn = preprocessor.transform(X_train)
    X_val_nn = preprocessor.transform(X_val)

    if hasattr(X_train_nn, "toarray"):
        X_train_nn = X_train_nn.toarray()
        X_val_nn = X_val_nn.toarray()
    else:
        X_train_nn = np.asarray(X_train_nn)
        X_val_nn = np.asarray(X_val_nn)

    architectures = [(32, 16), (64, 32), (128, 64)]
    learning_rates = [0.001, 0.0005, 0.0001]
    batch_sizes = [32, 64]
    dropout_rates = [0.0, 0.1, 0.2]
    l2_regs = [0.0, 1e-4, 5e-4]

    best_auc = -1.0
    best_config = None
    best_model = None

    print("===== Neural Network Hyperparameter Tuning =====\n")

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
    )

    class_weight = {
        0: class_weights_array[0],
        1: class_weights_array[1],
    }

    for units1, units2 in architectures:
        for lr in learning_rates:
            for batch in batch_sizes:
                for dr in dropout_rates:
                    for l2r in l2_regs:

                        print("\nTesting config:")
                        print(f"   units=({units1},{units2}), lr={lr}, batch={batch}, dropout={dr}, l2={l2r}")

                        model = build_model(
                            input_dim=X_train_nn.shape[1],
                            units1=units1,
                            units2=units2,
                            lr=lr,
                            dropout_rate=dr,
                            l2_reg=l2r
                        )

                        callbacks = [
                            keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=3,
                                restore_best_weights=True
                            ),
                            keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss",
                                factor=0.5,
                                patience=2,
                                min_lr=1e-6,
                                verbose=0
                            )
                        ]

                        model.fit(
                            X_train_nn,
                            y_train,
                            validation_data=(X_val_nn, y_val),
                            epochs=30,
                            batch_size=batch,
                            callbacks=callbacks,
                            class_weight=class_weight,
                            verbose=0
                        )

                        y_prob_val = model.predict(X_val_nn, verbose=0).ravel()
                        auc_score = roc_auc_score(y_val, y_prob_val)

                        print(f"   Validation ROC-AUC: {auc_score:.4f}")

                        if auc_score > best_auc:
                            best_auc = auc_score
                            best_model = model
                            best_config = {
                                "units1": units1,
                                "units2": units2,
                                "learning_rate": lr,
                                "batch_size": batch,
                                "dropout_rate": dr,
                                "l2_reg": l2r,
                            }

    print("\n===== Best Configuration Found =====")
    print(best_config)
    print(f"Best validation ROC-AUC: {best_auc:.4f}")

    return best_model, best_config, best_auc


if __name__ == "__main__":
    best_model, best_params, best_auc = tune_neural_network()
    print("\nBest parameters (for final NN model):", best_params)
    print("Best validation ROC-AUC:", best_auc)