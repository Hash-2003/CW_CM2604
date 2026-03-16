from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_tree_preprocessor,
)

def tune_random_forest():
    X, y = load_and_prepare_data()

    X_train, _, _, y_train, _, _ = split_data(X, y)

    preprocessor = build_tree_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("rf", RandomForestClassifier(random_state=2025, n_jobs=-1)),
        ]
    )

    param_grid = {
        "rf__n_estimators": [200, 300, 500],
        "rf__max_depth": [None, 8, 10, 15],
        "rf__min_samples_split": [2, 10, 20],
        "rf__min_samples_leaf": [1, 3, 5],
        "rf__max_features": ["sqrt", "log2"],
        "rf__class_weight": [None, "balanced", "balanced_subsample"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("Running GridSearchCV for Random Forest...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print("Best CV ROC-AUC:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


if __name__ == "__main__":
    best_model, best_params = tune_random_forest()
    print("\nBest parameters (for final Random Forest):", best_params)