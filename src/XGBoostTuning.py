from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_tree_preprocessor,
)


def tune_xgboost():
    X, y = load_and_prepare_data()
    X_train, _, _, y_train, _, _ = split_data(X, y)

    preprocessor = build_tree_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("xgb", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=2025,
                n_jobs=-1,
            )),
        ]
    )

    param_grid = {
        "xgb__n_estimators": [200, 300, 500],
        "xgb__learning_rate": [0.03, 0.05, 0.1],
        "xgb__max_depth": [3, 5, 7],
        "xgb__subsample": [0.8, 1.0],
        "xgb__colsample_bytree": [0.8, 1.0],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__scale_pos_weight": [1, 2, 3],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("Running GridSearchCV for XGBoost...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print("Best CV ROC-AUC:", grid_search.best_score_)

    return grid_search.best_estimator_, grid_search.best_params_


if __name__ == "__main__":
    best_model, best_params = tune_xgboost()
    print("\nBest parameters (for final XGBoost):", best_params)