from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline

from preprocess import (
    load_and_prepare_data,
    split_data,
    build_tree_preprocessor,
)


def tune_decision_tree():

    X, y = load_and_prepare_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_tree_preprocessor(X_train)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("tree", DecisionTreeClassifier(random_state=2025)),
        ]
    )

    param_grid = {
        "tree__max_depth": [None, 5, 7, 8, 10, 15, 20],
        "tree__min_samples_split": [2, 8, 10, 20, 50],
        "tree__min_samples_leaf": [1, 3, 4, 5, 10],
        "tree__class_weight": [None, "balanced"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    print("Running GridSearchCV for Decision Tree...")
    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:", grid_search.best_params_)
    print("Best CV ROC-AUC:", grid_search.best_score_)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return best_model, best_params


if __name__ == "__main__":
    best_model, best_params = tune_decision_tree()
    print("\nBest parameters (for final model):", best_params)