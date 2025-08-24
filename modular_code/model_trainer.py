# model_trainer.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.model_selection import GridSearchCV

def default_model_zoo(random_state: int = 42) -> Dict[str, Pipeline]:
    """
    Returns a dict of model name -> Pipeline(StandardScaler -> Estimator).
    Scaling is included to avoid leakage (fit on train only inside Pipeline).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, n_jobs=None))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),  # RF not sensitive to scaling, keep light
            ("model", RandomForestClassifier(random_state=random_state))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model", DecisionTreeClassifier(random_state=random_state))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier())
        ]),
        
              
    }

def default_param_grids() -> Dict[str, dict]:
    """
    Minimal, sane grids you can expand later.
    """
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 3.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "saga"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [None, 8, 14],
            "model__min_samples_split": [2, 5],
        },
        "Decision Tree": {
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
        },
        "KNN": {
            "model__n_neighbors": [3, 5, 7,  nine := 9],
            "model__weights": ["uniform", "distance"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },

    }

def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_grid_search: bool = False,
    cv: int = 3,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """
    Trains a suite of models and returns fitted estimators.
    Set use_grid_search=True to perform GridSearchCV with default grids.
    """
    models = default_model_zoo(random_state=random_state)
    fitted: Dict[str, Pipeline] = {}

    if use_grid_search:
        grids = default_param_grids()

    for name, pipe in models.items():
        if use_grid_search and name in grids:
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grids[name],
                cv=cv,
                n_jobs=n_jobs,
                scoring="f1_macro",
                refit=True,
            )
            gs.fit(X_train, y_train)
            fitted[name] = gs.best_estimator_
            print(f"[{name}] best params: {gs.best_params_}")
        else:
            pipe.fit(X_train, y_train)
            fitted[name] = pipe
    return fitted
