from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from catboost import CatBoostRegressor
except Exception:  # pragma: no cover
    CatBoostRegressor = None


def make_preprocessor(X):
    """Build a preprocessing block that handles numeric and categorical columns."""
    categorical = [c for c in X.columns if X[c].dtype == "object"]
    numeric = [c for c in X.columns if c not in categorical]

    # Median-impute numeric columns, then standardize for linear models.
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    # One-hot encode categorical variables while tolerating unseen levels at test time.
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def make_models(config: dict, preprocessor) -> dict:
    """Assemble the model zoo used in the baseline comparison."""
    models = {
        "ridge": Pipeline([("preprocessor", preprocessor), ("model", Ridge(alpha=float(config["ridge"]["alpha"]))) ]),
        "lasso": Pipeline([("preprocessor", preprocessor), ("model", Lasso(alpha=float(config["lasso"]["alpha"]), max_iter=20000))]),
        "et": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=int(config["et"]["n_estimators"]),
                    max_depth=config["et"]["max_depth"],
                    min_samples_leaf=int(config["et"]["min_samples_leaf"]),
                    random_state=int(config["et"]["random_state"]),
                    n_jobs=-1,
                ),
            ),
        ]),
        "hgb": Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=float(config["hgb"]["learning_rate"]),
                    max_depth=int(config["hgb"]["max_depth"]),
                    max_iter=int(config["hgb"]["max_iter"]),
                    min_samples_leaf=int(config["hgb"]["min_samples_leaf"]),
                    random_state=int(config["hgb"]["random_state"]),
                ),
            ),
        ]),
    }

    # CatBoost is optional so the project still runs in a lighter environment.
    if CatBoostRegressor is not None:
        models["catboost"] = Pipeline([
            ("preprocessor", preprocessor),
            (
                "model",
                CatBoostRegressor(
                    iterations=int(config["catboost"]["iterations"]),
                    learning_rate=float(config["catboost"]["learning_rate"]),
                    depth=int(config["catboost"]["depth"]),
                    loss_function=config["catboost"]["loss_function"],
                    random_seed=int(config["catboost"]["random_seed"]),
                    verbose=False,
                ),
            ),
        ])

    models["mlp"] = Pipeline([
        ("preprocessor", preprocessor),
        (
            "model",
            MLPRegressor(
                hidden_layer_sizes=tuple(config["mlp"]["hidden_layer_sizes"]),
                activation="relu",
                alpha=float(config["mlp"]["alpha"]),
                learning_rate_init=float(config["mlp"]["learning_rate_init"]),
                max_iter=int(config["mlp"]["max_iter"]),
                early_stopping=True,
                n_iter_no_change=int(config["mlp"]["n_iter_no_change"]),
                random_state=int(config["mlp"]["random_state"]),
            ),
        ),
    ])

    return models