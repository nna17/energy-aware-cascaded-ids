"""Feature transform fitted on the training split only, plus importance ranking for elastic inputs."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, RobustScaler, StandardScaler

_SCALERS = {
    "standard": StandardScaler,
    "robust": RobustScaler,
    "quantile": lambda: QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=0),
}


class Preprocessor:
    """Numeric: impute + scale. Categorical: one-hot (rare categories grouped). Constant columns removed."""

    def __init__(self, scaler: str = "quantile", max_categories: int = 16):
        self.scaler = scaler
        self.max_categories = max_categories
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "Preprocessor":
        numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in feature_cols if c not in numeric]
        transformers = [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", _SCALERS[self.scaler]())]), numeric),
        ]
        if categorical:
            transformers.append((
                "cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=self.max_categories, sparse_output=False),
                categorical,
            ))
        self.pipeline = Pipeline([
            ("columns", ColumnTransformer(transformers, verbose_feature_names_out=False)),
            ("drop_constant", VarianceThreshold(0.0)),
        ])
        self.pipeline.fit(df[feature_cols])
        self.feature_names = list(self.pipeline.get_feature_names_out())
        self._feature_cols = feature_cols
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform(df[self._feature_cols]).astype(np.float32)


def rank_features(X: np.ndarray, y: np.ndarray, seed: int = 0, max_rows: int = 200_000) -> np.ndarray:
    """Column order by LightGBM gain importance (most important first).

    Elastic sub-models use the first k columns, so ranking decides which features small models keep.
    """
    rng = np.random.default_rng(seed)
    if len(X) > max_rows:
        idx = rng.choice(len(X), max_rows, replace=False)
        X, y = X[idx], y[idx]
    model = lgb.LGBMClassifier(n_estimators=100, num_leaves=31, importance_type="gain",
                               class_weight="balanced", random_state=seed, verbose=-1)
    model.fit(X, y)
    return np.argsort(-model.feature_importances_, kind="stable")
