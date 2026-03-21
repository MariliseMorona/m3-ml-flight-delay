"""Pré-processamento para modelos sklearn (imputação + escala + one-hot)."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "DEP_MIN",
    "ARR_MIN",
    "SCHEDULED_TIME",
    "DISTANCE",
]
CATEGORICAL = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]


def make_column_transformer(max_categories: int = 40) -> ColumnTransformer:
    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    max_categories=max_categories,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, NUMERIC),
            ("cat", cat_pipe, CATEGORICAL),
        ]
    )
