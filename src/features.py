"""Engenharia de features sem vazamento temporal (informação disponível antes da partida)."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Colunas usadas como preditores — nada medido após a partida programada
FEATURE_COLUMNS = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DISTANCE",
]


def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Converte HHMM (inteiro ou string) para minutos desde meia-noite."""

    def one(v) -> float:
        if pd.isna(v):
            return np.nan
        s = str(int(float(v))).zfill(4)
        h, m = int(s[:-2]), int(s[-2:])
        return h * 60 + m

    return series.map(one)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["DEP_MIN"] = hhmm_to_minutes(out["SCHEDULED_DEPARTURE"])
    out["ARR_MIN"] = hhmm_to_minutes(out["SCHEDULED_ARRIVAL"])
    return out


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove cancelados e desviados (opcional: foco em voos que chegam ao destino planeado).
    Cria alvo de classificação (atraso na chegada > 15 min) e alvo de regressão (minutos).
    """
    d = df.loc[df["CANCELLED"] == 0].copy()
    d = d.loc[d["DIVERTED"] == 0]
    # Atraso na chegada: padrão da indústria para "atrasado" = > 15 min
    d["DELAYED_ARRIVAL"] = (d["ARRIVAL_DELAY"] > 15).astype(int)
    d["ARRIVAL_DELAY_MIN"] = pd.to_numeric(d["ARRIVAL_DELAY"], errors="coerce")
    d = add_time_features(d)
    # Regressão: apenas voos com chegada registada
    d = d.dropna(subset=["ARRIVAL_DELAY_MIN", "DEP_MIN", "ARR_MIN"])
    return d


def X_y_classification(d: pd.DataFrame):
    y = d["DELAYED_ARRIVAL"].values
    X = d[
        [
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "AIRLINE",
            "ORIGIN_AIRPORT",
            "DESTINATION_AIRPORT",
            "DEP_MIN",
            "ARR_MIN",
            "SCHEDULED_TIME",
            "DISTANCE",
        ]
    ].copy()
    for col in ("AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        X[col] = X[col].astype(str)
    return X, y


def X_y_regression(d: pd.DataFrame):
    y = d["ARRIVAL_DELAY_MIN"].values
    X, _ = X_y_classification(d)
    return X, y
