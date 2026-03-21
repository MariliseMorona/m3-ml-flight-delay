"""Carregamento de flights, airlines e airports com amostragem reprodutível."""
from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_airlines(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_airports(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_flights_sample(
    path: Path,
    n: int = 400_000,
    seed: int = 42,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """
    Amostra aproximadamente proporcional ao longo do ficheiro (cobre vários meses).
    Para n=None, carrega o ficheiro completo (requer RAM elevada).
    """
    if n is None:
        return pd.read_csv(path, low_memory=False)

    # fração alvo para manter ~n linhas no total (~5.82M no dataset típico 2015)
    total_approx = 5_819_079
    frac = min(1.0, (n * 1.05) / total_approx)
    parts: list[pd.DataFrame] = []
    for i, chunk in enumerate(
        pd.read_csv(path, chunksize=chunksize, low_memory=False)
    ):
        take = max(1, int(len(chunk) * frac))
        take = min(take, len(chunk))
        parts.append(chunk.sample(n=take, random_state=seed + i))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


def load_flights_full(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)
