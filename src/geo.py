"""Utilitários geográficos: merge voos ↔ aeroportos para mapas de rotas e atrasos."""
from __future__ import annotations

import pandas as pd


AIRPORT_COLS = ["IATA_CODE", "LATITUDE", "LONGITUDE", "CITY", "STATE"]


def load_airports_geo(airports_path) -> pd.DataFrame:
    """Carrega airports.csv e retorna apenas as colunas geográficas necessárias."""
    df = pd.read_csv(airports_path)
    cols = [c for c in AIRPORT_COLS if c in df.columns]
    return df[cols].dropna(subset=["LATITUDE", "LONGITUDE"])


def airport_delay_stats(model_df: pd.DataFrame, airports_geo: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega atraso médio e taxa de atraso >15 min por aeroporto de ORIGEM.
    Faz merge com coordenadas lat/lon.
    """
    g = (
        model_df.groupby("ORIGIN_AIRPORT", as_index=False)
        .agg(
            delay_rate=("DELAYED_ARRIVAL", "mean"),
            mean_delay=("ARRIVAL_DELAY_MIN", "mean"),
            n_flights=("DELAYED_ARRIVAL", "size"),
        )
    )
    g["ORIGIN_AIRPORT"] = g["ORIGIN_AIRPORT"].astype(str)
    geo = airports_geo.copy()
    geo["IATA_CODE"] = geo["IATA_CODE"].astype(str)
    merged = g.merge(geo, left_on="ORIGIN_AIRPORT", right_on="IATA_CODE", how="inner")
    return merged


def build_route_stats(model_df: pd.DataFrame, airports_geo: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """
    Agrega atraso médio por rota (origin → destination) e faz merge com lat/lon de
    origem e destino. Retorna as top_n rotas com maior volume de voos.
    """
    g = (
        model_df.groupby(["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"], as_index=False)
        .agg(
            mean_delay=("ARRIVAL_DELAY_MIN", "mean"),
            delay_rate=("DELAYED_ARRIVAL", "mean"),
            n_flights=("DELAYED_ARRIVAL", "size"),
        )
        .nlargest(top_n, "n_flights")
    )
    for col in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        g[col] = g[col].astype(str)

    geo = airports_geo.copy()
    geo["IATA_CODE"] = geo["IATA_CODE"].astype(str)

    g = g.merge(
        geo[["IATA_CODE", "LATITUDE", "LONGITUDE"]].rename(
            columns={"IATA_CODE": "ORIGIN_AIRPORT", "LATITUDE": "orig_lat", "LONGITUDE": "orig_lon"}
        ),
        on="ORIGIN_AIRPORT",
        how="inner",
    )
    g = g.merge(
        geo[["IATA_CODE", "LATITUDE", "LONGITUDE"]].rename(
            columns={"IATA_CODE": "DESTINATION_AIRPORT", "LATITUDE": "dest_lat", "LONGITUDE": "dest_lon"}
        ),
        on="DESTINATION_AIRPORT",
        how="inner",
    )
    return g
