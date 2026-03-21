"""Caminhos dos dados — ajuste ou use variável de ambiente FLIGHTS_DATA_DIR."""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FLIGHTS_DATA_DIR", PROJECT_ROOT / "data" / "raw"))

FLIGHTS_CSV = DATA_DIR / "flights.csv"
AIRLINES_CSV = DATA_DIR / "airlines.csv"
AIRPORTS_CSV = DATA_DIR / "airports.csv"

# Amostra padrão: ~5.8M linhas no ficheiro completo.
# - Reduza (ex.: 150_000–250_000) para notebooks mais rápidos em máquinas com pouca RAM.
# - Aumente (ex.: 600_000–800_000) para estimativas mais estáveis (mais RAM/tempo).
# - None = carregar todas as linhas (só com RAM suficiente).
DEFAULT_SAMPLE_SIZE = 500_000
RANDOM_SEED = 42
