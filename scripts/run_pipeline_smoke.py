"""Teste rápido do pipeline (amostra pequena)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

os.environ.setdefault("FLIGHTS_DATA_DIR", str(Path.home() / "Downloads"))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import AIRLINES_CSV, FLIGHTS_CSV, RANDOM_SEED
from src.features import build_model_frame, X_y_classification
from src.load_data import load_airlines, load_flights_sample
from src.preprocess import make_column_transformer

n = 60_000
df = load_flights_sample(FLIGHTS_CSV, n=n, seed=RANDOM_SEED)
_ = load_airlines(AIRLINES_CSV)
d = build_model_frame(df)
X, y = X_y_classification(d)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
pipe = Pipeline(
    [
        ("prep", make_column_transformer()),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)
pipe.fit(X_train, y_train)
proba = pipe.predict_proba(X_test)[:, 1]
pred = pipe.predict(X_test)
print("ROC-AUC:", roc_auc_score(y_test, proba))
print("F1:", f1_score(y_test, pred))
print("smoke ok")
