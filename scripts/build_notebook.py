"""Gera notebooks Jupyter e Google Colab a partir das mesmas células.

Uso:
  python scripts/build_notebook.py                 # ambos (.ipynb local + Colab)
  python scripts/build_notebook.py --local       # só Jupyter local
  python scripts/build_notebook.py --colab       # só variante Colab
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJ = Path(__file__).resolve().parents[1]
NB_LOCAL = PROJ / "notebooks" / "pipeline_voos_eua.ipynb"
NB_COLAB = PROJ / "notebooks" / "pipeline_voos_eua_colab.ipynb"


def build_cells(*, colab: bool) -> list:
    cells: list = []

    def md(s: str) -> None:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})

    def code(s: str) -> None:
        cells.append({"cell_type": "code", "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)})

    if colab:
        md(
            """# Google Colab / Jupyter — pipeline de atrasos de voos

Este ficheiro corre **no Google Colab** (monta o Drive) e também **no Jupyter ou PyCharm local**: se não existir `google.colab`, a célula seguinte **não** monta o Drive e usa a pasta do projeto onde estiver o `config.py`.

1. **Colab:** envie o projeto para o Drive (`config.py`, `src/`, `data/raw/*.csv`) e ajuste `PROJECT_ROOT` se não for `MyDrive/TechChallenge`.
2. **Local:** prefira `pipeline_voos_eua.ipynb`; se usar este ficheiro, abra-o com a **raiz do projeto** como diretório de trabalho ou deixe a deteção automática.
3. `flights.csv` é grande (~565 MB): use amostra em `config.py` (`DEFAULT_SAMPLE_SIZE`) se faltar RAM."""
        )
        code(
            r"""# Dependências no runtime do Colab
%pip install -q pandas numpy scikit-learn matplotlib seaborn"""
        )
        code(
            r"""from pathlib import Path
import os
import sys

try:
    from google.colab import drive

    IN_COLAB = True
except ModuleNotFoundError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")
    # Ajuste se a pasta TechChallenge no Drive for noutro sítio:
    PROJECT_ROOT = Path("/content/drive/MyDrive/TechChallenge")
    # Alternativa: PROJECT_ROOT = Path("/content/TechChallenge")
else:
    # Jupyter / PyCharm / VS Code: localizar pasta que contém config.py
    PROJECT_ROOT = next(
        p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists()
    )
    print("Ambiente local (sem google.colab). PROJECT_ROOT detetado automaticamente.")

if not (PROJECT_ROOT / "config.py").exists():
    raise FileNotFoundError(
        f"Não encontrei config.py em {PROJECT_ROOT}. "
        "No Colab, ajuste PROJECT_ROOT após drive.mount. No local, abra o notebook a partir da pasta TechChallenge."
    )

sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)
print("PROJECT_ROOT:", PROJECT_ROOT.resolve(), "| Colab:", IN_COLAB)"""
        )

    md(
        """# Pipeline de ciência de dados — atrasos de voos (EUA)

**Objetivos:** explorar o dataset de voos domésticos (2015), tratar dados ausentes, treinar modelos **supervisionados** (classificação de atraso na chegada e regressão do tempo de atraso) e **não supervisionados** (PCA e clusterização por companhia), com interpretação crítica.

**Nota metodológica:** os preditores usados correspondem apenas a informação **disponível antes da partida** (sem `DEPARTURE_DELAY`, tempos reais de voo, etc.), para evitar *data leakage*."""
    )

    if colab:
        code(
            r"""import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AIRLINES_CSV, DEFAULT_SAMPLE_SIZE, FLIGHTS_CSV, RANDOM_SEED
from src.features import build_model_frame, X_y_classification
from src.load_data import load_airlines, load_flights_sample
from src.preprocess import make_column_transformer

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 5)
np.random.seed(RANDOM_SEED)

SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE
print("ROOT:", ROOT)
print("Amostra:", SAMPLE_SIZE)"""
        )
    else:
        code(
            r"""import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "config.py").exists())
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import AIRLINES_CSV, DEFAULT_SAMPLE_SIZE, FLIGHTS_CSV, RANDOM_SEED
from src.features import build_model_frame, X_y_classification
from src.load_data import load_airlines, load_flights_sample
from src.preprocess import make_column_transformer

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 5)
np.random.seed(RANDOM_SEED)

SAMPLE_SIZE = DEFAULT_SAMPLE_SIZE
print("ROOT:", ROOT)
print("Amostra:", SAMPLE_SIZE)"""
        )

    code(
        r"""airlines = load_airlines(AIRLINES_CSV)
flights_raw = load_flights_sample(FLIGHTS_CSV, n=SAMPLE_SIZE, seed=RANDOM_SEED)
flights_raw["AIRLINE"] = flights_raw["AIRLINE"].astype(str)

flights_eda = flights_raw.merge(
    airlines,
    left_on="AIRLINE",
    right_on="IATA_CODE",
    how="left",
    suffixes=("", "_full"),
)
flights_eda["AIRLINE_NAME"] = flights_eda["AIRLINE_full"].fillna(flights_eda["AIRLINE"])

print(flights_raw.shape)
flights_raw.head()"""
    )

    md(
        """## 1. Exploração dos dados (EDA)

### 1.1 Estatísticas descritivas e dados ausentes

Colunas de atraso atribuível (`AIR_SYSTEM_DELAY`, etc.) só têm valores quando há decomposição do atraso — não são usadas como preditores (evitam *leakage*). `TAIL_NUMBER` ausente não entra no modelo."""
    )

    code(
        r"""desc_num = flights_raw.select_dtypes(include=[np.number]).describe().T
desc_num.round(2)"""
    )

    code(
        r"""missing_pct = (flights_raw.isna().mean() * 100).sort_values(ascending=False)
missing_pct = missing_pct[missing_pct > 0]
ax = missing_pct.head(20).plot(kind="barh", color="steelblue", figsize=(8, 6))
ax.set_xlabel("% ausente")
ax.set_title("Top colunas com valores ausentes")
plt.tight_layout()
plt.show()
print("Colunas com >0% ausente (amostra):\n", missing_pct.head(15).to_string())"""
    )

    md("### 1.2 Visualizações")

    code(
        r"""oper = flights_raw[flights_raw["CANCELLED"] == 0].copy()
oper["DELAYED_15"] = (oper["ARRIVAL_DELAY"] > 15).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
oper.groupby("MONTH")["DELAYED_15"].mean().plot(ax=axes[0], marker="o", color="darkred")
axes[0].set_title("Taxa de atraso na chegada (>15 min) por mês")
axes[0].set_ylabel("Proporção")
axes[0].set_xlabel("Mês")
oper.groupby("DAY_OF_WEEK")["DELAYED_15"].mean().plot(ax=axes[1], marker="o", color="teal")
axes[1].set_title("Taxa de atraso por dia da semana (1=Seg)")
axes[1].set_xlabel("Dia da semana")
plt.tight_layout()
plt.show()"""
    )

    code(
        r"""top_air = (
    oper.groupby("AIRLINE", as_index=False)
    .agg(delay_rate=("DELAYED_15", "mean"), n=("DELAYED_15", "size"))
    .query("n >= 500")
    .sort_values("delay_rate", ascending=False)
    .head(12)
)
top_air = top_air.merge(
    airlines.rename(columns={"AIRLINE": "AIRLINE_NAME"}),
    left_on="AIRLINE",
    right_on="IATA_CODE",
    how="left",
)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=top_air, y="AIRLINE", x="delay_rate", hue="AIRLINE", palette="RdYlGn_r", ax=ax, legend=False)
ax.set_title("Taxa de atraso (>15 min) — companhias com volume na amostra")
ax.set_xlabel("Taxa")
plt.tight_layout()
plt.show()"""
    )

    code(
        r"""sample_delay = oper["ARRIVAL_DELAY"].dropna()
sample_delay = sample_delay[(sample_delay > -60) & (sample_delay < 180)]
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(sample_delay, bins=60, kde=True, ax=ax, color="slateblue")
ax.set_title("Distribuição do atraso na chegada (min) — voos não cancelados")
ax.set_xlabel("ARRIVAL_DELAY (min)")
plt.tight_layout()
plt.show()"""
    )

    code(
        r"""corr_cols = ["MONTH", "DAY_OF_WEEK", "SCHEDULED_TIME", "DISTANCE", "ARRIVAL_DELAY"]
c = oper[corr_cols].dropna().corr()
plt.figure(figsize=(6, 5))
sns.heatmap(c, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlação linear (subset — EDA)")
plt.tight_layout()
plt.show()"""
    )

    md(
        """### 1.3 Tratamento de ausentes (modelagem)

- **Cancelados / desviados:** excluídos.
- **Numéricas:** imputação pela **mediana** no pré-processamento.
- **Categóricas:** `OneHotEncoder(handle_unknown='ignore')`."""
    )

    md(
        """## 2. Modelagem supervisionada

**Classificação:** `ARRIVAL_DELAY > 15` min. **Regressão:** `ARRIVAL_DELAY` (min). **Split:** 80/20, estratificado na classe."""
    )

    code(
        r"""model_df = build_model_frame(flights_raw)
X_c, y_c = X_y_classification(model_df)
y_r_series = model_df.loc[X_c.index, "ARRIVAL_DELAY_MIN"]

X_train_c, X_test_c, y_train_c, y_test_c, y_train_r, y_test_r = train_test_split(
    X_c, y_c, y_r_series, test_size=0.2, random_state=RANDOM_SEED, stratify=y_c
)
X_train_r, X_test_r = X_train_c, X_test_c

print(model_df.shape, "linhas após filtros")
print("Proporção atrasados:", round(float(np.mean(y_c)), 3))"""
    )

    md("### 2.1 Classificação — Regressão logística vs Random Forest")

    code(
        r"""prep = make_column_transformer(max_categories=40)

clf_lr = Pipeline(
    [
        ("prep", prep),
        (
            "model",
            LogisticRegression(
                max_iter=2500,
                solver="saga",
                class_weight="balanced",
                random_state=RANDOM_SEED,
            ),
        ),
    ]
)
clf_rf = Pipeline(
    [
        ("prep", make_column_transformer(max_categories=40)),
        (
            "model",
            RandomForestClassifier(
                n_estimators=120,
                max_depth=16,
                min_samples_leaf=50,
                class_weight="balanced_subsample",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ]
)

clf_lr.fit(X_train_c, y_train_c)
clf_rf.fit(X_train_c, y_train_c)


def eval_clf(name, pipe):
    proba = pipe.predict_proba(X_test_c)[:, 1]
    pred = pipe.predict(X_test_c)
    return {
        "modelo": name,
        "ROC-AUC": roc_auc_score(y_test_c, proba),
        "F1": f1_score(y_test_c, pred),
        "Accuracy": accuracy_score(y_test_c, pred),
    }


metrics_clf = pd.DataFrame([eval_clf("Regressão logística", clf_lr), eval_clf("Random Forest", clf_rf)])
display(metrics_clf)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, pipe) in zip(
    axes,
    [("Regressão logística", clf_lr), ("Random Forest", clf_rf)],
):
    proba = pipe.predict_proba(X_test_c)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_c, proba)
    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test_c, proba):.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(name)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend(loc="lower right")
plt.suptitle("Curvas ROC — atraso na chegada (>15 min)")
plt.tight_layout()
plt.show()

ConfusionMatrixDisplay.from_estimator(clf_rf, X_test_c, y_test_c, cmap="Blues")
plt.title("Matriz de confusão — Random Forest")
plt.show()
print(classification_report(y_test_c, clf_rf.predict(X_test_c), digits=3))"""
    )

    md("### 2.2 Regressão — Ridge vs Random Forest (MAE, RMSE, R²)")

    code(
        r"""reg_ridge = Pipeline(
    [
        ("prep", make_column_transformer(max_categories=40)),
        ("model", Ridge(alpha=2.0, random_state=RANDOM_SEED)),
    ]
)
reg_rf = Pipeline(
    [
        ("prep", make_column_transformer(max_categories=40)),
        (
            "model",
            RandomForestRegressor(
                n_estimators=120,
                max_depth=16,
                min_samples_leaf=50,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
        ),
    ]
)

reg_ridge.fit(X_train_r, y_train_r)
reg_rf.fit(X_train_r, y_train_r)


def eval_reg(name, pipe):
    pred = pipe.predict(X_test_r)
    return {
        "modelo": name,
        "MAE": mean_absolute_error(y_test_r, pred),
        "RMSE": root_mean_squared_error(y_test_r, pred),
        "R2": r2_score(y_test_r, pred),
    }


display(pd.DataFrame([eval_reg("Ridge", reg_ridge), eval_reg("Random Forest", reg_rf)]))

fig, ax = plt.subplots(figsize=(5, 5))
pred = reg_rf.predict(X_test_r)
yt = y_test_r.values if hasattr(y_test_r, "values") else np.asarray(y_test_r)
sns.scatterplot(x=yt[:8000], y=pred[:8000], alpha=0.15, ax=ax)
ax.plot([-100, 200], [-100, 200], "r--", lw=1)
ax.set_xlabel("Atraso real (min)")
ax.set_ylabel("Atraso previsto (min)")
ax.set_title("RF regressão: real vs previsto (subamostra)")
plt.tight_layout()
plt.show()"""
    )

    md(
        """## 3. Modelagem não supervisionada

### 3.1 PCA — projeção 2D (variáveis numéricas pré-voo), cor = atrasado."""
    )

    code(
        r"""num_cols = ["MONTH", "DAY_OF_WEEK", "DEP_MIN", "ARR_MIN", "SCHEDULED_TIME", "DISTANCE"]
pca_df = model_df.dropna(subset=num_cols).sample(n=min(12000, len(model_df)), random_state=RANDOM_SEED)
Xn_s = StandardScaler().fit_transform(pca_df[num_cols].values)
pca = PCA(n_components=2, random_state=RANDOM_SEED)
Z = pca.fit_transform(Xn_s)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    Z[:, 0],
    Z[:, 1],
    c=pca_df["DELAYED_ARRIVAL"].values,
    cmap="coolwarm",
    alpha=0.35,
    s=8,
)
plt.colorbar(scatter, label="Atrasado (>15 min)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
plt.title("PCA (numéricas pré-voo)")
plt.tight_layout()
plt.show()
print("Variância explicada:", pca.explained_variance_ratio_.round(4))"""
    )

    md("### 3.2 K-Means — perfis agregados por companhia")

    code(
        r"""airlines_feat = airlines.rename(columns={"AIRLINE": "AIRLINE_NAME"})

g = model_df.groupby("AIRLINE", as_index=False).agg(
    delay_rate=("DELAYED_ARRIVAL", "mean"),
    mean_delay=("ARRIVAL_DELAY_MIN", "mean"),
    n_flights=("DELAYED_ARRIVAL", "size"),
    mean_dist=("DISTANCE", "mean"),
)
g["AIRLINE"] = g["AIRLINE"].astype(str)
feat_s = StandardScaler().fit_transform(g[["delay_rate", "mean_delay", "n_flights", "mean_dist"]].values)
km = KMeans(n_clusters=4, random_state=RANDOM_SEED, n_init=10)
g["cluster"] = km.fit_predict(feat_s)
g_lbl = g.merge(airlines_feat, left_on="AIRLINE", right_on="IATA_CODE", how="left")

plt.figure(figsize=(8, 6))
sns.scatterplot(data=g_lbl, x="delay_rate", y="mean_delay", hue="cluster", palette="Set2", s=140)
for _, r in g_lbl.iterrows():
    plt.annotate(r["AIRLINE"], (r["delay_rate"], r["mean_delay"]), fontsize=7, alpha=0.9)
plt.xlabel("Taxa atraso >15 min")
plt.ylabel("Atraso médio na chegada (min)")
plt.title("Clusters de companhias (K-Means, k=4)")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.show()

display(
    g_lbl.sort_values("cluster")[
        ["AIRLINE", "AIRLINE_NAME", "delay_rate", "mean_delay", "n_flights", "cluster"]
    ]
)"""
    )

    md(
        """## 4. Conclusões, limitações e próximos passos

### Principais conclusões
- A taxa de atraso varia com **mês**, **dia da semana** e **companhia**, visível no EDA e nos clusters.
- Com apenas informação pré-voo, a **classificação** costuma ter ROC-AUC moderado (o problema é intrinsecamente ruidoso); **Random Forest** tende a captar não linearidades vs regressão logística.
- A **regressão** do valor exato do atraso é mais difícil: R² baixo é esperado; MAE/RMSE em minutos permitem comparar algoritmos.
- **PCA** mostra sobreposição entre classes: os atrasos não são linearmente separáveis neste subespaço.
- **K-Means** nos perfis de companhia agrupa operadores com comportamentos semelhantes de pontualidade e magnitude de atraso (interpretável com o nome da companhia).

### Limitações
- **Amostra** parcial do ficheiro total para viabilizar RAM/tempo; estimativas têm variância de amostragem.
- **Definição de atraso** (>15 min) é arbitrária; outros limiares mudam a prevalência.
- **Regressão** usa também voos adiantados (atraso negativo); alternativa seria modelar só `max(0, delay)`.
- **Sazonalidade e eventos** (clima extremo, greves) não estão explicitamente como features.
- **Cardinalidade** de aeroportos: `max_categories` no one-hot agrupa categorias pouco frequentes.

### Melhorias
- Incluir **histórico** (atraso médio da rota/companhia em janelas anteriores) com validação temporal rigorosa.
- **Calibração** de probabilidades (Platt / isotónica) para decisões operacionais.
- Modelos de **séries tempora** ou **gradient boosting** (XGBoost/LightGBM) com *early stopping*.
- **Validação por tempo** (treino em meses anteriores, teste no mês seguinte) em vez de split aleatório.
- Enriquecer com **dados meteorológicos** ou **capacidade** de aeroporto."""
    )

    return cells


def write_ipynb(path: Path, cells: list, *, colab_meta: bool) -> None:
    meta = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    if colab_meta:
        meta["colab"] = {"provenance": []}

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": meta,
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Escrito:", path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera notebook(s) do pipeline de voos.")
    ap.add_argument("--colab", action="store_true", help="Gera só pipeline_voos_eua_colab.ipynb")
    ap.add_argument("--local", action="store_true", help="Gera só pipeline_voos_eua.ipynb")
    args = ap.parse_args()

    gen_local = True
    gen_colab = True
    if args.colab or args.local:
        if args.colab and not args.local:
            gen_local, gen_colab = False, True
        elif args.local and not args.colab:
            gen_local, gen_colab = True, False

    if gen_local:
        write_ipynb(NB_LOCAL, build_cells(colab=False), colab_meta=False)
    if gen_colab:
        write_ipynb(NB_COLAB, build_cells(colab=True), colab_meta=True)


if __name__ == "__main__":
    main()
