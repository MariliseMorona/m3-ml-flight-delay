# Atrasos de voos (EUA) — pipeline de ciência de dados

Projeto em Python que cobre um **pipeline completo** de ciência de dados sobre voos domésticos nos Estados Unidos (dataset tipo *Flight Delays and Cancellations*, ex. ano 2015): exploração (EDA), tratamento de ausentes, **modelação supervisionada** (classificação e regressão), **modelação não supervisionada** (PCA e clusterização) e **interpretação crítica** (conclusões, limitações, próximos passos).

---

## Índice

1. [O que o projeto faz](#o-que-o-projeto-faz)
2. [Dados necessários](#dados-necessários)
3. [Estrutura do repositório](#estrutura-do-repositório)
4. [Configuração central (`config.py`)](#configuração-central-configpy)
5. [Ambiente Python e dependências](#ambiente-python-e-dependências)
6. [Executar no Jupyter (local)](#executar-no-jupyter-local)
7. [Executar no Google Colab](#executar-no-google-colab)
8. [Regenerar os notebooks](#regenerar-os-notebooks)
9. [Scripts e testes rápidos](#scripts-e-testes-rápidos)
10. [Metodologia e requisitos académicos](#metodologia-e-requisitos-académicos)

---

## O que o projeto faz

| Fase | Conteúdo |
|------|-----------|
| **EDA** | Estatísticas descritivas, percentagem de ausentes por coluna, gráficos (atraso por mês/dia da semana, companhias, distribuição de `ARRIVAL_DELAY`, correlações). |
| **Dados** | Exclusão de cancelados/desviados na modelação; imputação mediana + one-hot no pré-processamento; códigos categóricos como texto (evita erros de tipo). |
| **Supervisionado — classificação** | Prever se o voo atrasa na chegada (> 15 min): **regressão logística** vs **Random Forest** (ROC-AUC, F1, acurácia, matriz de confusão, ROC). |
| **Supervisionado — regressão** | Prever minutos de `ARRIVAL_DELAY`: **Ridge** vs **Random Forest** (MAE, RMSE, R²). |
| **Não supervisionado** | **PCA** 2D em variáveis numéricas pré-voo; **K-Means** em perfis agregados por companhia aérea. |
| **Síntese** | Conclusões, limitações e melhorias sugeridas (no notebook). |

**Nota importante:** os preditores usados são apenas informação **disponível antes da partida** (sem `DEPARTURE_DELAY`, tempos reais de voo, etc.), para evitar *data leakage*.

---

## Dados necessários

Na pasta de dados (por defeito `data/raw/`) devem existir pelo menos:

| Ficheiro | Descrição |
|----------|-----------|
| `flights.csv` | Registos de voos (ficheiro grande, ~5,8M linhas / centenas de MB). |
| `airlines.csv` | Código IATA → nome da companhia. |
| `airports.csv` | Opcional no pipeline atual (pode ser usado para extensões / mapas). |

O dicionário de variáveis do `flights.csv` pode ser consultado no PDF `dicionario_dados_flights.pdf` (fornecido com o desafio).

**Copiar CSVs para o projeto:**

```bash
cp /caminho/para/flights.csv /caminho/para/airlines.csv /caminho/para/airports.csv ./data/raw/
```

Mais detalhes em [`data/raw/README.md`](data/raw/README.md).

---

## Estrutura do repositório

```
TechChallenge/
├── config.py                 # Caminhos, amostra, seed
├── requirements.txt
├── README.md
├── data/
│   └── raw/                  # flights.csv, airlines.csv, airports.csv
│       └── README.md
├── src/
│   ├── __init__.py
│   ├── load_data.py          # Carga + amostragem por chunks
│   ├── features.py           # Alvos e matriz X (sem leakage)
│   └── preprocess.py         # Pipeline sklearn (imputação, escala, OHE)
├── notebooks/
│   ├── pipeline_voos_eua.ipynb       # Jupyter local
│   └── pipeline_voos_eua_colab.ipynb # Google Colab
└── scripts/
    ├── build_notebook.py     # Gera os .ipynb a partir do código-fonte das células
    └── run_pipeline_smoke.py # Teste mínimo de classificação (CLI)
```

---

## Configuração central (`config.py`)

Todas as definições abaixo aplicam-se **tanto ao Jupyter como ao Colab**, desde que o notebook corra com o diretório de trabalho na raiz do projeto (no Colab isto é feito com `os.chdir(PROJECT_ROOT)` na primeira célula).

| Variável / opção | Descrição |
|------------------|-----------|
| `PROJECT_ROOT` | Pasta onde está `config.py` (calculada automaticamente). |
| `DATA_DIR` | Onde estão os CSVs. Por defeito: `PROJECT_ROOT / "data" / "raw"`. |
| `FLIGHTS_DATA_DIR` (ambiente) | Se definida, **substitui** o caminho acima (útil para dados noutro disco ou pasta). |
| `FLIGHTS_CSV`, `AIRLINES_CSV`, `AIRPORTS_CSV` | Caminhos completos derivados de `DATA_DIR`. |
| `DEFAULT_SAMPLE_SIZE` | Número de linhas lidas de `flights.csv` na amostragem por chunks. `None` = ficheiro completo (muita RAM). |
| `RANDOM_SEED` | Semente para reprodutibilidade (`42`). |

**Sugestões para `DEFAULT_SAMPLE_SIZE`:**

- Máquina com pouca RAM ou execução rápida: `150_000` – `250_000`.
- Equilíbrio (predefinição atual): `500_000`.
- Modelos mais estáveis (mais tempo/RAM): `600_000` – `800_000`.
- Dataset completo: `None` (apenas se tiver RAM suficiente).

**Variável de ambiente (Linux / macOS):**

```bash
export FLIGHTS_DATA_DIR="/caminho/para/pasta/com/csvs"
jupyter notebook notebooks/pipeline_voos_eua.ipynb
```

No Windows (PowerShell):

```powershell
$env:FLIGHTS_DATA_DIR = "C:\caminho\para\csvs"
```

---

## Ambiente Python e dependências

- **Python:** 3.11 ou superior (testado também com 3.14 em venv local).
- **RAM:** cerca de **2–4 GB** livres é um mínimo razoável para a amostra padrão; o CSV completo de voos exige muito mais.

Criar ambiente virtual e instalar pacotes:

```bash
cd TechChallenge
python3 -m venv .venv
source .venv/bin/activate          
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Pacotes principais:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `jupyter`, `ipykernel`.

---

## Executar no Jupyter (local)

### Pré-requisitos

1. Dependências instaladas (`requirements.txt`).
2. Ficheiros `flights.csv` e `airlines.csv` acessíveis em `data/raw/` **ou** `FLIGHTS_DATA_DIR` apontando para a pasta certa.
3. Abrir o notebook a partir da **raiz do projeto** ou de qualquer subpasta: o código procura `config.py` nos ascendentes de `Path.cwd()`.

### Passos

```bash
cd /caminho/para/TechChallenge
source .venv/bin/activate
jupyter notebook notebooks/pipeline_voos_eua.ipynb
```

Alternativas: **JupyterLab** ou **VS Code** com extensão Jupyter — abrir `notebooks/pipeline_voos_eua.ipynb` e selecionar o interpretador Python do `.venv`.

### Kernel

Registar o venv como kernel (opcional):

```bash
.venv/bin/python -m ipykernel install --user --name=techchallenge --display-name="Python (TechChallenge)"
```

Depois, no Jupyter, escolha o kernel **Python (TechChallenge)**.

### Ordem de execução

Use **Run All** ou execute as células de cima para baixo. Após alterar `config.py`, **reinicie o kernel** e volte a correr as células para recarregar `DEFAULT_SAMPLE_SIZE` e caminhos.

---

## Executar no Google Colab

O ficheiro `notebooks/pipeline_voos_eua_colab.ipynb` contém as **mesmas análises** que o notebook local, com células extra no início para o ambiente Colab.

### 1. Colocar o projeto acessível ao Colab

Escolha **uma** das opções:

- **Google Drive:** envie a pasta completa `TechChallenge` (com `config.py`, `src/`, `data/raw/` e os CSVs) para o Drive, por exemplo `Meu Drive/TechChallenge`.
- **ZIP em `/content`:** faça upload do ZIP, descomprima e garanta que existe `.../TechChallenge/config.py`.

### 2. Abrir o notebook no Colab

1. Aceda a [Google Colab](https://colab.research.google.com).
2. **Ficheiro → Carregar notebook** e selecione `pipeline_voos_eua_colab.ipynb`.

### 3. Ajustar o caminho do projeto

Na célula que monta o Drive e define `PROJECT_ROOT`, altere para o caminho **real** da pasta que contém `config.py`, por exemplo:

```python
PROJECT_ROOT = Path("/content/drive/MyDrive/TechChallenge")
```

Se descomprimiu em `/content`:

```python
PROJECT_ROOT = Path("/content/TechChallenge")
```

### 4. Executar as primeiras células (ordem obrigatória)

1. `%pip install -q ...` — instala dependências no runtime do Colab.
2. `drive.mount("/content/drive")` — autenticação Google (só se usar Drive).
3. Definição de `PROJECT_ROOT`, `sys.path` e `os.chdir(PROJECT_ROOT)` — o notebook importa `config` e `src` a partir daqui.

### 5. Dados e limites do Colab

- O `flights.csv` completo (~565 MB) pode caber no disco do Colab, mas o **upload pelo browser** é lento; o uso do **Drive** é o mais prático.
- Se faltar espaço ou RAM, **reduza** `DEFAULT_SAMPLE_SIZE` em `config.py` no Drive e volte a correr o notebook (ou edite temporariamente no Colab).
- Sessões Colab podem **expirar**; guarde cópias importantes no Drive.

### Diferença face ao Jupyter local

| Aspeto | Jupyter local | Colab |
|--------|----------------|--------|
| Instalação de pacotes | `pip` no venv | `%pip install` na primeira célula |
| Dados | `data/raw/` ou `FLIGHTS_DATA_DIR` | Drive ou `/content` + mesmo `config.py` |
| Raiz do projeto | Detetada via `config.py` | `PROJECT_ROOT` explícito + `chdir` |

---

## Regenerar os notebooks

O conteúdo das células está definido em `scripts/build_notebook.py`. Sempre que editar esse script, regenere os `.ipynb`:

```bash
source .venv/bin/activate
python scripts/build_notebook.py              # gera local + Colab
python scripts/build_notebook.py --local      # só pipeline_voos_eua.ipynb
python scripts/build_notebook.py --colab      # só pipeline_voos_eua_colab.ipynb
```

---

## Scripts e testes rápidos

### Teste mínimo (classificação, amostra pequena)

Valida imports, carga por amostra e um pipeline de regressão logística:

```bash
cd TechChallenge
source .venv/bin/activate
# Opcional: dados noutra pasta
export FLIGHTS_DATA_DIR="/caminho/para/csvs"
python scripts/run_pipeline_smoke.py
```

Por defeito, o smoke usa `FLIGHTS_DATA_DIR` se existir; caso contrário usa `data/raw/` via `config.py`.

---

## Metodologia e requisitos académicos

- **Classificação:** classe positiva = atraso na chegada **> 15 minutos** (`ARRIVAL_DELAY`).
- **Regressão:** alvo = `ARRIVAL_DELAY` em minutos (valores negativos = chegada antecipada).
- **Partição:** 80% treino / 20% teste, estratificada na classe.
- **Não supervisionado:** PCA para visualização; K-Means (k=4) em agregados por companhia.

Documentação adicional das colunas: **`dicionario_dados_flights.pdf`**.

---

## Resolução de problemas

| Problema | O que verificar |
|----------|------------------|
| `FileNotFoundError` nos CSVs | `data/raw/` ou `FLIGHTS_DATA_DIR`; nomes exatos `flights.csv` / `airlines.csv`. |
| Colab: não encontra `config.py` | `PROJECT_ROOT` correto; pasta contém `config.py` e `src/`. |
| Memória esgotada | Diminuir `DEFAULT_SAMPLE_SIZE` ou usar máquina com mais RAM. |
| Erro no `OneHotEncoder` / tipos mistos | Já tratado em `src/features.py` (códigos como `str`); atualize o projeto se tiver versão antiga. |
| `ModuleNotFoundError` | `pip install -r requirements.txt` e kernel Jupyter apontando para esse ambiente. |

---

## Licença e dados

Os dados de voos são usualmente disponibilizados para fins educacionais; confirme a licença da fonte original que utilizou. Este repositório contém apenas código e documentação do pipeline.
