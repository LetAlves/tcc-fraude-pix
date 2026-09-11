"""
Pré-processamento e pipeline de modelagem para o baseline de junho.

Cobre: split temporal, imputação de nulos, encoding de categóricas,
escalonamento e montagem do pipeline (com SMOTE ou peso de classe,
em pipelines separados, nunca misturados — decisão registrada em
reports/anotacoes_metodologia.md, seção de junho).

Todo ajuste estatístico (mediana, frequências, escala) é feito somente
com o conjunto de treino, e reaplicado em validação/teste — nunca o
contrário, para não vazar informação de val/teste no ajuste.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as PipelineDesbalanceamento
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Colunas fora do conjunto de features ──────────────────────────────────
COLUNA_ALVO = "isFraud"
COLUNAS_DESCARTADAS = [
    "TransactionID",  # chave, não é feature
    "TransactionDT",  # usado só para o split temporal, não entra no modelo
    "card4",  # bandeira do cartão — sem equivalente conceitual em Pix (decisão registrada)
    "card6",  # crédito/débito — mesma razão acima
]

# Categorias com poucos valores distintos e visíveis (bandeira/tipo já removidos) — one-hot
COLUNAS_CATEGORICAS_BAIXA_CARDINALIDADE = [
    "ProductCD", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]


# ─── Codificador de frequência (para categóricas de alta cardinalidade) ────

class CodificadorFrequencia(BaseEstimator, TransformerMixin):
    """
    Substitui cada categoria pela sua frequência relativa observada no treino.

    Categorias nunca vistas no treino (inclusive nulos, se não tratados antes)
    recebem frequência 0.0 — sinaliza "raro/desconhecido" sem inventar uma
    categoria inexistente no treino.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "CodificadorFrequencia":
        X = pd.DataFrame(X)
        self.mapas_ = {coluna: X[coluna].value_counts(normalize=True) for coluna in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = pd.DataFrame(X).copy()
        for coluna in X.columns:
            X[coluna] = X[coluna].map(self.mapas_[coluna]).fillna(0.0)
        return X.to_numpy(dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        # Dentro de um Pipeline, este transformador recebe o array NumPy da etapa
        # de imputação — sem nomes de coluna, então `mapas_` fica indexado por
        # posição (0, 1, 2...). O sklearn repassa aqui os nomes reais da etapa
        # anterior via `input_features`; usá-los preserva "DeviceInfo" em vez de
        # "categoricas_alta__1" nos gráficos do SHAP.
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray(list(self.mapas_.keys()))


# ─── Redução de precisão numérica ───────────────────────────────────────────

def reduzir_precisao(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas numéricas para o menor tipo que as comporta sem perda útil.

    O IEEE-CIS é lido em `float64`/`int64` por padrão, mas nenhuma das colunas
    precisa dessa faixa: `TransactionAmt` não passa de dezenas de milhares e os
    grupos anônimos são contagens e indicadores pequenos. `float32` guarda cerca
    de sete algarismos significativos, muito acima da precisão dos dados.

    Aplicar **depois** da engenharia de features: as janelas móveis de
    `pix_features` acumulam somas ao longo de milhares de linhas e devem ser
    calculadas na precisão cheia; a redução serve ao caminho da modelagem, onde
    o custo de memória é o que limita o tamanho do experimento.

    Reduz o DataFrame completo de ~2,2 GB para ~1,2 GB e faz a matriz
    pré-processada de treino cair de ~1,5 GB para ~0,76 GB.
    """
    convertido = df.copy(deep=False)
    for coluna in df.columns:
        tipo = df[coluna].dtype
        if pd.api.types.is_float_dtype(tipo) and tipo.itemsize > 4:
            convertido[coluna] = df[coluna].astype("float32")
        elif pd.api.types.is_integer_dtype(tipo):
            convertido[coluna] = pd.to_numeric(df[coluna], downcast="integer")

    antes = df.memory_usage(deep=True).sum() / 1e6
    depois = convertido.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "Precisão reduzida — %.0f MB para %.0f MB (%.0f%% do original)",
        antes, depois, 100 * depois / antes,
    )
    return convertido


# ─── Identificação de colunas ───────────────────────────────────────────────

def identificar_colunas(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Separa as colunas de features em 3 grupos, por regra automática:
        - numéricas: dtype numérico, fora das listas de exceção;
        - categóricas de baixa cardinalidade: lista explícita (one-hot);
        - categóricas de alta cardinalidade: dtype texto (object), fora da
          lista de baixa cardinalidade (encoding por frequência).

    Não modifica nem lê `df` além dos dtypes/nomes de coluna.
    """
    colunas_features = [
        c for c in df.columns
        if c != COLUNA_ALVO and c not in COLUNAS_DESCARTADAS
    ]

    categoricas_baixa = [c for c in colunas_features if c in COLUNAS_CATEGORICAS_BAIXA_CARDINALIDADE]
    # Checa por tipo numérico (não por dtype == object): pandas recentes usam
    # um dtype "str" dedicado para texto em vez do "object" tradicional, então
    # "not numérico" é o teste robusto pra pegar texto em qualquer representação.
    numericas = [
        c for c in colunas_features
        if c not in categoricas_baixa and pd.api.types.is_numeric_dtype(df[c])
    ]
    categoricas_alta = [
        c for c in colunas_features
        if c not in categoricas_baixa and c not in numericas
    ]

    logger.info(
        "Colunas identificadas — numéricas: %d | categóricas baixa cardinalidade: %d | categóricas alta cardinalidade: %d",
        len(numericas), len(categoricas_baixa), len(categoricas_alta),
    )
    return numericas, categoricas_baixa, categoricas_alta


# ─── Split temporal ──────────────────────────────────────────────────────────

def dividir_temporal(
    df: pd.DataFrame,
    frac_treino: float = 0.7,
    frac_val: float = 0.15,
    coluna_tempo: str = "TransactionDT",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide o dataset em treino/validação/teste por corte temporal (não aleatório).

    Ordena por `coluna_tempo` e corta em blocos contíguos: treino = transações
    mais antigas, teste = mais recentes. Se uma posição nominal de corte cair
    dentro de um grupo com o mesmo timestamp, a fronteira avança até o fim do
    grupo. Assim, eventos simultâneos nunca são separados entre conjuntos.

    Esse deslocamento torna as proporções 70/15/15 aproximadas quando há empate
    na fronteira. A diferença é limitada ao tamanho do grupo deslocado e evita
    que o modelo "veja" no treino um instante também presente na validação.

    `frac_treino + frac_val` deve ser menor que 1; o restante vira teste.
    """
    if not 0.0 < frac_treino < 1.0:
        raise ValueError("frac_treino deve estar entre 0 e 1")
    if not 0.0 < frac_val < 1.0:
        raise ValueError("frac_val deve estar entre 0 e 1")
    if frac_treino + frac_val >= 1.0:
        raise ValueError("frac_treino + frac_val deve ser menor que 1 (sobra pro teste)")
    if coluna_tempo not in df.columns:
        raise ValueError(f"coluna temporal ausente: {coluna_tempo}")
    if df.empty:
        raise ValueError("não é possível dividir um dataset vazio")
    if df[coluna_tempo].isna().any():
        raise ValueError(f"{coluna_tempo} não pode conter valores ausentes")

    # mergesort é estável: entre linhas de mesmo timestamp (2,9% das linhas do
    # IEEE-CIS) preserva a ordem original, mantendo a fronteira do corte igual
    # entre execuções. O quicksort padrão do sort_values não garante isso.
    ordenado = df.sort_values(coluna_tempo, kind="mergesort").reset_index(drop=True)
    timestamps = ordenado[coluna_tempo].to_numpy()
    n = len(ordenado)
    corte_treino = int(n * frac_treino)
    corte_val = int(n * (frac_treino + frac_val))

    def avancar_ate_proximo_timestamp(corte: int) -> int:
        """Move uma fronteira interna até depois do grupo temporal empatado."""

        if corte <= 0 or corte >= n:
            return corte
        timestamp_anterior = timestamps[corte - 1]
        while corte < n and timestamps[corte] == timestamp_anterior:
            corte += 1
        return corte

    corte_treino = avancar_ate_proximo_timestamp(corte_treino)
    corte_val = avancar_ate_proximo_timestamp(corte_val)
    if corte_val <= corte_treino and corte_treino < n:
        # Se as duas posições nominais caírem no mesmo grupo temporal, reserve
        # ao menos o próximo grupo completo para validação.
        corte_val = avancar_ate_proximo_timestamp(corte_treino + 1)
    if not 0 < corte_treino < corte_val < n:
        raise ValueError(
            "não é possível formar treino, validação e teste sem dividir "
            f"grupos de {coluna_tempo}"
        )

    treino = ordenado.iloc[:corte_treino]
    val = ordenado.iloc[corte_treino:corte_val]
    teste = ordenado.iloc[corte_val:]

    logger.info(
        "Split temporal — treino: %d (%.1f%%) | val: %d (%.1f%%) | teste: %d (%.1f%%)",
        len(treino), 100 * len(treino) / n,
        len(val), 100 * len(val) / n,
        len(teste), 100 * len(teste) / n,
    )
    return treino, val, teste


# ─── Pipeline de pré-processamento ──────────────────────────────────────────

def construir_preprocessador(numericas: list[str], categoricas_baixa: list[str], categoricas_alta: list[str]) -> ColumnTransformer:
    """
    Monta o ColumnTransformer: imputação + transformação por grupo de coluna.

        - numéricas: mediana (imputação) + StandardScaler (escalonamento);
        - categóricas baixa cardinalidade: categoria explícita "ausente" + one-hot;
        - categóricas alta cardinalidade: categoria explícita "ausente" + frequência.

    A mediana, as categorias do one-hot e as frequências são todas aprendidas
    só no `fit` (ou seja, só com o conjunto de treino, ao usar dentro de um Pipeline).
    """
    pipeline_numerica = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    pipeline_categorica_baixa = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="ausente")),
        ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
    ])
    pipeline_categorica_alta = Pipeline([
        ("imputar", SimpleImputer(strategy="constant", fill_value="ausente")),
        ("codificar", CodificadorFrequencia()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numericas", pipeline_numerica, numericas),
            ("categoricas_baixa", pipeline_categorica_baixa, categoricas_baixa),
            ("categoricas_alta", pipeline_categorica_alta, categoricas_alta),
        ],
        remainder="drop",
    )


def montar_pipeline_modelo(
    preprocessador: ColumnTransformer,
    usar_smote: bool = False,
    class_weight: str | dict | None = None,
    random_state: int = 42,
) -> PipelineDesbalanceamento:
    """
    Monta o pipeline completo: pré-processamento (+ SMOTE opcional) + Regressão Logística.

    `usar_smote` e `class_weight` não devem ser combinados no mesmo pipeline
    (decisão registrada: comparar em pipelines separados, não misturar as
    duas estratégias de tratamento de desbalanceamento).

    O `preprocessador` recebido é clonado, não reaproveitado: o sklearn não
    clona as etapas de um Pipeline, então dois pipelines montados a partir do
    mesmo objeto compartilhariam o ajuste — o `fit` do segundo sobrescreveria
    silenciosamente a mediana, as categorias e a escala aprendidas pelo
    primeiro. Com o clone, cada pipeline aprende do seu próprio conjunto de
    treino e a comparação entre estratégias continua válida.
    """
    if usar_smote and class_weight is not None:
        raise ValueError("Não combinar SMOTE com class_weight — comparar em pipelines separados.")

    etapas = [("preprocessamento", clone(preprocessador))]
    if usar_smote:
        etapas.append(("smote", SMOTE(random_state=random_state)))
    etapas.append((
        "modelo",
        LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state),
    ))
    return PipelineDesbalanceamento(etapas)
