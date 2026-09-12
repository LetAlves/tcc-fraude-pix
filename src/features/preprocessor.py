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
    # As colunas de ponto flutuante são convertidas em uma única chamada. Atribuir
    # coluna a coluna em um DataFrame largo força o pandas a refragmentar e
    # reconsolidar os blocos a cada passo, e a consolidação do bloco float64
    # inteiro pede 1,76 GiB de uma vez no dataset completo — o suficiente para
    # abortar a execução quando a memória livre está apertada.
    tipos_float = {
        coluna: "float32"
        for coluna in df.columns
        if pd.api.types.is_float_dtype(df[coluna].dtype) and df[coluna].dtype.itemsize > 4
    }
    convertido = df.astype(tipos_float) if tipos_float else df.copy(deep=False)

    # Inteiros são poucas colunas (chave, tempo, alvo), então o laço aqui é barato.
    for coluna in convertido.columns:
        if pd.api.types.is_integer_dtype(convertido[coluna].dtype):
            convertido[coluna] = pd.to_numeric(convertido[coluna], downcast="integer")

    # deep=False: medir com deep=True percorre todas as strings das colunas de
    # texto, o que custa minutos no dataset completo. Como essas colunas não são
    # tocadas pela conversão, a diferença entre antes e depois é a mesma.
    antes = df.memory_usage(deep=False).sum() / 1e6
    depois = convertido.memory_usage(deep=False).sum() / 1e6
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
    mais antigas, teste = mais recentes. `frac_treino + frac_val` deve ser
    menor que 1; o restante vira teste.

    As frações são **aproximadas**: cada fronteira é empurrada até a próxima
    mudança de `coluna_tempo`, de modo que transações com o mesmo instante
    nunca caiam em conjuntos diferentes. O desvio máximo em relação à fração
    pedida é o tamanho do bloco de empate atravessado.

    Levanta `ValueError` se não for possível formar três períodos não vazios —
    o que acontece quando há poucos instantes distintos em relação às frações
    pedidas (no limite, um único instante para todas as linhas).
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

    # mergesort é estável: entre linhas de mesmo timestamp (5,7% das linhas do
    # IEEE-CIS) preserva a ordem original, mantendo a fronteira do corte igual
    # entre execuções. O quicksort padrão do sort_values não garante isso.
    ordenado = df.sort_values(coluna_tempo, kind="mergesort").reset_index(drop=True)
    n = len(ordenado)
    tempos = ordenado[coluna_tempo].to_numpy()

    def avancar_fronteira(posicao: int) -> int:
        """
        Empurra o corte até a próxima mudança de instante.

        Transações que compartilham `coluna_tempo` formam um bloco indivisível:
        cortar no meio dele colocaria eventos simultâneos em conjuntos
        diferentes, o que é vazamento temporal — o modelo veria, no treino,
        eventos do mesmo instante que precisa prever. O bloco inteiro fica à
        esquerda da fronteira.
        """
        if posicao <= 0:
            return 0
        if posicao >= n:
            return n
        return int(np.searchsorted(tempos, tempos[posicao - 1], side="right"))

    corte_treino = avancar_fronteira(int(n * frac_treino))
    corte_val = max(avancar_fronteira(int(n * (frac_treino + frac_val))), corte_treino)
    if corte_val == corte_treino and corte_treino < n:
        # Se as duas posições nominais caírem no mesmo bloco, o próximo grupo
        # temporal completo forma a validação, desde que ainda reste um teste.
        corte_val = avancar_fronteira(corte_treino + 1)

    treino = ordenado.iloc[:corte_treino]
    val = ordenado.iloc[corte_treino:corte_val]
    teste = ordenado.iloc[corte_val:]

    if min(len(treino), len(val), len(teste)) == 0:
        raise ValueError(
            f"Não foi possível formar três períodos não vazios: {n} linhas com "
            f"apenas {len(np.unique(tempos))} instantes distintos. Empurrar os "
            f"cortes até a próxima mudança de '{coluna_tempo}' esvazia ao menos "
            f"um conjunto. Ajuste as frações ou use um recorte com mais instantes."
        )

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
