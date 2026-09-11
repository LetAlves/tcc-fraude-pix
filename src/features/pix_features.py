"""
Engenharia de features "Pix" — atributos derivados por analogia ao IEEE-CIS.

Implementa os 4 atributos derivados validados na ata de mapeamento de
16/08/2026 (reports/reunioes/2026-08-16_mapeamento_ieee_cis_pix.md), que por
sua vez seguem as ressalvas documentadas em
reports/pessoa_2/maio/05_mapeamento_e_kickoff.md (seção 4):

    - valor_atipico_proxy
    - frequencia_recente_proxy
    - dispositivo_raro_proxy
    - posicao_ciclo_diario_relativa

Regra de linguagem (decisão da dupla): cada atributo é um "papel analítico
análogo" a um sinal de risco Pix, nunca uma equivalência direta. `card1` é
um identificador mascarado usado como proxy de histórico — não é uma conta
ou chave Pix real.

Todos os atributos são causais: usam exclusivamente transações anteriores
no tempo (TransactionDT) para o mesmo card1, evitando vazamento de dados
(data leakage) do futuro para o passado.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEGUNDOS_POR_DIA = 86400


# ─── Auxiliar causal ────────────────────────────────────────────────────────

def _mediana_expandida_causal(
    serie_ordenada: pd.Series,
    chave_grupo: pd.Series,
    tempo_ordenado: pd.Series,
) -> pd.Series:
    """
    Mediana expandida por grupo, usando só observações estritamente anteriores.

    Pré-requisito: as séries já devem estar ordenadas por grupo e tempo.
    Linhas com o mesmo instante recebem a mesma estatística, calculada antes
    daquele instante; nenhuma delas é tratada como histórico da outra.
    """
    mediana = serie_ordenada.groupby(chave_grupo).expanding().median()
    mediana = mediana.reset_index(level=0, drop=True)
    mediana_anterior = mediana.groupby(chave_grupo).shift(1)
    grupos_temporais = pd.DataFrame(
        {
            "_grupo": chave_grupo,
            "_tempo": tempo_ordenado,
        },
        index=serie_ordenada.index,
    )
    primeira_linha = ~grupos_temporais.duplicated()
    valor_antes_do_instante = mediana_anterior.where(primeira_linha)
    return valor_antes_do_instante.groupby(
        [chave_grupo, tempo_ordenado]
    ).transform("max")


# ─── Atributos derivados ────────────────────────────────────────────────────

def valor_atipico_proxy(
    df: pd.DataFrame,
    coluna_valor: str = "TransactionAmt",
    coluna_cartao: str = "card1",
    coluna_tempo: str = "TransactionDT",
) -> pd.Series:
    """
    Desvio robusto do valor da transação frente ao histórico anterior do mesmo card1.

    Fórmula: z = (valor - mediana_anterior) / (mad_anterior * 1.4826 + eps)
        mediana_anterior e mad_anterior usam só transações do mesmo card1
        com TransactionDT estritamente menor que a linha atual (1.4826 é o
        fator que torna o MAD comparável a um desvio-padrão sob normalidade).
    Colunas-fonte: TransactionAmt, card1, TransactionDT.
    População: transações com card1 não nulo.
    Janela: expandida (todo o histórico anterior disponível na partição carregada).
    Unidade: adimensional (z-score robusto).
    Nulos: NaN nas primeiras transações de cada card1 (sem histórico anterior).
    Risco de vazamento: nenhum — só usa observações anteriores no tempo.
    Ressalva (ata 16/08/2026): card1 é um proxy mascarado, não uma conta Pix real.
    """
    ordenado = df.sort_values(
        [coluna_cartao, coluna_tempo],
        kind="mergesort",
        na_position="last",
    )
    mediana_anterior = _mediana_expandida_causal(
        ordenado[coluna_valor],
        ordenado[coluna_cartao],
        ordenado[coluna_tempo],
    )
    desvio_abs = (ordenado[coluna_valor] - mediana_anterior).abs()
    mad_anterior = _mediana_expandida_causal(
        desvio_abs,
        ordenado[coluna_cartao],
        ordenado[coluna_tempo],
    )

    eps = 1e-6
    z = (ordenado[coluna_valor] - mediana_anterior) / (mad_anterior * 1.4826 + eps)
    # Winsorização em ±30: quando o MAD anterior é ~0 (cartão com histórico
    # pouco variado), a divisão explode numericamente sem agregar sinal real
    # além de "muito atípico" — limitar preserva esse sinal sem gerar
    # magnitudes absurdas (ex.: bilhões) que dominariam a normalização depois.
    z = z.clip(lower=-30, upper=30)
    return z.reindex(df.index).rename("valor_atipico_proxy")


def frequencia_recente_proxy(
    df: pd.DataFrame,
    coluna_cartao: str = "card1",
    coluna_tempo: str = "TransactionDT",
    janela_segundos: int = SEGUNDOS_POR_DIA,
) -> pd.Series:
    """
    Contagem de transações anteriores do mesmo card1 dentro de uma janela de tempo relativa.

    Fórmula: contagem de linhas do mesmo card1 com TransactionDT no
        intervalo (t - janela_segundos, t), estritamente anterior a t.
    Colunas-fonte: card1, TransactionDT.
    População: transações com card1 não nulo.
    Janela: `janela_segundos` (padrão 86400s). TransactionDT não tem
        fuso/origem conhecidos — "86400s" é a unidade do relógio relativo do
        dataset, não necessariamente um dia civil.
    Unidade: contagem inteira (>= 0).
    Nulos: NaN quando card1 ou TransactionDT está ausente; 0 quando existe
        identificador válido, mas não há evento estritamente anterior.
    Risco de vazamento: nenhum — o intervalo aberto (t - janela, t) exclui
        a própria linha, eventos do mesmo instante e qualquer evento futuro.
    """
    if janela_segundos <= 0:
        raise ValueError("janela_segundos deve ser maior que zero")

    resultado = pd.Series(np.nan, index=df.index, dtype="float64")
    validos = df[coluna_cartao].notna() & df[coluna_tempo].notna()
    ordenado = df.loc[validos, [coluna_cartao, coluna_tempo]].sort_values(
        [coluna_cartao, coluna_tempo],
        kind="mergesort",
    )
    for _, grupo in ordenado.groupby(coluna_cartao, sort=False):
        tempos = grupo[coluna_tempo].to_numpy()
        inicio = np.searchsorted(tempos, tempos - janela_segundos, side="right")
        fim = np.searchsorted(tempos, tempos, side="left")
        resultado.loc[grupo.index] = (fim - inicio).astype("float64")

    return resultado.rename("frequencia_recente_proxy")


def dispositivo_raro_proxy(
    df: pd.DataFrame,
    coluna_cartao: str = "card1",
    coluna_dispositivo: str = "DeviceInfo",
    coluna_tempo: str = "TransactionDT",
) -> pd.Series:
    """
    Raridade do DeviceInfo no histórico anterior do mesmo card1.

    Fórmula: raridade = 1 / (contagem_anterior_do_par(card1, DeviceInfo) + 1).
        Próximo de 1 = dispositivo nunca visto antes nesse card1;
        próximo de 0 = dispositivo já visto muitas vezes.
    Colunas-fonte: card1, DeviceInfo, TransactionDT.
    População: apenas transações com card1 e DeviceInfo não nulos (~24,4% do
        treino têm dado de identidade — ver reports/pessoa_2/maio/05_mapeamento_e_kickoff.md).
    Janela: expandida, causal (cumcount após ordenar por TransactionDT).
    Unidade: adimensional, em (0, 1].
    Nulos: NaN quando card1, DeviceInfo ou TransactionDT está ausente;
        o tratamento de imputação pertence ao pré-processador.
    Risco de vazamento: nenhum — a contagem usa apenas ocorrências em
        instantes estritamente anteriores, excluindo empates temporais.
    """
    resultado = pd.Series(np.nan, index=df.index, dtype="float64")
    validos = (
        df[coluna_cartao].notna()
        & df[coluna_dispositivo].notna()
        & df[coluna_tempo].notna()
    )
    historico = df.loc[
        validos,
        [coluna_cartao, coluna_dispositivo, coluna_tempo],
    ]

    contagens_por_instante = historico.groupby(
        [coluna_cartao, coluna_dispositivo, coluna_tempo],
        sort=True,
    ).size()
    contagens_anteriores = (
        contagens_por_instante.groupby(level=[0, 1]).cumsum()
        - contagens_por_instante
    )
    chaves = pd.MultiIndex.from_frame(
        historico[[coluna_cartao, coluna_dispositivo, coluna_tempo]]
    )
    anteriores_por_linha = contagens_anteriores.reindex(chaves).to_numpy()
    resultado.loc[historico.index] = 1.0 / (anteriores_por_linha + 1.0)

    return resultado.rename("dispositivo_raro_proxy")


def posicao_ciclo_diario_relativa(
    df: pd.DataFrame,
    coluna_tempo: str = "TransactionDT",
) -> pd.DataFrame:
    """
    Posição cíclica de TransactionDT dentro de um período de 86400s.

    Fórmula: posicao = (TransactionDT mod 86400) / 86400, em [0, 1).
        Também retorna a codificação seno/cosseno (2*pi*posicao) para o
        modelo não enxergar uma descontinuidade artificial na virada do ciclo.
    Colunas-fonte: TransactionDT.
    População: todas as transações.
    Janela: não aplicável (transformação por linha, sem histórico).
    Unidade: posicao em [0, 1); seno/cosseno em [-1, 1].
    Nulos: nenhum esperado (TransactionDT não tem nulos no dataset).
    Risco de vazamento: nenhum — depende só da própria linha.
    Ressalva (ata 16/08/2026): NÃO é "horário local" — a origem e o fuso de
        TransactionDT não são publicados; é só uma posição cíclica relativa.
    """
    posicao = (df[coluna_tempo] % SEGUNDOS_POR_DIA) / SEGUNDOS_POR_DIA
    angulo = 2 * np.pi * posicao
    return pd.DataFrame(
        {
            "posicao_ciclo_diario_relativa": posicao,
            "posicao_ciclo_diario_sen": np.sin(angulo),
            "posicao_ciclo_diario_cos": np.cos(angulo),
        },
        index=df.index,
    )


# ─── Função principal ───────────────────────────────────────────────────────

def criar_features_pix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona os 4 atributos derivados (mais seno/cosseno do ciclo diário) a uma cópia de `df`.

    Não modifica `df` no lugar — retorna um novo DataFrame.
    """
    logger.info("Calculando valor_atipico_proxy...")
    valor_atipico = valor_atipico_proxy(df)

    logger.info("Calculando frequencia_recente_proxy...")
    frequencia_recente = frequencia_recente_proxy(df)

    logger.info("Calculando dispositivo_raro_proxy...")
    dispositivo_raro = dispositivo_raro_proxy(df)

    logger.info("Calculando posicao_ciclo_diario_relativa...")
    ciclo_diario = posicao_ciclo_diario_relativa(df)

    df_com_features = pd.concat(
        [df, valor_atipico, frequencia_recente, dispositivo_raro, ciclo_diario],
        axis=1,
    )
    logger.info("Features Pix adicionadas: %s", df_com_features.shape)
    return df_com_features


# ─── Execução direta (smoke test) ───────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.data_loader import carregar_dados

    _, _, df_merged = carregar_dados(nrows=50_000)
    df_features = criar_features_pix(df_merged)

    colunas_novas = [
        "valor_atipico_proxy",
        "frequencia_recente_proxy",
        "dispositivo_raro_proxy",
        "posicao_ciclo_diario_relativa",
        "posicao_ciclo_diario_sen",
        "posicao_ciclo_diario_cos",
    ]
    print("\nShape final:", df_features.shape)
    print("\nPreview das features novas:")
    print(df_features[colunas_novas].describe())
