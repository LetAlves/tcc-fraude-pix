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

def _mediana_expandida_causal(serie_ordenada: pd.Series, chave_grupo: pd.Series) -> pd.Series:
    """
    Mediana expandida por grupo, usando só observações estritamente anteriores.

    Pré-requisito: `serie_ordenada` e `chave_grupo` já devem estar ordenadas
    por tempo (ex.: TransactionDT) antes da chamada.
    """
    mediana = serie_ordenada.groupby(chave_grupo).expanding().median()
    mediana = mediana.reset_index(level=0, drop=True)
    return mediana.groupby(chave_grupo).shift(1)


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
    ordenado = df.sort_values(coluna_tempo)
    mediana_anterior = _mediana_expandida_causal(ordenado[coluna_valor], ordenado[coluna_cartao])
    desvio_abs = (ordenado[coluna_valor] - mediana_anterior).abs()
    mad_anterior = _mediana_expandida_causal(desvio_abs, ordenado[coluna_cartao])

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
    Nulos: 0 quando não há transação anterior na janela (ausência de evento,
        não ausência de dado).
    Risco de vazamento: nenhum — janela fechada à esquerda (`closed="left"`)
        exclui a própria linha e qualquer evento futuro.
    """
    # Ordena por (cartao, tempo): dentro de cada grupo fica cronológico, e essa
    # é a mesma ordem "em blocos" que groupby().rolling() usa internamente —
    # necessário para reatribuir o índice original por posição depois.
    # Só as duas colunas-fonte entram na ordenação. Ordenar o DataFrame inteiro
    # por (cartão, tempo) é uma ordem genuinamente nova, então o pandas
    # materializa as 434 colunas — 1,76 GiB só no bloco float64 do dataset
    # completo, o que estourava a memória antes de qualquer modelo rodar.
    ordenado = df[[coluna_cartao, coluna_tempo]].sort_values([coluna_cartao, coluna_tempo])
    ordenado["_dt_sintetico"] = pd.to_datetime(ordenado[coluna_tempo], unit="s")

    contagem = (
        ordenado.groupby(coluna_cartao)
        .rolling(f"{janela_segundos}s", on="_dt_sintetico", closed="left")[coluna_tempo]
        .count()
    )
    # Com on=, o 2º nível do MultiIndex vira a data sintética (que se repete
    # entre cartões) em vez do índice original — por isso reatribuímos por
    # posição, e não por reindex baseado em rótulo.
    contagem = contagem.reset_index(drop=True)
    contagem.index = ordenado.index
    return contagem.reindex(df.index).rename("frequencia_recente_proxy")


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
    Nulos: NaN quando DeviceInfo está ausente — tratamento de imputação fica
        a cargo de preprocessor.py (tarefa seguinte), não deste módulo.
    Risco de vazamento: nenhum — cumcount só conta ocorrências anteriores
        na ordem cronológica.
    """
    ordenado = df.sort_values(coluna_tempo)
    tem_dispositivo = ordenado[coluna_dispositivo].notna()
    chave_par = ordenado[coluna_cartao].astype(str) + "|" + ordenado[coluna_dispositivo].astype(str)

    contagem_anterior = chave_par.groupby(chave_par).cumcount()
    raridade = 1.0 / (contagem_anterior + 1)
    raridade = raridade.where(tem_dispositivo)
    return raridade.reindex(df.index).rename("dispositivo_raro_proxy")


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
