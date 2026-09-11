"""
Avaliação de modelos de classificação de fraude.

Reúne em um só lugar as métricas usadas no trabalho, a matriz de confusão e
a curva de precisão-recall, para que baseline (junho) e modelos de julho
sejam medidos exatamente pelo mesmo código — comparar números produzidos por
implementações diferentes não é comparação.

Três decisões de projeto, todas vindas de problemas concretos:

1. **A função central recebe probabilidades, não um modelo.** Assim ela serve
   a qualquer estimador (regressão logística, XGBoost, Random Forest) e não
   força uma segunda passagem de inferência sobre o conjunto inteiro.
2. **O limiar é sempre explícito.** O padrão 0,5 é uma escolha, não uma
   constante da natureza; deixá-lo implícito no `predict` escondeu essa
   decisão durante todo o baseline de junho.
3. **Toda métrica vem acompanhada da taxa base.** Em um problema com 3,5% de
   fraude, a AUC-PR de um classificador aleatório é igual à proporção de
   positivos: sem esse número ao lado, a AUC-PR não tem régua de leitura.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)

LIMIAR_PADRAO = 0.5

# Ordem fixa das métricas nas tabelas comparativas. AUC-PR vem primeiro por ser
# a métrica principal do trabalho, dada a proporção de fraude.
ORDEM_METRICAS = ["auc_pr", "auc_pr_relativo", "auc_roc", "recall", "precisao", "f1"]


# ─── Métricas ────────────────────────────────────────────────────────────────

def avaliar_probabilidades(
    y_verdadeiro,
    probabilidades,
    limiar: float = LIMIAR_PADRAO,
) -> dict:
    """
    Calcula todas as métricas do trabalho a partir das probabilidades previstas.

    `probabilidades` é a probabilidade da classe positiva (fraude), tipicamente
    `modelo.predict_proba(X)[:, 1]`.

    Retorna, além das métricas: `n`, `n_fraudes` e `taxa_base` (contexto sem o
    qual a AUC-PR não se interpreta) e `auc_pr_relativo`, quantas vezes a
    AUC-PR supera a de um classificador aleatório.

    As métricas de ordenação (`auc_roc`, `auc_pr`) não dependem do limiar; as de
    decisão (`recall`, `precisao`, `f1`) dependem, e por isso o limiar usado é
    devolvido junto.
    """
    y_verdadeiro = np.asarray(y_verdadeiro)
    probabilidades = np.asarray(probabilidades)
    previsoes = probabilidades >= limiar

    taxa_base = float(y_verdadeiro.mean())
    metricas = {
        "auc_roc": float(roc_auc_score(y_verdadeiro, probabilidades)),
        "auc_pr": float(average_precision_score(y_verdadeiro, probabilidades)),
        "recall": float(recall_score(y_verdadeiro, previsoes, zero_division=0)),
        "precisao": float(precision_score(y_verdadeiro, previsoes, zero_division=0)),
        "f1": float(f1_score(y_verdadeiro, previsoes, zero_division=0)),
        "limiar": float(limiar),
        "n": int(len(y_verdadeiro)),
        "n_fraudes": int(y_verdadeiro.sum()),
        "taxa_base": taxa_base,
    }
    metricas["auc_pr_relativo"] = metricas["auc_pr"] / taxa_base if taxa_base > 0 else float("nan")
    return metricas


def avaliar(modelo, X, y, limiar: float = LIMIAR_PADRAO) -> dict:
    """Conveniência: extrai as probabilidades do modelo e avalia."""
    return avaliar_probabilidades(y, modelo.predict_proba(X)[:, 1], limiar=limiar)


# ─── Matriz de confusão ──────────────────────────────────────────────────────

def matriz_confusao(
    y_verdadeiro,
    probabilidades,
    limiar: float = LIMIAR_PADRAO,
) -> pd.DataFrame:
    """
    Matriz de confusão rotulada, em contagens absolutas.

    As contagens absolutas importam tanto quanto as taxas: "recall de 0,70" e
    "18.000 transações marcadas para revisão manual" descrevem o mesmo modelo,
    mas só o segundo número diz se o sistema é operável.
    """
    previsoes = np.asarray(probabilidades) >= limiar
    vn, fp, fn, vp = confusion_matrix(y_verdadeiro, previsoes, labels=[0, 1]).ravel()
    return pd.DataFrame(
        [[vn, fp], [fn, vp]],
        index=pd.Index(["real: legítima", "real: fraude"], name="observado"),
        columns=pd.Index(["prevista: legítima", "prevista: fraude"], name="previsto"),
    )


def resumo_operacional(
    y_verdadeiro,
    probabilidades,
    limiar: float = LIMIAR_PADRAO,
) -> dict:
    """
    Traduz a matriz de confusão para a linguagem de quem opera o sistema.

    Responde: quantas fraudes foram recuperadas, quantas transações seriam
    enviadas para revisão, e que fração do total isso representa.
    """
    previsoes = np.asarray(probabilidades) >= limiar
    vn, fp, fn, vp = confusion_matrix(y_verdadeiro, previsoes, labels=[0, 1]).ravel()
    marcadas = int(vp + fp)
    total = int(vn + fp + fn + vp)
    return {
        "fraudes_recuperadas": int(vp),
        "fraudes_perdidas": int(fn),
        "transacoes_marcadas": marcadas,
        "alarmes_falsos": int(fp),
        "fracao_marcada": marcadas / total if total else float("nan"),
    }


# ─── Curva de precisão-recall ────────────────────────────────────────────────

def curva_precisao_recall(y_verdadeiro, probabilidades):
    """
    Pontos da curva de precisão-recall e a linha de referência do acaso.

    Retorna `(precisao, recall, limiares, taxa_base)`. A `taxa_base` é a
    precisão de um classificador aleatório e deve ser desenhada como linha
    horizontal: sem ela, uma curva ruim em dado desbalanceado parece razoável.
    """
    precisao, recall, limiares = precision_recall_curve(y_verdadeiro, probabilidades)
    return precisao, recall, limiares, float(np.asarray(y_verdadeiro).mean())


def plotar_curva_precisao_recall(y_verdadeiro, probabilidades, rotulo=None, eixo=None):
    """
    Desenha a curva de precisão-recall com a linha do acaso.

    `matplotlib` é importado aqui dentro, e não no topo do módulo, para que as
    métricas possam ser calculadas em ambiente sem interface gráfica.
    """
    import matplotlib.pyplot as plt

    precisao, recall, _, taxa_base = curva_precisao_recall(y_verdadeiro, probabilidades)
    auc_pr = average_precision_score(y_verdadeiro, probabilidades)

    if eixo is None:
        _, eixo = plt.subplots(figsize=(6, 4.5))

    nome = f"{rotulo} " if rotulo else ""
    eixo.plot(recall, precisao, label=f"{nome}(AUC-PR = {auc_pr:.4f})")
    eixo.axhline(
        taxa_base, linestyle="--", linewidth=1, color="gray",
        label=f"acaso ({taxa_base:.4f})",
    )
    eixo.set_xlabel("Recall")
    eixo.set_ylabel("Precisão")
    eixo.set_xlim(0, 1)
    eixo.set_ylim(0, 1)
    eixo.legend(loc="upper right", fontsize=9)
    return eixo


# ─── Apresentação ────────────────────────────────────────────────────────────

def imprimir(metricas: dict, nome: str) -> None:
    """Imprime as métricas em bloco legível, com o contexto antes dos números."""
    print(f"--- {nome} ---")
    print(
        f"  {metricas['n']:,} transações | {metricas['n_fraudes']:,} fraudes"
        f" | taxa base {100 * metricas['taxa_base']:.3f}%"
        f" | limiar {metricas['limiar']:.2f}".replace(",", ".")
    )
    for chave in ("auc_roc", "auc_pr", "recall", "precisao", "f1"):
        print(f"  {chave}: {metricas[chave]:.4f}")
    print(f"  auc_pr_relativo: {metricas['auc_pr_relativo']:.1f}x o acaso")


def comparar(resultados: dict[str, dict]) -> pd.DataFrame:
    """
    Tabela comparativa a partir de `{nome_do_modelo: métricas}`.

    Mantém apenas as métricas de desempenho, na ordem de leitura do trabalho;
    contagens e taxa base ficam de fora por serem idênticas entre modelos
    avaliados no mesmo conjunto.
    """
    tabela = pd.DataFrame(resultados).T
    colunas = [c for c in ORDEM_METRICAS if c in tabela.columns]
    return tabela[colunas]
