"""
Busca de hiperparâmetros do XGBoost com Optuna.

O XGBoost é o modelo principal da proposta aprovada, e o baseline de junho
(regressão logística) é o piso que ele precisa superar: AUC-PR de 0,3930 na
validação, com precisão de 0,1350 e recall de 0,6772.

Quatro decisões fixadas aqui, todas com motivo:

1. **A métrica otimizada é a AUC-PR na validação**, a mesma que orienta o
   trabalho desde o EDA. Otimizar AUC-ROC premiaria ordenar bem os 96,5% de
   casos fáceis; otimizar acurácia premiaria não detectar nada.
2. **`scale_pos_weight` não entra na busca.** Ele é fixado na razão real entre
   classes do treino — é a forma do XGBoost expressar a ponderação de classe
   adotada em junho. Deixá-lo variar transformaria a estratégia de
   desbalanceamento em mais um hiperparâmetro, contradizendo uma decisão já
   registrada e comparada.
3. **O pré-processamento é ajustado uma vez, fora da busca.** Ele não depende
   dos hiperparâmetros, então refazê-lo a cada tentativa repetiria trabalho
   idêntico dezenas de vezes. Não há vazamento: o ajuste usa somente o treino,
   e validação e teste apenas recebem a transformação.
4. **Parada antecipada pela validação.** Cada tentativa treina até 600 árvores
   mas para quando a AUC-PR de validação deixa de melhorar. Isso usa o conjunto
   de validação para decidir onde parar — que é o papel dele — e mantém o
   conjunto de teste intocado.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
from xgboost import XGBClassifier

from src.models.evaluator import avaliar_probabilidades

logger = logging.getLogger(__name__)

N_ARVORES_MAXIMO = 600
RODADAS_SEM_MELHORA = 50
SEMENTE = 42


def espaco_busca(trial: optuna.Trial) -> dict:
    """
    Faixas de busca dos hiperparâmetros.

    Cobrem os três eixos que governam um modelo de árvores impulsionadas:
    capacidade (`max_depth`, `min_child_weight`), velocidade de aprendizado
    (`learning_rate`) e regularização (`subsample`, `colsample_bytree`,
    `gamma`, `reg_alpha`, `reg_lambda`).
    """
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def criar_modelo(parametros: dict, razao_classes: float, n_estimators: int | None = None) -> XGBClassifier:
    """Monta o classificador com os parâmetros fixos do projeto mais os variáveis."""
    return XGBClassifier(
        **parametros,
        n_estimators=n_estimators or N_ARVORES_MAXIMO,
        tree_method="hist",
        scale_pos_weight=razao_classes,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=SEMENTE,
    )


def razao_entre_classes(y) -> float:
    """Razão legítimas:fraudes, usada como `scale_pos_weight`."""
    y = np.asarray(y)
    positivos = int((y == 1).sum())
    if positivos == 0:
        raise ValueError("não há exemplos positivos para calcular a razão entre classes")
    return float((y == 0).sum() / positivos)


def buscar(
    M_treino,
    y_treino,
    M_validacao,
    y_validacao,
    n_tentativas: int = 50,
    tempo_limite_s: int | None = 3600,
) -> optuna.Study:
    """
    Executa a busca e devolve o estudo do Optuna.

    `tempo_limite_s` é uma trava de segurança: a busca para no que vier
    primeiro, número de tentativas ou tempo. Sem ela, uma combinação lenta
    pode estender a execução muito além do previsto em uma máquina com pouca
    memória.
    """
    razao = razao_entre_classes(y_treino)
    logger.info("scale_pos_weight fixado em %.2f (razão legítimas:fraudes no treino)", razao)

    def objetivo(trial: optuna.Trial) -> float:
        modelo = criar_modelo(espaco_busca(trial), razao)
        modelo.set_params(early_stopping_rounds=RODADAS_SEM_MELHORA)
        modelo.fit(M_treino, y_treino, eval_set=[(M_validacao, y_validacao)], verbose=False)

        metricas = avaliar_probabilidades(
            y_validacao, modelo.predict_proba(M_validacao)[:, 1]
        )
        # Guardados para a análise posterior: quantas árvores a parada
        # antecipada manteve, e as métricas de decisão no limiar padrão.
        trial.set_user_attr("n_arvores", int(modelo.best_iteration + 1))
        trial.set_user_attr("recall", metricas["recall"])
        trial.set_user_attr("precisao", metricas["precisao"])
        trial.set_user_attr("auc_roc", metricas["auc_roc"])
        return metricas["auc_pr"]

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMENTE),
        study_name="xgboost_aucpr_validacao",
    )
    estudo.optimize(objetivo, n_trials=n_tentativas, timeout=tempo_limite_s,
                    show_progress_bar=False)
    return estudo


def resumo(estudo: optuna.Study) -> dict:
    """Extrai do estudo o que vai para o relatório e para as anotações."""
    melhor = estudo.best_trial
    concluidas = [t for t in estudo.trials if t.state == optuna.trial.TrialState.COMPLETE]
    valores = [t.value for t in concluidas]
    return {
        "n_tentativas": len(estudo.trials),
        "n_concluidas": len(concluidas),
        "melhor_auc_pr": estudo.best_value,
        "melhores_parametros": estudo.best_params,
        "n_arvores_da_melhor": melhor.user_attrs.get("n_arvores"),
        "recall_da_melhor": melhor.user_attrs.get("recall"),
        "precisao_da_melhor": melhor.user_attrs.get("precisao"),
        "auc_roc_da_melhor": melhor.user_attrs.get("auc_roc"),
        "auc_pr_pior": min(valores) if valores else None,
        "auc_pr_mediana": float(np.median(valores)) if valores else None,
    }
