"""
Busca de hiperparâmetros do Random Forest com Optuna (experimento complementar).

O Random Forest foi treinado sem busca em julho, como modelo comparativo. O
XGBoost, por outro lado, passou por 50 tentativas de Optuna. A comparação final
ficou assimétrica: com hiperparâmetros padrão os dois empataram (0,5293 contra
0,5298 de AUC-PR na validação), e só o XGBoost recebeu ajuste. Este módulo
aplica ao Random Forest o mesmo protocolo, para que a diferença restante possa
ser atribuída ao algoritmo, e não ao esforço de ajuste.

As decisões seguem as de ``xgboost_tuning.py``:

1. **A métrica otimizada é a AUC-PR na validação completa.** O teste não é
   usado: o modelo entregue continua sendo o XGBoost, cujo teste já foi aberto.
2. **``class_weight='balanced'`` não entra na busca.** É a ponderação de classe
   adotada em junho, a mesma expressa no XGBoost por ``scale_pos_weight``.
3. **O pré-processamento é ajustado uma vez, somente no treino.**
4. **Busca na cauda do treino.** As tentativas usam as 150 mil transações mais
   recentes do treino; a melhor configuração é retreinada no treino completo.
5. **Número de árvores fixo em 100**, o mesmo do comparativo de julho. A busca
   varia a forma de cada árvore (profundidade, tamanho das folhas, atributos e
   linhas sorteados), não o tamanho da floresta.

Diferente do XGBoost, não há poda: o Random Forest não produz um sinal
intermediário por árvore que permita interromper tentativas ruins.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluator import avaliar_probabilidades
from src.models.random_forest import N_ARVORES, SEMENTE

logger = logging.getLogger(__name__)

PROFUNDIDADE_MINIMA = 8
PROFUNDIDADE_MAXIMA = 64


def espaco_busca(trial: optuna.Trial) -> dict:
    """
    Faixas de busca dos hiperparâmetros.

    ``max_depth`` pode ficar sem limite, como no comparativo de julho, ou entre
    8 e 64 níveis (64 foi a profundidade média observada com árvores completas).
    ``min_samples_leaf`` controla a memorização, ``max_features`` o sorteio de
    atributos por divisão e ``max_samples`` a fração de linhas de cada árvore.
    """
    sem_limite = trial.suggest_categorical("sem_limite_profundidade", [True, False])
    max_depth = (
        None
        if sem_limite
        else trial.suggest_int("max_depth", PROFUNDIDADE_MINIMA, PROFUNDIDADE_MAXIMA, log=True)
    )
    return {
        "max_depth": max_depth,
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50, log=True),
        "max_features": trial.suggest_float("max_features", 0.02, 0.5, log=True),
        "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
    }


def parametros_do_estudo(parametros_optuna: dict) -> dict:
    """Converte os parâmetros gravados pelo Optuna nos do scikit-learn."""
    parametros = dict(parametros_optuna)
    sem_limite = parametros.pop("sem_limite_profundidade")
    parametros["max_depth"] = None if sem_limite else parametros["max_depth"]
    return parametros


def criar_modelo(parametros: dict | None = None, n_estimators: int = N_ARVORES) -> RandomForestClassifier:
    """Random Forest com os parâmetros fixos do projeto mais os variáveis da busca."""
    return RandomForestClassifier(
        **(parametros or {}),
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEMENTE,
    )


def buscar(
    M_treino,
    y_treino,
    M_validacao,
    y_validacao,
    n_tentativas: int = 30,
    caminho_storage: str | Path | None = None,
    n_estimators: int = N_ARVORES,
) -> optuna.Study:
    """
    Executa a busca e devolve o estudo do Optuna.

    Com ``caminho_storage``, o estudo é persistido em SQLite e retomado até
    alcançar ``n_tentativas`` no total, o que permite interromper a execução.
    """

    def objetivo(trial: optuna.Trial) -> float:
        modelo = criar_modelo(espaco_busca(trial), n_estimators=n_estimators)
        modelo.fit(M_treino, y_treino)
        metricas = avaliar_probabilidades(y_validacao, modelo.predict_proba(M_validacao)[:, 1])
        profundidades = [arvore.get_depth() for arvore in modelo.estimators_]
        trial.set_user_attr("auc_roc", metricas["auc_roc"])
        trial.set_user_attr("profundidade_media", float(np.mean(profundidades)))
        return metricas["auc_pr"]

    storage = None
    if caminho_storage is not None:
        caminho = Path(caminho_storage).resolve()
        caminho.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{caminho.as_posix()}"

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMENTE),
        study_name="random_forest_aucpr_validacao_outubro",
        storage=storage,
        load_if_exists=storage is not None,
    )
    restantes = max(0, n_tentativas - len(estudo.trials))
    logger.info(
        "Estudo com %d tentativa(s) registrada(s); faltam %d para o alvo de %d",
        len(estudo.trials), restantes, n_tentativas,
    )
    if restantes:
        estudo.optimize(objetivo, n_trials=restantes, show_progress_bar=False, gc_after_trial=True)
    return estudo


def resumo(estudo: optuna.Study) -> dict:
    """Extrai do estudo o que vai para o relatório."""
    concluidas = [t for t in estudo.trials if t.state == optuna.trial.TrialState.COMPLETE]
    valores = [t.value for t in concluidas]
    melhor = estudo.best_trial
    return {
        "n_tentativas": len(estudo.trials),
        "n_concluidas": len(concluidas),
        "melhor_auc_pr_busca": estudo.best_value,
        "melhores_parametros": parametros_do_estudo(estudo.best_params),
        "auc_roc_da_melhor_busca": melhor.user_attrs.get("auc_roc"),
        "auc_pr_mediana": float(np.median(valores)) if valores else None,
        "auc_pr_pior": min(valores) if valores else None,
        "tentativas": [
            {
                "numero": t.number,
                "estado": t.state.name,
                "auc_pr": t.value,
                "parametros": parametros_do_estudo(t.params) if t.params else None,
                "profundidade_media": t.user_attrs.get("profundidade_media"),
                "segundos": t.duration.total_seconds() if t.duration else None,
            }
            for t in estudo.trials
        ],
    }
