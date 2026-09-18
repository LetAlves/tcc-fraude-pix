"""
Busca de hiperparâmetros do XGBoost com Optuna.

O XGBoost é o modelo principal da proposta aprovada, e o baseline de junho
(regressão logística) é o piso que ele precisa superar: AUC-PR de 0,3930 na
validação, com precisão de 0,1350 e recall de 0,6772.

Seis decisões fixadas aqui, todas com motivo:

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
4. **Parada antecipada pela validação.** Cada tentativa treina até 300 árvores
   mas para quando a AUC-PR de validação deixa de melhorar. Isso usa o conjunto
   de validação para decidir onde parar — que é o papel dele — e mantém o
   conjunto de teste intocado.
5. **Busca na cauda do treino.** O ajuste usa as 150 mil transações mais
   recentes do treino, por serem temporalmente mais próximas da validação. Os
   parâmetros escolhidos são reaplicados ao treino completo; a métrica desse
   modelo refeito é o resultado final, e não a métrica da amostra.
6. **Poda de tentativas ruins.** Depois de cinco tentativas completas e 50
   árvores de aquecimento, o Optuna pode interromper configurações abaixo da
   mediana. A poda usa a `aucpr` interna do XGBoost como sinal intermediário;
   a ordenação final das tentativas completas usa a AUC-PR produzida pelo
   `evaluator.py`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import optuna
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from src.models.evaluator import avaliar_probabilidades

logger = logging.getLogger(__name__)

N_ARVORES_MAXIMO = 300
PROFUNDIDADE_MAXIMA = 8
RODADAS_SEM_MELHORA = 50
SEMENTE = 42
TENTATIVAS_INICIAIS_SEM_PODA = 5
RODADAS_INICIAIS_SEM_PODA = 50
INTERVALO_PODA = 10


class CallbackPodaOptuna(TrainingCallback):
    """Reporta a AUC-PR interna do XGBoost e interrompe tentativas ruins."""

    def __init__(self, trial: optuna.Trial) -> None:
        self.trial = trial

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        try:
            auc_pr = float(evals_log["validation_0"]["aucpr"][-1])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "o pruning exige eval_set com a métrica 'aucpr'"
            ) from exc

        self.trial.report(auc_pr, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"tentativa podada na árvore {epoch + 1}; aucpr={auc_pr:.6f}"
            )
        return False


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
        "max_depth": trial.suggest_int("max_depth", 3, PROFUNDIDADE_MAXIMA),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def criar_modelo(
    parametros: dict,
    razao_classes: float,
    n_estimators: int | None = None,
    callbacks: list[TrainingCallback] | None = None,
) -> XGBClassifier:
    """Monta o classificador com os parâmetros fixos do projeto mais os variáveis."""
    return XGBClassifier(
        **parametros,
        n_estimators=n_estimators or N_ARVORES_MAXIMO,
        tree_method="hist",
        scale_pos_weight=razao_classes,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=SEMENTE,
        callbacks=callbacks,
    )


def razao_entre_classes(y) -> float:
    """Razão legítimas:fraudes, usada como `scale_pos_weight`."""
    y = np.asarray(y)
    positivos = int((y == 1).sum())
    if positivos == 0:
        raise ValueError("não há exemplos positivos para calcular a razão entre classes")
    return float((y == 0).sum() / positivos)


def selecionar_cauda_treino(M_treino, y_treino, n_linhas: int = 150_000):
    """Seleciona as linhas mais recentes do treino sem alterar sua ordem."""
    if n_linhas <= 0:
        raise ValueError("n_linhas deve ser maior que zero")
    if len(M_treino) != len(y_treino):
        raise ValueError("matriz e alvo precisam ter o mesmo número de linhas")

    inicio = max(0, len(y_treino) - n_linhas)
    y_cauda = (
        y_treino.iloc[inicio:]
        if hasattr(y_treino, "iloc")
        else np.asarray(y_treino)[inicio:]
    )
    return M_treino[inicio:], y_cauda


def buscar(
    M_treino,
    y_treino,
    M_validacao,
    y_validacao,
    n_tentativas: int = 50,
    tempo_limite_s: int | None = None,
    razao_classes: float | None = None,
    caminho_storage: str | Path | None = None,
) -> optuna.Study:
    """
    Executa a busca e devolve o estudo do Optuna.

    Quando `caminho_storage` é informado, o estudo é persistido em SQLite e
    retomado até alcançar `n_tentativas` no total. `tempo_limite_s` limita
    apenas a execução atual; `None` permite concluir a quantidade solicitada.
    """
    razao = (
        razao_entre_classes(y_treino)
        if razao_classes is None
        else float(razao_classes)
    )
    if not np.isfinite(razao) or razao <= 0:
        raise ValueError("razao_classes deve ser um número positivo e finito")
    logger.info("scale_pos_weight fixado em %.2f (razão legítimas:fraudes no treino)", razao)

    def objetivo(trial: optuna.Trial) -> float:
        modelo = criar_modelo(
            espaco_busca(trial),
            razao,
            callbacks=[CallbackPodaOptuna(trial)],
        )
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
        trial.set_user_attr("scale_pos_weight", razao)
        return metricas["auc_pr"]

    storage = None
    if caminho_storage is not None:
        caminho = Path(caminho_storage).resolve()
        caminho.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{caminho.as_posix()}"

    estudo = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEMENTE),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=TENTATIVAS_INICIAIS_SEM_PODA,
            n_warmup_steps=RODADAS_INICIAIS_SEM_PODA,
            interval_steps=INTERVALO_PODA,
        ),
        study_name="xgboost_aucpr_validacao_julho_v2",
        storage=storage,
        load_if_exists=storage is not None,
    )
    restantes = max(0, n_tentativas - len(estudo.trials))
    logger.info(
        "Estudo com %d tentativa(s) registrada(s); faltam %d para o alvo de %d",
        len(estudo.trials), restantes, n_tentativas,
    )
    if restantes:
        estudo.optimize(
            objetivo,
            n_trials=restantes,
            timeout=tempo_limite_s,
            show_progress_bar=False,
            gc_after_trial=True,
        )
    return estudo


def resumo(estudo: optuna.Study) -> dict:
    """Extrai do estudo o que vai para o relatório e para as anotações."""
    melhor = estudo.best_trial
    concluidas = [t for t in estudo.trials if t.state == optuna.trial.TrialState.COMPLETE]
    podadas = [t for t in estudo.trials if t.state == optuna.trial.TrialState.PRUNED]
    falhas = [t for t in estudo.trials if t.state == optuna.trial.TrialState.FAIL]
    valores = [t.value for t in concluidas]
    return {
        "n_tentativas": len(estudo.trials),
        "n_concluidas": len(concluidas),
        "n_podadas": len(podadas),
        "n_falhas": len(falhas),
        "melhor_auc_pr": estudo.best_value,
        "melhores_parametros": estudo.best_params,
        "n_arvores_da_melhor": melhor.user_attrs.get("n_arvores"),
        "recall_da_melhor": melhor.user_attrs.get("recall"),
        "precisao_da_melhor": melhor.user_attrs.get("precisao"),
        "auc_roc_da_melhor": melhor.user_attrs.get("auc_roc"),
        "auc_pr_pior": min(valores) if valores else None,
        "auc_pr_mediana": float(np.median(valores)) if valores else None,
        "tentativas": [_resumir_tentativa(trial) for trial in estudo.trials],
    }


def _resumir_tentativa(trial: optuna.trial.FrozenTrial) -> dict:
    """Distingue a métrica final do evaluator do sinal parcial de poda."""
    completa = trial.state == optuna.trial.TrialState.COMPLETE
    podada = trial.state == optuna.trial.TrialState.PRUNED
    return {
        "numero": trial.number,
        "estado": trial.state.name,
        "auc_pr": trial.value if completa else None,
        "sinal_poda_aucpr_xgboost": trial.value if podada else None,
        "parametros": trial.params,
        "n_arvores": trial.user_attrs.get("n_arvores"),
        "duracao_s": (
            trial.duration.total_seconds()
            if trial.duration is not None
            else None
        ),
    }
