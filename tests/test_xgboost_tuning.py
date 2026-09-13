import unittest

import numpy as np
import optuna
import pandas as pd

from src.models.xgboost_tuning import (
    CallbackPodaOptuna,
    N_ARVORES_MAXIMO,
    PROFUNDIDADE_MAXIMA,
    _resumir_tentativa,
    espaco_busca,
    selecionar_cauda_treino,
)


class TrialPodaFalso:
    def __init__(self, deve_podar: bool) -> None:
        self.deve_podar = deve_podar
        self.relatorios = []

    def report(self, valor: float, step: int) -> None:
        self.relatorios.append((valor, step))

    def should_prune(self) -> bool:
        return self.deve_podar


class XGBoostTuningTest(unittest.TestCase):
    def test_espaco_reduzido_respeita_limites_aprovados(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "learning_rate": 0.1,
                "max_depth": PROFUNDIDADE_MAXIMA,
                "min_child_weight": 2.0,
                "subsample": 0.8,
                "colsample_bytree": 0.7,
                "gamma": 1.0,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }
        )

        parametros = espaco_busca(trial)

        self.assertEqual(parametros["max_depth"], 8)
        self.assertEqual(PROFUNDIDADE_MAXIMA, 8)
        self.assertEqual(N_ARVORES_MAXIMO, 300)

    def test_seleciona_as_linhas_mais_recentes_do_treino(self) -> None:
        matriz = np.arange(30).reshape(10, 3)
        alvo = pd.Series(range(10))

        matriz_cauda, alvo_cauda = selecionar_cauda_treino(matriz, alvo, 4)

        np.testing.assert_array_equal(matriz_cauda, matriz[-4:])
        self.assertEqual(alvo_cauda.tolist(), [6, 7, 8, 9])

    def test_rejeita_amostra_invalida(self) -> None:
        with self.assertRaisesRegex(ValueError, "maior que zero"):
            selecionar_cauda_treino(np.zeros((3, 2)), np.zeros(3), 0)

        with self.assertRaisesRegex(ValueError, "mesmo número"):
            selecionar_cauda_treino(np.zeros((3, 2)), np.zeros(2), 2)

    def test_callback_reporta_metrica_sem_interromper(self) -> None:
        trial = TrialPodaFalso(deve_podar=False)
        callback = CallbackPodaOptuna(trial)

        interromper = callback.after_iteration(
            None, 7, {"validation_0": {"aucpr": [0.51, 0.53]}}
        )

        self.assertFalse(interromper)
        self.assertEqual(trial.relatorios, [(0.53, 7)])

    def test_callback_poda_tentativa_ruim(self) -> None:
        trial = TrialPodaFalso(deve_podar=True)
        callback = CallbackPodaOptuna(trial)

        with self.assertRaises(optuna.TrialPruned):
            callback.after_iteration(
                None, 50, {"validation_0": {"aucpr": [0.42]}}
            )

    def test_resumo_nao_confunde_sinal_de_poda_com_auc_pr_final(self) -> None:
        completa = optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=0.57,
        )
        podada = optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            value=0.41,
        )

        resumo_completa = _resumir_tentativa(completa)
        resumo_podada = _resumir_tentativa(podada)

        self.assertEqual(resumo_completa["auc_pr"], 0.57)
        self.assertIsNone(resumo_completa["sinal_poda_aucpr_xgboost"])
        self.assertIsNone(resumo_podada["auc_pr"])
        self.assertEqual(resumo_podada["sinal_poda_aucpr_xgboost"], 0.41)


if __name__ == "__main__":
    unittest.main()
