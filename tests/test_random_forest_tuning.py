"""Testes da busca de hiperparâmetros do Random Forest, com dados sintéticos."""

import unittest

import numpy as np
import optuna

from src.models.random_forest_tuning import (
    buscar,
    criar_modelo,
    espaco_busca,
    parametros_do_estudo,
    resumo,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _dados(semente=0, n=400):
    gerador = np.random.default_rng(semente)
    X = gerador.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + gerador.normal(scale=0.5, size=n) > 1.2).astype(int)
    return X, y


class EspacoBuscaTest(unittest.TestCase):
    def test_parametros_ficam_nas_faixas(self):
        estudo = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
        for _ in range(30):
            trial = estudo.ask()
            parametros = espaco_busca(trial)
            self.assertTrue(parametros["max_depth"] is None or 8 <= parametros["max_depth"] <= 64)
            self.assertTrue(1 <= parametros["min_samples_leaf"] <= 50)
            self.assertTrue(0.02 <= parametros["max_features"] <= 0.5)
            self.assertTrue(0.3 <= parametros["max_samples"] <= 1.0)
            estudo.tell(trial, 0.0)

    def test_conversao_dos_parametros_gravados(self):
        self.assertEqual(
            parametros_do_estudo({"sem_limite_profundidade": True, "min_samples_leaf": 3}),
            {"max_depth": None, "min_samples_leaf": 3},
        )
        self.assertEqual(
            parametros_do_estudo({"sem_limite_profundidade": False, "max_depth": 20})["max_depth"],
            20,
        )


class CriarModeloTest(unittest.TestCase):
    def test_ponderacao_de_classe_e_semente_fixas(self):
        modelo = criar_modelo({"min_samples_leaf": 5})
        self.assertEqual(modelo.class_weight, "balanced")
        self.assertEqual(modelo.random_state, 42)
        self.assertEqual(modelo.n_estimators, 100)


class BuscaTest(unittest.TestCase):
    def test_busca_curta_otimiza_aucpr_na_validacao(self):
        X_treino, y_treino = _dados(0)
        X_validacao, y_validacao = _dados(1)
        estudo = buscar(X_treino, y_treino, X_validacao, y_validacao, n_tentativas=3, n_estimators=10)
        relatorio = resumo(estudo)

        self.assertEqual(relatorio["n_concluidas"], 3)
        self.assertGreater(relatorio["melhor_auc_pr_busca"], y_validacao.mean())
        self.assertNotIn("sem_limite_profundidade", relatorio["melhores_parametros"])


if __name__ == "__main__":
    unittest.main()
