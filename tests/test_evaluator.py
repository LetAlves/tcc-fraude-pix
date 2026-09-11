"""Testes do módulo de avaliação.

Os valores esperados são calculados à mão em cenários pequenos, não copiados
de uma execução anterior: um teste que compara o código com ele mesmo não
detecta erro nenhum.
"""

import unittest

import numpy as np

from src.models.evaluator import (avaliar_probabilidades, comparar,
                                  curva_precisao_recall, matriz_confusao,
                                  resumo_operacional)


class AvaliarProbabilidadesTest(unittest.TestCase):

    def test_separacao_perfeita(self) -> None:
        y = [0, 0, 1, 1]
        probas = [0.1, 0.2, 0.8, 0.9]

        m = avaliar_probabilidades(y, probas)

        self.assertAlmostEqual(m["auc_roc"], 1.0)
        self.assertAlmostEqual(m["auc_pr"], 1.0)
        self.assertAlmostEqual(m["recall"], 1.0)
        self.assertAlmostEqual(m["precisao"], 1.0)
        self.assertAlmostEqual(m["f1"], 1.0)

    def test_contagens_e_taxa_base(self) -> None:
        y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # 1 fraude em 10

        m = avaliar_probabilidades(y, [0.0] * 9 + [1.0])

        self.assertEqual(m["n"], 10)
        self.assertEqual(m["n_fraudes"], 1)
        self.assertAlmostEqual(m["taxa_base"], 0.1)

    def test_auc_pr_relativo_e_a_razao_sobre_a_taxa_base(self) -> None:
        y = [0, 0, 0, 1]  # taxa base 0,25
        probas = [0.1, 0.2, 0.3, 0.9]

        m = avaliar_probabilidades(y, probas)

        self.assertAlmostEqual(m["auc_pr_relativo"], m["auc_pr"] / 0.25)

    def test_limiar_muda_as_metricas_de_decisao_e_nao_as_de_ordenacao(self) -> None:
        y = [0, 0, 1, 1]
        probas = [0.10, 0.45, 0.55, 0.90]

        frouxo = avaliar_probabilidades(y, probas, limiar=0.4)
        rigido = avaliar_probabilidades(y, probas, limiar=0.6)

        # ordenação não depende do corte
        self.assertAlmostEqual(frouxo["auc_roc"], rigido["auc_roc"])
        self.assertAlmostEqual(frouxo["auc_pr"], rigido["auc_pr"])

        # limiar 0,4 marca 3 linhas: 2 fraudes (ambas) e 1 legítima
        self.assertAlmostEqual(frouxo["recall"], 1.0)
        self.assertAlmostEqual(frouxo["precisao"], 2 / 3)

        # limiar 0,6 marca só a de 0,90: 1 fraude de 2, sem falso positivo
        self.assertAlmostEqual(rigido["recall"], 0.5)
        self.assertAlmostEqual(rigido["precisao"], 1.0)

    def test_limiar_e_devolvido_nas_metricas(self) -> None:
        m = avaliar_probabilidades([0, 1], [0.2, 0.8], limiar=0.33)
        self.assertAlmostEqual(m["limiar"], 0.33)

    def test_nenhuma_previsao_positiva_nao_levanta_erro(self) -> None:
        """Precisão sem positivos previstos é indefinida; deve virar 0, não exceção."""
        m = avaliar_probabilidades([0, 1], [0.01, 0.02], limiar=0.5)

        self.assertEqual(m["precisao"], 0.0)
        self.assertEqual(m["recall"], 0.0)
        self.assertEqual(m["f1"], 0.0)


class MatrizConfusaoTest(unittest.TestCase):

    def test_posicoes_da_matriz(self) -> None:
        #        prevista:  leg  fraude
        # real legítima      2      1
        # real fraude        1      2
        y = [0, 0, 0, 1, 1, 1]
        probas = [0.1, 0.2, 0.9, 0.1, 0.8, 0.9]

        matriz = matriz_confusao(y, probas)

        self.assertEqual(matriz.loc["real: legítima", "prevista: legítima"], 2)
        self.assertEqual(matriz.loc["real: legítima", "prevista: fraude"], 1)
        self.assertEqual(matriz.loc["real: fraude", "prevista: legítima"], 1)
        self.assertEqual(matriz.loc["real: fraude", "prevista: fraude"], 2)

    def test_matriz_completa_mesmo_sem_positivos_previstos(self) -> None:
        """Sem `labels=[0, 1]` o sklearn devolveria matriz 1x1 e o ravel quebraria."""
        matriz = matriz_confusao([0, 0, 1], [0.1, 0.1, 0.2], limiar=0.5)

        self.assertEqual(matriz.shape, (2, 2))
        self.assertEqual(matriz.loc["real: fraude", "prevista: legítima"], 1)

    def test_resumo_operacional(self) -> None:
        y = [0, 0, 0, 1, 1, 1]
        probas = [0.1, 0.2, 0.9, 0.1, 0.8, 0.9]

        resumo = resumo_operacional(y, probas)

        self.assertEqual(resumo["fraudes_recuperadas"], 2)
        self.assertEqual(resumo["fraudes_perdidas"], 1)
        self.assertEqual(resumo["alarmes_falsos"], 1)
        self.assertEqual(resumo["transacoes_marcadas"], 3)
        self.assertAlmostEqual(resumo["fracao_marcada"], 0.5)


class CurvaPrecisaoRecallTest(unittest.TestCase):

    def test_devolve_a_taxa_base_como_linha_do_acaso(self) -> None:
        y = [0, 0, 0, 1]

        *_, taxa_base = curva_precisao_recall(y, [0.1, 0.2, 0.3, 0.9])

        self.assertAlmostEqual(taxa_base, 0.25)

    def test_tamanhos_coerentes(self) -> None:
        y = [0, 0, 1, 1]
        precisao, recall, limiares, _ = curva_precisao_recall(y, [0.1, 0.4, 0.6, 0.9])

        self.assertEqual(len(precisao), len(recall))
        self.assertEqual(len(precisao), len(limiares) + 1)


class CompararTest(unittest.TestCase):

    def test_uma_linha_por_modelo_em_ordem_de_leitura(self) -> None:
        a = avaliar_probabilidades([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        b = avaliar_probabilidades([0, 0, 1, 1], [0.4, 0.9, 0.3, 0.6])

        tabela = comparar({"modelo_a": a, "modelo_b": b})

        self.assertEqual(list(tabela.index), ["modelo_a", "modelo_b"])
        self.assertEqual(list(tabela.columns)[0], "auc_pr")
        self.assertNotIn("n_fraudes", tabela.columns)
        self.assertGreater(tabela.loc["modelo_a", "auc_pr"], tabela.loc["modelo_b", "auc_pr"])


if __name__ == "__main__":
    unittest.main()
