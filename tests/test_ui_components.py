"""Testes das explicações determinísticas usadas pelo dashboard."""

from __future__ import annotations

import math
import unittest

from src.ui.components import descrever_feature, preparar_fatores_shap


class MetadadosFeatureTest(unittest.TestCase):
    def test_explica_valor_atipico_criado_no_projeto(self) -> None:
        metadados = descrever_feature("valor_atipico_proxy")

        self.assertEqual(metadados["rotulo"], "Valor fora do padrão")
        self.assertIn("se afasta do histórico", metadados["descricao"])

    def test_explica_frequencia_recente_criada_no_projeto(self) -> None:
        metadados = descrever_feature("frequencia_recente_proxy")

        self.assertEqual(metadados["rotulo"], "Frequência recente")
        self.assertIn("transações anteriores", metadados["descricao"])

    def test_explica_dispositivo_raro_criado_no_projeto(self) -> None:
        metadados = descrever_feature("dispositivo_raro_proxy")

        self.assertEqual(metadados["rotulo"], "Dispositivo pouco frequente")
        self.assertIn("frequência histórica", metadados["descricao"])

    def test_reconhece_nome_com_prefixo_do_transformador(self) -> None:
        metadados = descrever_feature("num__posicao_ciclo_diario_relativa")

        self.assertEqual(metadados["rotulo"], "Posição no ciclo diário")
        self.assertEqual(
            metadados["nome_tecnico"], "num__posicao_ciclo_diario_relativa"
        )


class PreparacaoFatoresTest(unittest.TestCase):
    def test_preserva_valor_e_contribuicao_validos(self) -> None:
        fatores = preparar_fatores_shap(
            [
                {
                    "feature": "frequencia_recente_proxy",
                    "valor": 22,
                    "contribuicao": 0.31,
                }
            ]
        )

        self.assertEqual(fatores[0]["valor"], 22)
        self.assertEqual(fatores[0]["contribuicao"], 0.31)

    def test_descarta_contribuicoes_invalidas(self) -> None:
        fatores = preparar_fatores_shap(
            [
                {"feature": "valor_atipico_proxy", "contribuicao": math.nan},
                {"feature": "frequencia_recente_proxy", "contribuicao": "0.2"},
                {"feature": "dispositivo_raro_proxy", "contribuicao": True},
            ]
        )

        self.assertEqual(fatores, [])

    def test_omite_variaveis_anonimas_do_ieee_cis(self) -> None:
        fatores = preparar_fatores_shap(
            [
                {"feature": "C13", "valor": 4, "contribuicao": 0.8},
                {"feature": "card1", "valor": 1234, "contribuicao": -0.5},
                {"feature": "TransactionAmt", "valor": 2500, "contribuicao": 0.4},
                {
                    "feature": "valor_atipico_proxy",
                    "valor": 2.3,
                    "contribuicao": 0.1,
                },
            ]
        )

        self.assertEqual(
            [fator["nome_tecnico"] for fator in fatores], ["valor_atipico_proxy"]
        )


if __name__ == "__main__":
    unittest.main()
