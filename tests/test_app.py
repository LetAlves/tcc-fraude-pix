"""Testes do contrato da interface Streamlit, sem executar o pipeline."""

from __future__ import annotations

import unittest

from app import IntegracaoPendenteError, carregar_pipeline, ler_transacao_json
from app import validar_resultado_pipeline


class EntradaTransacaoTest(unittest.TestCase):
    def test_aceita_objeto_json(self) -> None:
        self.assertEqual(
            ler_transacao_json('{"TransactionID": 123, "TransactionAmt": 45.5}'),
            {"TransactionID": 123, "TransactionAmt": 45.5},
        )

    def test_recusa_json_invalido(self) -> None:
        with self.assertRaisesRegex(ValueError, "JSON inválido"):
            ler_transacao_json('{"TransactionID":')

    def test_recusa_lista_e_objeto_vazio(self) -> None:
        with self.assertRaisesRegex(ValueError, "objeto JSON"):
            ler_transacao_json("[]")
        with self.assertRaisesRegex(ValueError, "não pode ser vazia"):
            ler_transacao_json("{}")


class ContratoPipelineTest(unittest.TestCase):
    def test_normaliza_resposta_valida(self) -> None:
        resultado = validar_resultado_pipeline(
            {
                "predicao": {"classe": "suspeita", "probabilidade": 0.8},
                "fatores_shap": [
                    {"feature": "valor_atipico_proxy", "contribuicao": 0.1}
                ],
                "explicacao_rag": "  Explicação apoiada nas fontes.  ",
                "documentos_recuperados": None,
            }
        )

        self.assertEqual(resultado["explicacao_rag"], "Explicação apoiada nas fontes.")
        self.assertEqual(resultado["documentos_recuperados"], [])

    def test_recusa_resposta_incompleta(self) -> None:
        with self.assertRaisesRegex(ValueError, "campos ausentes"):
            validar_resultado_pipeline({"predicao": {"classe": 1}})

    def test_exige_classe_na_predicao(self) -> None:
        with self.assertRaisesRegex(ValueError, "chave 'classe'"):
            validar_resultado_pipeline(
                {
                    "predicao": {},
                    "fatores_shap": [],
                    "explicacao_rag": "Texto.",
                }
            )

    def test_sinaliza_integracao_nao_configurada(self) -> None:
        carregar_pipeline.clear()
        with self.assertRaisesRegex(IntegracaoPendenteError, "PREENCHER"):
            carregar_pipeline("{{PREENCHER}}", "{{PREENCHER}}")


if __name__ == "__main__":
    unittest.main()
