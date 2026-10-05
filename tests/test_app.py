"""Testes do contrato da interface Streamlit, sem executar o pipeline."""

from __future__ import annotations

import unittest

from app import (
    IntegracaoPendenteError,
    _classe_suspeita,
    carregar_pipeline,
    ler_transacao_json,
    validar_resultado_pipeline,
)


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


class ClassificacaoVisualTest(unittest.TestCase):
    def test_reconhece_rotulos_suspeitos_textuais_e_binarios(self) -> None:
        for classe in (1, 1.0, "1", "fraude", "suspeita"):
            with self.subTest(classe=classe):
                self.assertTrue(_classe_suspeita(classe))

    def test_nao_confunde_rotulos_negativos(self) -> None:
        for classe in (0, "0", "não suspeita", "legítima", None):
            with self.subTest(classe=classe):
                self.assertFalse(_classe_suspeita(classe))


if __name__ == "__main__":
    unittest.main()
