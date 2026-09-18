"""Testes de integração do pipeline completo.

Todos os componentes externos são dublês: carregar o modelo real, construir o
explicador SHAP e abrir o índice FAISS levaria minutos e exigiria artefatos que
não estão no repositório. O que se verifica aqui é a orquestração — ordem das
etapas, seleção dos fatores e formato do resultado.

O teste mais importante é o do contrato: `app.py` valida o retorno do pipeline,
e um campo renomeado aqui quebraria a interface sem aviso.
"""

import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.pipeline import TOP_FATORES, Pipeline, _como_quadro


def _pipeline_falso(probabilidade=0.87, com_rag=True):
    preprocessador = Mock()
    preprocessador.transform = Mock(return_value=np.zeros((1, 5), dtype="float32"))

    modelo = Mock()
    modelo.predict_proba = Mock(return_value=np.array([[1 - probabilidade, probabilidade]]))

    explicador = Mock()
    # contribuições propositalmente fora de ordem, com sinais diferentes
    explicador.shap_values = Mock(
        return_value=np.array([[0.05, -0.30, 0.10, 0.01, -0.02]]))

    recuperador = None
    if com_rag:
        from src.rag.retriever import DocumentoRecuperado

        recuperador = Mock()
        recuperador.recuperar = Mock(
            return_value=[DocumentoRecuperado("texto da norma", 0.91, {"fonte": "BCB 103/2021"})])

    return Pipeline(
        preprocessador=preprocessador,
        modelo=modelo,
        explicador_shap=explicador,
        recuperador=recuperador,
        nomes_features=["f0", "f1", "f2", "f3", "f4"],
        cliente_llm=lambda prompt: "Explicação gerada.",
    )


TRANSACAO = {"TransactionAmt": 372.5, "card1": 1234}


class ContratoDoResultadoTest(unittest.TestCase):
    """O formato é exigido por app.py; renomear campo aqui quebra a interface."""

    def setUp(self):
        self.resultado = _pipeline_falso().processar(TRANSACAO)

    def test_campos_obrigatorios_do_app(self):
        for campo in ("predicao", "fatores_shap", "explicacao_rag"):
            self.assertIn(campo, self.resultado)

    def test_fatores_shap_e_lista_de_objetos(self):
        fatores = self.resultado["fatores_shap"]
        self.assertIsInstance(fatores, list)
        self.assertTrue(all(isinstance(fator, dict) for fator in fatores))

    def test_explicacao_e_texto_nao_vazio(self):
        self.assertIsInstance(self.resultado["explicacao_rag"], str)
        self.assertTrue(self.resultado["explicacao_rag"].strip())

    def test_traz_probabilidade_e_documentos(self):
        self.assertAlmostEqual(self.resultado["probabilidade"], 0.87)
        self.assertEqual(len(self.resultado["documentos_recuperados"]), 1)
        self.assertEqual(
            self.resultado["documentos_recuperados"][0]["fonte"], "BCB 103/2021")


class SelecaoDosFatoresTest(unittest.TestCase):

    def test_escolhe_por_contribuicao_absoluta_preservando_o_sinal(self):
        fatores = _pipeline_falso().processar(TRANSACAO)["fatores_shap"]

        self.assertEqual(len(fatores), TOP_FATORES)
        # |−0,30| > |0,10| > |0,05|
        self.assertEqual([f["feature"] for f in fatores], ["f1", "f2", "f0"])
        self.assertLess(fatores[0]["contribuicao"], 0)
        self.assertEqual(fatores[0]["direcao"], "reduz")
        self.assertEqual(fatores[1]["direcao"], "aumenta")

    def test_saida_tridimensional_usa_a_classe_positiva(self):
        pipeline = _pipeline_falso()
        pipeline.explicador_shap.shap_values = Mock(
            return_value=np.zeros((1, 5, 2)))

        fatores = pipeline.processar(TRANSACAO)["fatores_shap"]

        self.assertEqual(len(fatores), TOP_FATORES)


class LimiarTest(unittest.TestCase):

    def test_probabilidade_acima_do_limiar_e_suspeita(self):
        resultado = _pipeline_falso(probabilidade=0.80).processar(TRANSACAO)
        self.assertEqual(resultado["predicao"], "suspeita")

    def test_probabilidade_abaixo_do_limiar_nao_e_suspeita(self):
        resultado = _pipeline_falso(probabilidade=0.10).processar(TRANSACAO)
        self.assertEqual(resultado["predicao"], "não suspeita")

    def test_limiar_configuravel_muda_a_decisao(self):
        pipeline = _pipeline_falso(probabilidade=0.30)
        pipeline.limiar = 0.20

        self.assertEqual(pipeline.processar(TRANSACAO)["predicao"], "suspeita")


class SemRagTest(unittest.TestCase):

    def test_pipeline_funciona_sem_recuperador(self):
        resultado = _pipeline_falso(com_rag=False).processar(TRANSACAO)

        self.assertEqual(resultado["documentos_recuperados"], [])
        self.assertTrue(resultado["explicacao_rag"])


class EntradaTest(unittest.TestCase):

    def test_aceita_dicionario_e_dataframe_de_uma_linha(self):
        self.assertEqual(len(_como_quadro(TRANSACAO)), 1)
        self.assertEqual(len(_como_quadro(pd.DataFrame([TRANSACAO]))), 1)

    def test_recusa_dataframe_com_varias_linhas(self):
        with self.assertRaises(ValueError):
            _como_quadro(pd.DataFrame([TRANSACAO, TRANSACAO]))

    def test_recusa_dicionario_vazio(self):
        with self.assertRaises(ValueError):
            _como_quadro({})

    def test_recusa_tipo_invalido(self):
        for invalido in ("texto", 42, None, [TRANSACAO]):
            with self.assertRaises(TypeError):
                _como_quadro(invalido)

    def test_preprocessador_e_aplicado_antes_do_modelo(self):
        """Prever sobre dados crus não levanta exceção: devolve número errado."""
        pipeline = _pipeline_falso()
        pipeline.processar(TRANSACAO)

        pipeline.preprocessador.transform.assert_called_once()
        matriz = pipeline.modelo.predict_proba.call_args[0][0]
        self.assertEqual(matriz.shape, (1, 5))


if __name__ == "__main__":
    unittest.main()
