"""Testes da recuperação documental.

Usa dublês para embedder e índice: baixar o modelo de embeddings e construir um
índice FAISS real tornaria os testes lentos e dependentes de rede, e o que se
verifica aqui é a orquestração, não a matemática do cosseno — essa já é testada
em `test_rag_vector_store.py`.
"""

import unittest
from unittest.mock import Mock

import numpy as np
from langchain_core.documents import Document

from src.rag.retriever import (DocumentoRecuperado, RecuperadorDocumentos,
                               consulta_a_partir_dos_fatores)
from src.rag.vector_store import SearchResult


def _indice_falso(documentos):
    indice = Mock()
    indice.documents = documentos
    indice.search = Mock(
        return_value=[
            SearchResult(documento, 0.9 - 0.1 * posicao)
            for posicao, documento in enumerate(documentos)
        ]
    )
    return indice


def _embedder_falso():
    embedder = Mock()
    embedder.encode_query = Mock(return_value=np.zeros((1, 384), dtype="float32"))
    return embedder


DOCUMENTOS = [
    Document(page_content="O MED permite devolução em casos de fraude.",
             metadata={"fonte": "Resolução BCB nº 103/2021", "versao": "1.0"}),
    Document(page_content="Relatório sobre golpes em pagamentos instantâneos.",
             metadata={"titulo": "FEBRABAN 2024"}),
]


class RecuperadorTest(unittest.TestCase):

    def setUp(self):
        self.recuperador = RecuperadorDocumentos(_embedder_falso(), _indice_falso(DOCUMENTOS))

    def test_devolve_texto_score_e_metadados(self):
        resultados = self.recuperador.recuperar("fraude em pix", top_k=2)

        self.assertEqual(len(resultados), 2)
        self.assertIsInstance(resultados[0], DocumentoRecuperado)
        self.assertIn("MED", resultados[0].texto)
        self.assertAlmostEqual(resultados[0].score, 0.9)
        self.assertEqual(resultados[0].metadados["versao"], "1.0")

    def test_top_k_e_repassado_ao_indice(self):
        self.recuperador.recuperar("consulta", top_k=3)
        _, kwargs = self.recuperador.indice.search.call_args
        self.assertEqual(kwargs["k"], 3)

    def test_consulta_vazia_e_recusada(self):
        for invalida in ("", "   ", None):
            with self.assertRaises(ValueError):
                self.recuperador.recuperar(invalida)

    def test_top_k_nao_positivo_e_recusado(self):
        for invalido in (0, -1):
            with self.assertRaises(ValueError):
                self.recuperador.recuperar("consulta", top_k=invalido)

    def test_indice_vazio_devolve_lista_vazia_sem_consultar(self):
        recuperador = RecuperadorDocumentos(_embedder_falso(), _indice_falso([]))

        self.assertEqual(recuperador.recuperar("consulta"), [])
        recuperador.indice.search.assert_not_called()

    def test_componentes_obrigatorios(self):
        with self.assertRaises(ValueError):
            RecuperadorDocumentos(None, _indice_falso(DOCUMENTOS))
        with self.assertRaises(ValueError):
            RecuperadorDocumentos(_embedder_falso(), None)


class DocumentoRecuperadoTest(unittest.TestCase):

    def test_fonte_procura_em_varias_chaves(self):
        self.assertEqual(
            DocumentoRecuperado("t", 1.0, {"fonte": "Res. 103"}).fonte, "Res. 103")
        self.assertEqual(
            DocumentoRecuperado("t", 1.0, {"titulo": "FEBRABAN"}).fonte, "FEBRABAN")

    def test_fonte_ausente_nao_quebra(self):
        self.assertEqual(DocumentoRecuperado("t", 1.0, {}).fonte, "fonte não identificada")

    def test_resumo_trunca_e_normaliza_espacos(self):
        documento = DocumentoRecuperado("palavra  " * 200, 1.0)
        resumo = documento.resumo(limite=50)

        self.assertLessEqual(len(resumo), 50)
        self.assertTrue(resumo.endswith("…"))
        self.assertNotIn("  ", resumo)

    def test_resumo_preserva_texto_curto(self):
        self.assertEqual(DocumentoRecuperado("curto", 1.0).resumo(), "curto")


class ConsultaAPartirDosFatoresTest(unittest.TestCase):

    def test_usa_os_nomes_das_variaveis(self):
        consulta = consulta_a_partir_dos_fatores(
            [{"feature": "TransactionAmt"}, {"feature": "C13"}], transacao_suspeita=True)

        self.assertIn("TransactionAmt", consulta)
        self.assertIn("C13", consulta)
        self.assertIn("suspeita", consulta)

    def test_distingue_sinalizada_de_nao_sinalizada(self):
        suspeita = consulta_a_partir_dos_fatores([{"feature": "X"}], True)
        normal = consulta_a_partir_dos_fatores([{"feature": "X"}], False)

        self.assertNotEqual(suspeita, normal)

    def test_sem_fatores_ou_sem_nomes_levanta_erro(self):
        with self.assertRaises(ValueError):
            consulta_a_partir_dos_fatores([], True)
        with self.assertRaises(ValueError):
            consulta_a_partir_dos_fatores([{"contribuicao": 0.1}], True)


if __name__ == "__main__":
    unittest.main()
