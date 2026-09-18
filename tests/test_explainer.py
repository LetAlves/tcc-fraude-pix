"""Testes da construção do prompt e da geração da explicação.

O prompt é testado como função pura, sem rede. As restrições metodológicas que
ele carrega — não afirmar causalidade, não inventar significado para colunas
anônimas, não sair das evidências — são verificadas explicitamente: são a razão
de o prompt existir, e uma edição futura que as remova deve quebrar um teste.
"""

import unittest

from src.rag.explainer import (LIMITE_FATORES, chamar_llm, explicar,
                               montar_prompt)

FATORES = [
    {"feature": "TransactionAmt", "contribuicao": 0.1234},
    {"feature": "C13", "contribuicao": -0.0456},
    {"feature": "valor_atipico_proxy", "contribuicao": 0.0321},
]

DOCUMENTOS = [
    {"fonte": "Resolução BCB nº 103/2021", "texto": "O MED trata de devolução."},
    {"fonte": "FEBRABAN 2024", "texto": "Relatório sobre golpes."},
]


class MontarPromptTest(unittest.TestCase):

    def test_inclui_fatores_com_nome_e_direcao(self):
        prompt = montar_prompt(FATORES, DOCUMENTOS)

        self.assertIn("TransactionAmt", prompt)
        self.assertIn("aumentou", prompt)
        self.assertIn("reduziu", prompt)

    def test_inclui_documentos_com_fonte(self):
        prompt = montar_prompt(FATORES, DOCUMENTOS)

        self.assertIn("Resolução BCB nº 103/2021", prompt)
        self.assertIn("FEBRABAN 2024", prompt)

    def test_carrega_as_tres_restricoes_metodologicas(self):
        prompt = montar_prompt(FATORES, DOCUMENTOS)

        self.assertIn("não causa", prompt.lower().replace("ã", "ã"))
        self.assertIn("anônimas", prompt)
        self.assertIn("somente as evidências", prompt)

    def test_limita_aos_tres_fatores_mais_influentes(self):
        muitos = FATORES + [{"feature": "NAO_DEVE_APARECER", "contribuicao": 0.001}]

        prompt = montar_prompt(muitos, DOCUMENTOS)

        self.assertNotIn("NAO_DEVE_APARECER", prompt)
        self.assertEqual(LIMITE_FATORES, 3)

    def test_sem_documentos_diz_que_nao_houve_recuperacao(self):
        prompt = montar_prompt(FATORES, [])

        self.assertIn("Nenhum documento", prompt)

    def test_probabilidade_e_decisao_aparecem_quando_informadas(self):
        prompt = montar_prompt(FATORES, DOCUMENTOS, probabilidade=0.873, sinalizada=True)

        self.assertIn("87,3%".replace(",", "."), prompt.replace(",", "."))
        self.assertIn("suspeita", prompt)

    def test_e_funcao_pura(self):
        self.assertEqual(
            montar_prompt(FATORES, DOCUMENTOS), montar_prompt(FATORES, DOCUMENTOS))

    def test_sem_fatores_levanta_erro(self):
        with self.assertRaises(ValueError):
            montar_prompt([], DOCUMENTOS)

    def test_aceita_documentos_do_retriever(self):
        from src.rag.retriever import DocumentoRecuperado

        documentos = [DocumentoRecuperado("texto da norma", 0.9, {"fonte": "BCB"})]

        prompt = montar_prompt(FATORES, documentos)

        self.assertIn("BCB", prompt)
        self.assertIn("texto da norma", prompt)


class ExplicarTest(unittest.TestCase):

    def test_usa_o_cliente_injetado_e_devolve_o_texto(self):
        chamadas = []

        def cliente(prompt):
            chamadas.append(prompt)
            return "  A transação foi sinalizada porque...  "

        resposta = explicar(FATORES, DOCUMENTOS, cliente=cliente)

        self.assertEqual(resposta, "A transação foi sinalizada porque...")
        self.assertEqual(len(chamadas), 1)
        self.assertIn("TransactionAmt", chamadas[0])

    def test_resposta_vazia_do_llm_levanta_erro(self):
        for vazia in ("", "   ", None):
            with self.assertRaises(RuntimeError):
                explicar(FATORES, DOCUMENTOS, cliente=lambda _: vazia)

    def test_sem_cliente_aponta_a_integracao_pendente(self):
        with self.assertRaises(NotImplementedError) as contexto:
            explicar(FATORES, DOCUMENTOS)
        self.assertIn("PREENCHER", str(contexto.exception))

    def test_chamar_llm_ainda_nao_esta_implementado(self):
        with self.assertRaises(NotImplementedError):
            chamar_llm("qualquer prompt")


if __name__ == "__main__":
    unittest.main()
