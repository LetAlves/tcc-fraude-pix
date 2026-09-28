"""Testes da construção do prompt e da geração da explicação.

O prompt é testado como função pura, sem rede. As restrições metodológicas que
ele carrega — não afirmar causalidade, não inventar significado para colunas
anônimas, não sair das evidências — são verificadas explicitamente: são a razão
de o prompt existir, e uma edição futura que as remova deve quebrar um teste.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.rag.explainer import (
    LIMITE_FATORES,
    ClienteAnthropic,
    ErroLLM,
    LLMIndisponivelError,
    chamar_llm,
    criar_cliente_llm_de_ambiente,
    explicar,
    montar_prompt,
    selecionar_top_fatores,
)

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
        muitos = [
            {"feature": "NAO_DEVE_APARECER", "contribuicao": 0.001},
            *reversed(FATORES),
        ]

        prompt = montar_prompt(muitos, DOCUMENTOS)

        self.assertNotIn("NAO_DEVE_APARECER", prompt)
        self.assertLess(prompt.index("TransactionAmt"), prompt.index("C13"))
        self.assertEqual(LIMITE_FATORES, 3)

    def test_recusa_contribuicao_invalida(self):
        for invalida in (None, float("nan"), float("inf"), "0.1"):
            with self.assertRaises(ValueError):
                selecionar_top_fatores([{"feature": "f", "contribuicao": invalida}])

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
                explicar(FATORES, DOCUMENTOS, cliente=lambda _, valor=vazia: valor)

    def test_sem_cliente_aponta_a_integracao_pendente(self):
        with patch(
            "src.rag.explainer.criar_cliente_llm_de_ambiente", return_value=None
        ), self.assertRaises(LLMIndisponivelError) as contexto:
            explicar(FATORES, DOCUMENTOS)
        self.assertIn("ANTHROPIC_API_KEY", str(contexto.exception))

    def test_chamar_llm_sem_configuracao_falha_de_forma_controlada(self):
        with patch(
            "src.rag.explainer.criar_cliente_llm_de_ambiente", return_value=None
        ), self.assertRaises(LLMIndisponivelError):
            chamar_llm("qualquer prompt")


class ClienteAnthropicTest(unittest.TestCase):

    def test_extrai_somente_blocos_de_texto_e_envia_configuracao(self):
        api = Mock()
        api.messages.create.return_value = SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text=" Explicação apoiada. "),
                SimpleNamespace(type="tool_use", text="ignorar"),
            ],
        )
        cliente = ClienteAnthropic(
            "chave-de-teste", model="modelo-teste", max_tokens=321, client=api
        )

        self.assertEqual(cliente("prompt"), "Explicação apoiada.")
        api.messages.create.assert_called_once_with(
            model="modelo-teste",
            max_tokens=321,
            temperature=0.0,
            messages=[{"role": "user", "content": "prompt"}],
        )

    def test_falha_do_provedor_e_encadeada_sem_expor_prompt(self):
        api = Mock()
        api.messages.create.side_effect = TimeoutError("segredo-no-erro")
        cliente = ClienteAnthropic("chave-de-teste", client=api)

        with self.assertRaisesRegex(ErroLLM, "não foi possível") as contexto:
            cliente("prompt sensível")
        self.assertIsInstance(contexto.exception.__cause__, TimeoutError)
        self.assertNotIn("prompt sensível", str(contexto.exception))

    def test_recusa_e_resposta_truncada_nao_sao_aceitas(self):
        for motivo in ("refusal", "max_tokens"):
            api = Mock()
            api.messages.create.return_value = SimpleNamespace(
                stop_reason=motivo,
                content=[SimpleNamespace(type="text", text="parcial")],
            )
            with self.assertRaises(ErroLLM):
                ClienteAnthropic("chave-de-teste", client=api)("prompt")

    def test_limita_custo_e_tempo_configuraveis(self):
        for configuracao in ({"max_tokens": 2_001}, {"timeout": 121.0}):
            with self.assertRaises(ValueError):
                ClienteAnthropic("chave-de-teste", client=Mock(), **configuracao)

    def test_configuracao_vem_do_ambiente(self):
        cliente_falso = Mock()
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "teste",
                "TCC_LLM_MODEL": "modelo-teste",
                "TCC_LLM_MAX_TOKENS": "222",
                "TCC_LLM_TIMEOUT": "12.5",
            },
            clear=True,
        ), patch("src.rag.explainer.ClienteAnthropic", return_value=cliente_falso) as classe:
            self.assertIs(criar_cliente_llm_de_ambiente(), cliente_falso)
        classe.assert_called_once_with(
            "teste", model="modelo-teste", max_tokens=222, timeout=12.5
        )


if __name__ == "__main__":
    unittest.main()
