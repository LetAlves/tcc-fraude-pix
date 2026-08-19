import unittest

from src.rag.langchain_basics import (
    KeywordRetriever,
    create_study_chain,
    criar_documentos_didaticos,
    render_study_prompt,
)


class LangChainBasicsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = KeywordRetriever(
            documents=criar_documentos_didaticos(),
            k=2,
        )

    def test_retriever_returns_traceable_med_source(self) -> None:
        documents = self.retriever.invoke("MED Resolução Banco Central Pix")

        self.assertGreaterEqual(len(documents), 1)
        self.assertEqual(
            documents[0].metadata["document_id"],
            "bcb-resolucao-103-2021",
        )
        self.assertIn("source_url", documents[0].metadata)

    def test_chain_formats_context_and_safety_rule(self) -> None:
        prompt = render_study_prompt("Como o SHAP ajuda na explicação?")

        self.assertIn("lundberg-lee-2017", prompt)
        self.assertIn("não invente", prompt)
        self.assertIn("não causalidade", prompt)

    def test_chain_rejects_empty_question(self) -> None:
        chain = create_study_chain(self.retriever)

        with self.assertRaises(ValueError):
            chain.invoke("   ")

    def test_unknown_query_reports_insufficient_evidence(self) -> None:
        prompt = render_study_prompt("ornitorrinco quasar vulcânico")

        self.assertIn("Evidência insuficiente", prompt)


if __name__ == "__main__":
    unittest.main()
