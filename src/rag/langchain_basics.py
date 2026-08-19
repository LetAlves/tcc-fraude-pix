"""Laboratório determinístico dos conceitos básicos usados no RAG.

Este módulo não chama um LLM e não pretende substituir o pipeline previsto para
julho/agosto. Ele demonstra, sem custo e de forma testável, três abstrações do
LangChain: documentos, retriever e composição de uma chain com prompt template.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Any

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda


MAX_QUESTION_LENGTH = 500


def _normalizar_tokens(texto: str) -> set[str]:
    """Converte texto em tokens simples para a busca didática por palavras."""

    normalizado = unicodedata.normalize("NFKD", texto.casefold())
    sem_acentos = "".join(char for char in normalizado if not unicodedata.combining(char))
    return set(re.findall(r"[a-z0-9]{3,}", sem_acentos))


class KeywordRetriever(BaseRetriever):
    """Retriever didático que ranqueia documentos por sobreposição lexical.

    O retriever real do TCC usará embeddings e FAISS. Esta implementação existe
    apenas para exercitar o contrato ``invoke(query) -> list[Document]`` sem
    download de modelo, chave de API ou acesso à rede.
    """

    documents: list[Document]
    k: int = 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Any = None,
    ) -> list[Document]:
        query_tokens = _normalizar_tokens(query)
        if not query_tokens:
            return []

        ranked: list[tuple[int, str, Document]] = []
        for document in self.documents:
            document_tokens = _normalizar_tokens(document.page_content)
            score = len(query_tokens & document_tokens)
            if score > 0:
                document_id = str(document.metadata.get("document_id", ""))
                ranked.append((score, document_id, document))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [document for _, _, document in ranked[: self.k]]


PROMPT_EXPLICACAO = PromptTemplate.from_template(
    """Você apoia a explicação de risco de um protótipo acadêmico.
Use somente o CONTEXTO recuperado. Não determine que houve fraude, não invente
o significado de atributos anônimos e informe quando a evidência for insuficiente.

PERGUNTA:
{question}

CONTEXTO:
{context}

SAÍDA ESPERADA:
- síntese objetiva;
- fontes usadas;
- limitação aplicável.
"""
)


def criar_documentos_didaticos() -> list[Document]:
    """Cria um corpus mínimo com metadados rastreáveis para o laboratório."""

    return [
        Document(
            page_content=(
                "O Mecanismo Especial de Devolução foi introduzido no Regulamento "
                "do Pix pela Resolução BCB nº 103, de 8 de junho de 2021. Regras "
                "operacionais devem ser conferidas no regulamento e guia vigentes."
            ),
            metadata={
                "document_id": "bcb-resolucao-103-2021",
                "issuer": "Banco Central do Brasil",
                "source_url": (
                    "https://www.bcb.gov.br/estabilidadefinanceira/"
                    "exibenormativo?numero=103&tipo=Resolu%C3%A7%C3%A3o+BCB"
                ),
                "version": "2021-06-08",
            },
        ),
        Document(
            page_content=(
                "SHAP atribui contribuições locais aos atributos usados pelo modelo. "
                "A contribuição descreve influência estatística na previsão, não "
                "causalidade, intenção criminosa ou significado oculto de uma coluna."
            ),
            metadata={
                "document_id": "lundberg-lee-2017",
                "issuer": "NeurIPS",
                "source_url": (
                    "https://papers.nips.cc/paper/2017/hash/"
                    "8a20a8621978632d76c43dfd28b67767-Abstract.html"
                ),
                "version": "2017",
            },
        ),
        Document(
            page_content=(
                "Retrieval-Augmented Generation recupera evidências externas antes "
                "da geração. A resposta continua exigindo validação de que cada "
                "afirmação é sustentada pelos trechos recuperados."
            ),
            metadata={
                "document_id": "lewis-et-al-2020-rag",
                "issuer": "NeurIPS",
                "source_url": (
                    "https://papers.nips.cc/paper_files/paper/2020/hash/"
                    "6b493230205f780e1bc26945df7481e5-Abstract.html"
                ),
                "version": "2020",
            },
        ),
    ]


def format_documents(documents: Sequence[Document]) -> str:
    """Formata conteúdo e metadados mínimos para entrada no prompt."""

    if not documents:
        return "Evidência insuficiente: nenhum documento relevante foi recuperado."

    blocks = []
    for document in documents:
        metadata = document.metadata
        source = metadata.get("source_url", "fonte não informada")
        document_id = metadata.get("document_id", "documento-sem-id")
        blocks.append(
            f"[{document_id}] {document.page_content}\nFonte: {source}"
        )
    return "\n\n".join(blocks)


def _validar_pergunta(question: object) -> str:
    if not isinstance(question, str):
        raise TypeError("A pergunta deve ser uma string.")

    normalized = question.strip()
    if not normalized:
        raise ValueError("A pergunta não pode ser vazia.")
    if len(normalized) > MAX_QUESTION_LENGTH:
        raise ValueError(
            f"A pergunta deve ter no máximo {MAX_QUESTION_LENGTH} caracteres."
        )
    return normalized


def create_study_chain(retriever: BaseRetriever) -> Runnable:
    """Compõe recuperação e prompt usando a interface Runnable do LangChain."""

    def retrieve_context(question: object) -> dict[str, str]:
        validated_question = _validar_pergunta(question)
        documents = retriever.invoke(validated_question)
        return {
            "question": validated_question,
            "context": format_documents(documents),
        }

    return RunnableLambda(retrieve_context) | PROMPT_EXPLICACAO


def render_study_prompt(question: str) -> str:
    """Executa o laboratório e devolve o prompt final, sem chamar um LLM."""

    retriever = KeywordRetriever(documents=criar_documentos_didaticos(), k=2)
    prompt_value = create_study_chain(retriever).invoke(question)
    return prompt_value.to_string()


if __name__ == "__main__":
    print(render_study_prompt("Qual é o papel do MED no Pix?"))
