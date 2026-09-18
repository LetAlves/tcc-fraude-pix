"""
Camada 3 (parte 1): recuperação documental por similaridade de cosseno.

Orquestra o que já existe no projeto — `MultilingualEmbedder` para transformar a
consulta em vetor e `FaissVectorStore` para a busca — e devolve um resultado
pronto para o explicador consumir: texto, score e metadados de proveniência.

**Este módulo não reimplementa busca vetorial.** O cosseno já é feito pelo
`IndexFlatIP` do `FaissVectorStore`, que normaliza os vetores; duplicar essa
lógica aqui criaria duas implementações da mesma conta, com a garantia de que
uma delas ficaria desatualizada.

A proveniência viaja junto com o texto porque o RAG deste trabalho recupera
documentos regulatórios: uma explicação que cita o MED sem dizer de qual norma
e de qual versão veio não é verificável, e verificabilidade é o motivo de haver
RAG em vez de só pedir a explicação ao modelo de linguagem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.rag.vector_store import DEFAULT_INDEX_DIR, FaissVectorStore

logger = logging.getLogger(__name__)

TOP_K_PADRAO = 5


@dataclass(frozen=True)
class DocumentoRecuperado:
    """Um documento devolvido pela busca, com score e proveniência."""

    texto: str
    score: float
    metadados: dict[str, Any] = field(default_factory=dict)

    @property
    def fonte(self) -> str:
        """Identificação curta da origem, para citar na explicação."""
        for chave in ("fonte", "source", "titulo", "title", "documento"):
            valor = self.metadados.get(chave)
            if valor:
                return str(valor)
        return "fonte não identificada"

    def resumo(self, limite: int = 400) -> str:
        """Trecho do texto, para caber no prompt sem estourar o contexto."""
        texto = " ".join(self.texto.split())
        return texto if len(texto) <= limite else texto[: limite - 1] + "…"


class RecuperadorDocumentos:
    """
    Busca documentos relevantes para uma consulta em linguagem natural.

    O embedder e o índice são injetados para que os testes possam usar dublês
    sem baixar o modelo de embeddings nem construir um índice FAISS real.
    """

    def __init__(self, embedder: Any, indice: FaissVectorStore) -> None:
        if embedder is None or indice is None:
            raise ValueError("embedder e índice são obrigatórios")
        self.embedder = embedder
        self.indice = indice

    @classmethod
    def a_partir_do_disco(
        cls,
        diretorio_indice: Path | str = DEFAULT_INDEX_DIR,
        embedder: Any | None = None,
    ) -> "RecuperadorDocumentos":
        """
        Carrega o índice persistido e, se necessário, constrói o embedder.

        O embedder padrão baixa o modelo de embeddings na primeira execução; em
        teste, injete um dublê em vez de chamar este construtor.
        """
        indice = FaissVectorStore.load(Path(diretorio_indice))
        if embedder is None:
            from src.rag.embeddings import MultilingualEmbedder

            embedder = MultilingualEmbedder()
        return cls(embedder, indice)

    def recuperar(self, consulta: str, top_k: int = TOP_K_PADRAO) -> list[DocumentoRecuperado]:
        """
        Devolve até `top_k` documentos ordenados por similaridade decrescente.

        Levanta `ValueError` para consulta vazia ou `top_k` não positivo — falhar
        aqui é melhor que devolver lista vazia, que o chamador confundiria com
        "nenhum documento relevante".
        """
        if not isinstance(consulta, str) or not consulta.strip():
            raise ValueError("a consulta não pode ser vazia")
        if top_k <= 0:
            raise ValueError("top_k deve ser positivo")

        if len(self.indice.documents) == 0:
            logger.warning("índice vetorial vazio: nenhuma recuperação possível")
            return []

        vetor = self.embedder.encode_query(consulta)
        resultados = self.indice.search(vetor, k=top_k)

        return [
            DocumentoRecuperado(
                texto=resultado.document.page_content,
                score=float(resultado.score),
                metadados=dict(resultado.document.metadata or {}),
            )
            for resultado in resultados
        ]


def consulta_a_partir_dos_fatores(fatores: list[dict[str, Any]], transacao_suspeita: bool) -> str:
    """
    Monta a consulta ao RAG a partir dos fatores mais influentes do SHAP.

    A consulta é construída com os **nomes das variáveis**, não com seus valores:
    a base documental contém normas e relatórios sobre padrões de fraude, não
    registros de transações. Procurar por "TransactionAmt igual a 372,50" não
    encontraria nada; procurar pelos conceitos associados, sim.

    Nomes de colunas anônimas entram como estão. Traduzi-los para conceitos de
    negócio seria inventar semântica que o dataset não publica — a mesma regra
    que vale para o SHAP.
    """
    if not fatores:
        raise ValueError("é necessário ao menos um fator para montar a consulta")

    nomes = [str(fator.get("feature", "")).strip() for fator in fatores]
    nomes = [nome for nome in nomes if nome]
    if not nomes:
        raise ValueError("os fatores não contêm nomes de variáveis utilizáveis")

    situacao = "transação sinalizada como suspeita" if transacao_suspeita else "transação não sinalizada"
    return (
        f"{situacao}; fatores de maior influência na decisão: {', '.join(nomes)}. "
        "Normas, prazos e procedimentos aplicáveis a fraude em pagamentos instantâneos."
    )
