"""Embeddings multilíngues sem truncamento silencioso dos chunks do corpus."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from langchain_core.documents import Document

from src.rag.knowledge_base import TokenizerProtocol, encode_without_truncation


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
DEFAULT_EMBEDDING_OVERLAP_TOKENS = 24


def _special_token_count(tokenizer: Any) -> int:
    method = getattr(tokenizer, "num_special_tokens_to_add", None)
    if not callable(method):
        return 2
    try:
        return int(method(pair=False))
    except TypeError:
        return int(method(False))


def embedding_content_limit(tokenizer: Any, max_sequence_length: int) -> int:
    """Calcula o limite de conteúdo após reservar os tokens especiais."""

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length deve ser positivo.")
    limit = max_sequence_length - _special_token_count(tokenizer)
    if limit <= 0:
        raise ValueError("O modelo não possui espaço para tokens de conteúdo.")
    return limit


def create_embedding_documents(
    parent_chunks: Sequence[Document],
    tokenizer: TokenizerProtocol,
    max_sequence_length: int,
    *,
    overlap_tokens: int = DEFAULT_EMBEDDING_OVERLAP_TOKENS,
) -> list[Document]:
    """Divide chunks pais em janelas compatíveis e preserva sua proveniência."""

    content_limit = embedding_content_limit(tokenizer, max_sequence_length)
    if overlap_tokens < 0 or overlap_tokens >= content_limit:
        raise ValueError("overlap_tokens é incompatível com o limite do modelo.")
    step = content_limit - overlap_tokens

    children: list[Document] = []
    for parent in parent_chunks:
        parent_chunk_id = parent.metadata.get("chunk_id")
        if not isinstance(parent_chunk_id, str) or not parent_chunk_id:
            raise ValueError("Todo chunk pai deve possuir chunk_id.")
        token_ids = encode_without_truncation(tokenizer, parent.page_content)
        window_index = 0
        for token_start in range(0, len(token_ids), step):
            window = token_ids[token_start : token_start + content_limit]
            if not window:
                break
            token_end = token_start + len(window)
            text = tokenizer.decode(window, skip_special_tokens=True).strip()
            if not text:
                continue

            identity = f"{parent_chunk_id}|{token_start}|{token_end}"
            embedding_chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            children.append(
                Document(
                    page_content=text,
                    metadata={
                        **parent.metadata,
                        "parent_chunk_id": parent_chunk_id,
                        "embedding_chunk_id": embedding_chunk_id,
                        "embedding_window_index": window_index,
                        "embedding_token_start": token_start,
                        "embedding_token_end": token_end,
                        "embedding_token_count": len(window),
                        "embedding_content_limit": content_limit,
                        "embedding_overlap_tokens": overlap_tokens,
                    },
                )
            )
            window_index += 1
            if token_end >= len(token_ids):
                break
    return children


def _normalized_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("O modelo retornou uma matriz de embeddings inválida.")
    if not np.isfinite(matrix).all():
        raise ValueError("O modelo retornou embedding com NaN ou infinito.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("O modelo retornou embedding de norma zero.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


class MultilingualEmbedder:
    """Adapter testável para o MiniLM multilíngue definido no cronograma."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        model: Any | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        if model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:  # pragma: no cover - mensagem de ambiente
                raise RuntimeError("Instale sentence-transformers para gerar embeddings.") from error
            options = {"revision": revision} if revision else {}
            model = SentenceTransformer(model_name, local_files_only=local_files_only, **options)
        self.model = model
        self.tokenizer = model.tokenizer
        self.max_sequence_length = int(model.max_seq_length)
        self.content_limit = embedding_content_limit(
            self.tokenizer, self.max_sequence_length
        )

    def create_documents(
        self,
        parent_chunks: Sequence[Document],
        *,
        overlap_tokens: int = DEFAULT_EMBEDDING_OVERLAP_TOKENS,
    ) -> list[Document]:
        return create_embedding_documents(
            parent_chunks,
            self.tokenizer,
            self.max_sequence_length,
            overlap_tokens=overlap_tokens,
        )

    def encode_documents(
        self,
        documents: Sequence[Document],
        *,
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        if not documents:
            raise ValueError("Não há documentos para gerar embeddings.")
        texts = [document.page_content for document in documents]
        encoder = getattr(self.model, "encode_document", None) or self.model.encode
        values = encoder(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _normalized_matrix(values)

    def encode_query(self, query: str) -> np.ndarray:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("A consulta não pode ser vazia.")
        token_count = len(encode_without_truncation(self.tokenizer, normalized_query))
        if token_count > self.content_limit:
            raise ValueError(
                f"A consulta possui {token_count} tokens; limite: {self.content_limit}."
            )
        encoder = getattr(self.model, "encode_query", None) or self.model.encode
        values = encoder(
            [normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return _normalized_matrix(values)[0]
