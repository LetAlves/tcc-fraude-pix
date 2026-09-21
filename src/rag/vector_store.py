"""Índice FAISS exato, persistente e verificável para o corpus do TCC."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document


DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[2] / "data" / "rag" / "index"
INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as error:  # pragma: no cover - mensagem de ambiente
        raise RuntimeError("Instale faiss-cpu para usar o banco vetorial.") from error
    return faiss


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_matrix(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("Matriz vetorial inválida.")
    if not np.isfinite(matrix).all():
        raise ValueError("A matriz contém NaN ou infinito.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("A matriz contém vetor de norma zero.")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float


class FaissVectorStore:
    """Busca cosseno por ``IndexFlatIP`` com metadados alinhados por posição."""

    def __init__(
        self,
        index: Any,
        documents: list[Document],
        *,
        model_name: str,
        model_revision: str | None = None,
    ) -> None:
        if int(index.ntotal) != len(documents):
            raise ValueError("Quantidade de vetores difere da quantidade de documentos.")
        self.index = index
        self.documents = documents
        self.model_name = model_name
        self.model_revision = model_revision

    @classmethod
    def from_embeddings(
        cls,
        embeddings: Any,
        documents: list[Document],
        *,
        model_name: str,
        model_revision: str | None = None,
    ) -> "FaissVectorStore":
        matrix = _normalize_matrix(embeddings)
        if matrix.shape[0] != len(documents):
            raise ValueError("Cada documento deve possuir exatamente um embedding.")
        faiss = _import_faiss()
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        return cls(
            index,
            list(documents),
            model_name=model_name,
            model_revision=model_revision,
        )

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    def search(self, query_embedding: Any, *, k: int = 5) -> list[SearchResult]:
        if k <= 0:
            raise ValueError("k deve ser positivo.")
        query = _normalize_matrix(query_embedding)
        if query.shape != (1, self.dimension):
            raise ValueError(
                f"Dimensão da consulta {query.shape[1]} difere do índice {self.dimension}."
            )
        count = min(k, len(self.documents))
        scores, positions = self.index.search(query, count)
        return [
            SearchResult(self.documents[int(position)], float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]

    def save(self, directory: Path = DEFAULT_INDEX_DIR) -> dict[str, Any]:
        """Persiste índice e metadados com hashes; nenhum pickle é usado."""

        faiss = _import_faiss()
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        manifest_path = directory / MANIFEST_FILENAME

        temporary_index = directory / f"{INDEX_FILENAME}.part"
        temporary_metadata = directory / f"{METADATA_FILENAME}.part"
        faiss.write_index(self.index, str(temporary_index))
        with temporary_metadata.open("w", encoding="utf-8", newline="\n") as file:
            for document in self.documents:
                json.dump(
                    {"page_content": document.page_content, "metadata": document.metadata},
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")
        temporary_index.replace(index_path)
        temporary_metadata.replace(metadata_path)

        manifest = {
            "schema_version": "1.0",
            "created_at": datetime.now(UTC).isoformat(),
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "metric": "cosine_via_normalized_inner_product",
            "index_type": "IndexFlatIP",
            "dimension": self.dimension,
            "document_count": len(self.documents),
            "files": {
                INDEX_FILENAME: {"sha256": _sha256(index_path)},
                METADATA_FILENAME: {"sha256": _sha256(metadata_path)},
            },
        }
        temporary_manifest = directory / f"{MANIFEST_FILENAME}.part"
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_manifest.replace(manifest_path)
        return manifest

    @classmethod
    def load(cls, directory: Path = DEFAULT_INDEX_DIR) -> "FaissVectorStore":
        """Valida os hashes antes de carregar o índice binário FAISS."""

        faiss = _import_faiss()
        index_path = directory / INDEX_FILENAME
        metadata_path = directory / METADATA_FILENAME
        manifest_path = directory / MANIFEST_FILENAME
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, dict) or manifest.get("index_type") != "IndexFlatIP":
            raise ValueError("Manifesto do índice inválido ou tipo não autorizado.")

        files = manifest.get("files", {})
        for path in (index_path, metadata_path):
            expected = files.get(path.name, {}).get("sha256")
            if not isinstance(expected, str) or _sha256(path) != expected:
                raise ValueError(f"Falha de integridade em {path.name}.")

        documents: list[Document] = []
        with metadata_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Metadado inválido na linha {line_number}.")
                text = value.get("page_content")
                metadata = value.get("metadata")
                if not isinstance(text, str) or not isinstance(metadata, dict):
                    raise ValueError(f"Metadado inválido na linha {line_number}.")
                documents.append(Document(page_content=text, metadata=metadata))

        index = faiss.read_index(str(index_path))
        if (
            int(index.d) != manifest.get("dimension")
            or int(index.ntotal) != manifest.get("document_count")
            or len(documents) != manifest.get("document_count")
        ):
            raise ValueError("Índice e manifesto estão inconsistentes.")
        return cls(
            index,
            documents,
            model_name=str(manifest.get("model_name", "")),
            model_revision=manifest.get("model_revision"),
        )
