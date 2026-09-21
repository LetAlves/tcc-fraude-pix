"""Constrói os chunks, embeddings e o índice FAISS do corpus RAG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, MultilingualEmbedder
from src.rag.knowledge_base import (
    DEFAULT_CHUNKS,
    DEFAULT_MANIFEST,
    DEFAULT_PARENT_CHUNK_TOKENS,
    DEFAULT_PARENT_OVERLAP_TOKENS,
    DEFAULT_RAW_DIR,
    chunk_documents,
    load_corpus_pages,
    write_documents_jsonl,
)
from src.rag.vector_store import DEFAULT_INDEX_DIR, FaissVectorStore


def build_index(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    raw_dir: Path = DEFAULT_RAW_DIR,
    parent_chunks_path: Path = DEFAULT_CHUNKS,
    index_dir: Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    model_revision: str | None = None,
    batch_size: int = 32,
    local_files_only: bool = False,
) -> dict[str, object]:
    """Executa o pipeline e devolve um resumo verificável da construção."""

    embedder = MultilingualEmbedder(
        model_name,
        local_files_only=local_files_only,
        revision=model_revision,
    )
    pages = load_corpus_pages(manifest_path, raw_dir)
    parents = chunk_documents(
        pages,
        embedder.tokenizer,
        chunk_tokens=DEFAULT_PARENT_CHUNK_TOKENS,
        overlap_tokens=DEFAULT_PARENT_OVERLAP_TOKENS,
    )
    if not parents:
        raise ValueError("A extração não produziu chunks pais.")
    write_documents_jsonl(parents, parent_chunks_path)

    embedding_documents = embedder.create_documents(parents)
    vectors = embedder.encode_documents(
        embedding_documents,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    store = FaissVectorStore.from_embeddings(
        vectors,
        embedding_documents,
        model_name=model_name,
        model_revision=model_revision,
    )
    index_manifest = store.save(index_dir)
    return {
        "source_units": len(pages),
        "parent_chunks": len(parents),
        "embedding_windows": len(embedding_documents),
        "embedding_model": model_name,
        "model_max_sequence_length": embedder.max_sequence_length,
        "embedding_content_limit": embedder.content_limit,
        "vector_dimension": store.dimension,
        "index_manifest": index_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--parent-chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--model-revision",
        help="Revisão imutável do modelo Hugging Face usada no índice.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Usa somente um modelo já presente no cache local.",
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size deve ser positivo.")

    result = build_index(
        manifest_path=args.manifest,
        raw_dir=args.raw_dir,
        parent_chunks_path=args.parent_chunks,
        index_dir=args.index_dir,
        model_name=args.model,
        model_revision=args.model_revision,
        batch_size=args.batch_size,
        local_files_only=args.offline,
    )
    print("Índice RAG construído com sucesso:")
    for key, value in result.items():
        if key != "index_manifest":
            print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
