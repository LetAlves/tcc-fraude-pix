"""Consulta o índice RAG local e exibe fontes rastreáveis, sem chamar um LLM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, MultilingualEmbedder
from src.rag.vector_store import DEFAULT_INDEX_DIR, FaissVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Pergunta ou conceito a pesquisar no corpus.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k <= 0 or args.k > 20:
        parser.error("--k deve estar entre 1 e 20.")

    store = FaissVectorStore.load(args.index_dir)
    model_name = store.model_name or DEFAULT_EMBEDDING_MODEL
    embedder = MultilingualEmbedder(
        model_name, local_files_only=True, revision=store.model_revision,
    )
    results = store.search(embedder.encode_query(args.query), k=args.k)
    for rank, result in enumerate(results, start=1):
        metadata = result.document.metadata
        location = (
            f"página {metadata['page']}"
            if metadata.get("page") is not None
            else "documento HTML/JSON"
        )
        excerpt = " ".join(result.document.page_content.split())[:320]
        print(f"{rank}. score={result.score:.4f} — {metadata.get('title')} — {location}")
        print(f"   fonte: {metadata.get('source_url')}")
        print(f"   trecho: {excerpt}")


if __name__ == "__main__":
    main()
