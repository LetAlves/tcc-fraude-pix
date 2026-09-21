"""Reconstrói RAG offline do corpus coletado e registra uma nova proveniência.

Pré-requisito: executar download_rag_corpus.py e ter MiniLM no cache local.
Não atribui à nova construção a identidade do índice histórico de setembro.
"""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.build_rag_index import build_index
from src.rag.embeddings import DEFAULT_EMBEDDING_MODEL, MultilingualEmbedder
from src.rag.vector_store import FaissVectorStore

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "data/rag/index"
REPORT = ROOT / "reports/pessoa_2/setembro/reconstrucao_rag.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if REPORT.exists() or (INDEX.exists() and any(INDEX.iterdir())):
        raise FileExistsError("Índice ou relatório já existente; não será sobrescrito.")
    corpus_path = ROOT / "data/rag/manifest.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    historical = json.loads((ROOT / "reports/pessoa_2/setembro/corpus_snapshot_2026-09-10.json").read_text(encoding="utf-8"))
    old_hashes = {d["document_id"]: d["sha256"] for d in historical["documents"]}
    # Usa a revisão exata no cache tanto na construção quanto no teste semântico.
    snapshot = Path(snapshot_download(DEFAULT_EMBEDDING_MODEL, local_files_only=True))
    result = build_index(
        model_name=DEFAULT_EMBEDDING_MODEL,
        model_revision=snapshot.name,
        local_files_only=True,
    )
    manifest = result["index_manifest"]
    manifest["artifact_version"] = "reconstrucao-" + manifest["created_at"].split("T")[0]
    manifest["provenance"] = {
        "kind": "nova construcao; indice historico nao localizado",
        "historical_reference": "reports/pessoa_2/setembro/corpus_snapshot_2026-09-10.json",
        "historical_index_identical": None,
        "catalog_sha256": sha256(ROOT / "config/rag_corpus_sources.json"),
        "sources": corpus["documents"],
        "source_hash_comparison": {
            doc["document_id"]: doc["sha256"] == old_hashes.get(doc["document_id"])
            for doc in corpus["documents"]
        },
        "parameters": {
            "parent_chunk_tokens": 500, "parent_overlap_tokens": 50,
            "embedding_overlap_tokens": 24, "batch_size": 32,
        },
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in (
                "sentence-transformers", "transformers", "torch", "faiss-cpu", "pypdf", "numpy"
            )
        }},
    }
    (INDEX / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    store = FaissVectorStore.load(INDEX)
    embedder = MultilingualEmbedder(
        DEFAULT_EMBEDDING_MODEL,
        local_files_only=True,
        revision=snapshot.name,
    )
    smoke_queries = [
        "Como funciona o Mecanismo Especial de Devolução em caso de suspeita de fraude?",
        "Qual é o prazo para contestar uma transação de devolução do MED?",
    ]
    smoke = []
    for query in smoke_queries:
        results = store.search(embedder.encode_query(query), k=5)
        smoke.append({"query": query, "results": [{
            "rank": i, "score": r.score,
            "document_id": r.document.metadata["document_id"],
            "embedding_chunk_id": r.document.metadata["embedding_chunk_id"],
            "page": r.document.metadata.get("page"),
            "excerpt": r.document.page_content,
        } for i, r in enumerate(results, 1)]})
    result["embedding_model"] = DEFAULT_EMBEDDING_MODEL
    result["smoke_tests"] = smoke
    result["experimental_quality_metrics"] = "{{RESULTADO}}; testes de fumaca nao substituem avaliacao anotada"
    result["artifact_bytes"] = {p.name: p.stat().st_size for p in INDEX.iterdir() if p.is_file()}
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"index_manifest", "smoke_tests"}}, indent=2))


if __name__ == "__main__":
    main()
