import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document

from src.rag.knowledge_base import (
    chunk_documents,
    load_corpus_pages,
    read_documents_jsonl,
    write_documents_jsonl,
)


class WordTokenizer:
    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        result = []
        for token in text.split():
            if token not in self.token_to_id:
                identifier = len(self.token_to_id) + 1
                self.token_to_id[token] = identifier
                self.id_to_token[identifier] = token
            result.append(self.token_to_id[token])
        return result

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        return " ".join(self.id_to_token[token] for token in token_ids)


class KnowledgeBaseTest(unittest.TestCase):
    def test_parent_chunking_uses_500_tokens_and_50_overlap(self) -> None:
        tokenizer = WordTokenizer()
        document = Document(
            page_content=" ".join(f"t{index}" for index in range(700)),
            metadata={"document_id": "doc-1", "page": 3},
        )

        chunks = chunk_documents([document], tokenizer)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["token_count"], 500)
        self.assertEqual(chunks[1].metadata["token_start"], 450)
        self.assertEqual(chunks[0].page_content.split()[-50:], chunks[1].page_content.split()[:50])
        self.assertNotEqual(chunks[0].metadata["chunk_id"], chunks[1].metadata["chunk_id"])

    def test_html_ingestion_checks_hash_and_removes_obsolete_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            html_path = raw_dir / "norma.html"
            html_path.write_text(
                "<html><body><main><h1>Regra vigente</h1>"
                "<del>regra revogada</del><p>Texto aplicável.</p></main></body></html>",
                encoding="utf-8",
            )
            digest = hashlib.sha256(html_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "norma-1",
                                "title": "Norma",
                                "format": "html",
                                "local_filename": "norma.html",
                                "sha256": digest,
                                "effective_from": "2026-09-01",
                                "future_effective_from": "2026-10-26",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_corpus_pages(manifest_path, raw_dir)

            self.assertEqual(len(pages), 1)
            self.assertIn("Texto aplicável", pages[0].page_content)
            self.assertNotIn("regra revogada", pages[0].page_content)
            self.assertEqual(pages[0].metadata["effective_from"], "2026-09-01")
            self.assertEqual(
                pages[0].metadata["future_effective_from"], "2026-10-26"
            )

    def test_jsonl_round_trip_uses_plain_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "chunks.jsonl"
            original = [
                Document(page_content="conteúdo", metadata={"chunk_id": "abc", "page": 1})
            ]

            count = write_documents_jsonl(original, path)
            restored = read_documents_jsonl(path)

            self.assertEqual(count, 1)
            self.assertEqual(restored[0].page_content, "conteúdo")
            self.assertEqual(restored[0].metadata["chunk_id"], "abc")

    def test_bcb_json_ingestion_selects_normative_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            json_path = raw_dir / "norma.json"
            json_path.write_text(
                json.dumps(
                    {
                        "conteudo": [
                            {
                                "Id": 10,
                                "Titulo": "Resolução de teste",
                                "Assunto": "Assunto rastreável",
                                "Texto": (
                                    "<div><span style='text-decoration: line-through'>revogado</span>"
                                    "<p>Art. 1º Texto vigente.</p></div>"
                                ),
                                "DOU": "Publicada no DOU.",
                                "VersaoNormativo": 2,
                                "Revogado": False,
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "documents": [
                            {
                                "document_id": "norma-bcb",
                                "format": "bcb_normativo_json",
                                "local_filename": "norma.json",
                                "sha256": digest,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            pages = load_corpus_pages(manifest_path, raw_dir)

            self.assertIn("Texto vigente", pages[0].page_content)
            self.assertNotIn("revogado", pages[0].page_content)
            self.assertEqual(pages[0].metadata["bcb_record_id"], 10)


if __name__ == "__main__":
    unittest.main()
