import tempfile
import unittest
from pathlib import Path

import numpy as np
from langchain_core.documents import Document

from src.rag.vector_store import FaissVectorStore


class VectorStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(page_content="MED e devolução", metadata={"document_id": "med"}),
            Document(page_content="Pesquisa setorial", metadata={"document_id": "setor"}),
        ]
        self.vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def test_cosine_search_returns_most_similar_document(self) -> None:
        store = FaissVectorStore.from_embeddings(
            self.vectors,
            self.documents,
            model_name="modelo-teste",
            model_revision="revisao-imutavel",
        )

        results = store.search([0.9, 0.1], k=2)

        self.assertEqual(results[0].document.metadata["document_id"], "med")
        self.assertGreater(results[0].score, results[1].score)

    def test_persistence_round_trip_validates_counts_and_hashes(self) -> None:
        store = FaissVectorStore.from_embeddings(
            self.vectors,
            self.documents,
            model_name="modelo-teste",
            model_revision="revisao-imutavel",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            manifest = store.save(directory)
            restored = FaissVectorStore.load(directory)

            self.assertEqual(manifest["index_type"], "IndexFlatIP")
            self.assertEqual(manifest["model_revision"], "revisao-imutavel")
            self.assertEqual(restored.model_revision, "revisao-imutavel")
            self.assertEqual(restored.dimension, 2)
            self.assertEqual(len(restored.documents), 2)

            metadata_path = directory / "metadata.jsonl"
            metadata_path.write_text("alterado", encoding="utf-8")
            with self.assertRaises(ValueError):
                FaissVectorStore.load(directory)


if __name__ == "__main__":
    unittest.main()
