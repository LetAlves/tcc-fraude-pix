import json
import unittest
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class JulyDeliverablesTest(unittest.TestCase):
    def test_corpus_catalog_contains_unique_official_sources(self) -> None:
        config = json.loads(
            (PROJECT_ROOT / "config" / "rag_corpus_sources.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema_version"], "1.1")
        sources = config["sources"]
        document_ids = [source["document_id"] for source in sources]

        self.assertEqual(len(sources), 4)
        self.assertEqual(len(document_ids), len(set(document_ids)))
        self.assertEqual(
            set(document_ids),
            {
                "bcb-regulamento-pix-resolucao-1-2020",
                "bcb-resolucao-103-2021",
                "bcb-guia-med-4-4",
                "febraban-tecnologia-bancaria-2024-volume-1",
            },
        )
        for source in sources:
            self.assertEqual(urlparse(source["source_url"]).scheme, "https")
            self.assertEqual(
                urlparse(source.get("download_url", source["source_url"])).scheme,
                "https",
            )

        guide = next(
            source
            for source in sources
            if source["document_id"] == "bcb-guia-med-4-4"
        )
        self.assertEqual(guide["version_label"], "4.4")
        self.assertEqual(guide["effective_from"], "2026-09-01")
        self.assertEqual(guide["future_effective_from"], "2026-10-26")

    def test_raw_rag_artifacts_are_ignored_and_approved_index_is_versionable(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn("data/rag/raw/", gitignore)
        self.assertIn("data/rag/processed/", gitignore)
        self.assertIn("data/rag/index/*", gitignore)
        self.assertIn("!data/rag/index/index.faiss", gitignore)
        self.assertIn("!data/rag/index/metadata.jsonl", gitignore)
        self.assertIn("!data/rag/index/manifest.json", gitignore)
        self.assertIn("data/rag/manifest.json", gitignore)

    def test_shap_methodology_records_scale_and_anonymous_feature_limit(self) -> None:
        methodology = (
            PROJECT_ROOT / "monografia" / "capitulos" / "03_metodologia.tex"
        ).read_text(encoding="utf-8")

        self.assertIn("TreeExplainer", methodology)
        self.assertIn("feature\\_perturbation=interventional", methodology)
        self.assertIn("model\\_output=probability", methodology)
        self.assertIn("log-odds", methodology)
        self.assertIn("atributo anonimizado de alta influência", methodology)
        self.assertIn("não como causalidade", methodology)

    def test_joint_llm_decision_is_not_marked_complete(self) -> None:
        report = (
            PROJECT_ROOT / "reports" / "pessoa_2" / "julho" / "README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("decisão conjunta pendente", report)
        self.assertIn("m3_p1_1", report)
        self.assertIn("não foram marcadas como concluídas", report)

    def test_current_snapshot_matches_catalog_and_has_hashes(self) -> None:
        snapshot = json.loads(
            (
                PROJECT_ROOT
                / "reports"
                / "pessoa_2"
                / "setembro"
                / "corpus_snapshot_2026-09-10.json"
            ).read_text(encoding="utf-8")
        )
        config = json.loads(
            (PROJECT_ROOT / "config" / "rag_corpus_sources.json").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(len(snapshot["documents"]), 4)
        self.assertEqual(
            {document["document_id"] for document in snapshot["documents"]},
            {source["document_id"] for source in config["sources"]},
        )
        for document in snapshot["documents"]:
            self.assertEqual(len(document["sha256"]), 64)
            int(document["sha256"], 16)


if __name__ == "__main__":
    unittest.main()
