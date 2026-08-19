import json
import re
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.gerar_entrega_junho import calcular_estatisticas, render_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class JuneDeliverablesTest(unittest.TestCase):
    def test_feature_registry_has_unique_traceable_definitions(self) -> None:
        registry_path = PROJECT_ROOT / "config" / "pix_feature_registry.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        features = registry["features"]
        feature_ids = [feature["feature_id"] for feature in features]

        self.assertEqual(
            registry["decision_status"],
            "approved_and_implemented",
        )
        self.assertEqual(
            set(feature_ids),
            {
                "valor_atipico_cartao_proxy",
                "frequencia_recente_cartao_proxy",
                "dispositivo_raro_cartao_proxy",
                "posicao_ciclo_diario_relativa",
            },
        )
        self.assertEqual(len(feature_ids), len(set(feature_ids)))
        self.assertEqual(len(features), 4)
        for feature in features:
            self.assertTrue(feature["source_columns"])
            self.assertTrue(feature["definition"])
            self.assertTrue(feature["leakage_control"])
            self.assertTrue(feature["warning"])

    def test_report_statistics_use_real_join_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            pd.DataFrame(
                {
                    "TransactionID": [1, 2, 3, 4],
                    "isFraud": [0, 0, 0, 1],
                    "TransactionDT": [0, 100, 200, 86_400],
                    "TransactionAmt": [10.0, 20.0, 30.0, 100.0],
                    "card1": [10, 10, 11, 12],
                    "ProductCD": ["W", "W", "C", "C"],
                }
            ).to_csv(data_dir / "train_transaction.csv", index=False)
            pd.DataFrame(
                {
                    "TransactionID": [1, 4],
                    "DeviceType": ["desktop", "mobile"],
                    "DeviceInfo": ["Windows", "iOS Device"],
                }
            ).to_csv(data_dir / "train_identity.csv", index=False)

            stats = calcular_estatisticas(data_dir)

        self.assertEqual(stats["transaction_rows"], 4)
        self.assertEqual(stats["frauds"], 1)
        self.assertEqual(stats["fraud_rate"], 0.25)
        self.assertEqual(stats["identity_coverage"], 0.5)

    def test_report_keeps_pix_limitation_visible(self) -> None:
        registry = json.loads(
            (PROJECT_ROOT / "config" / "pix_feature_registry.json").read_text(
                encoding="utf-8"
            )
        )
        stats = {
            "transaction_rows": 4,
            "transaction_columns": 6,
            "identity_rows": 2,
            "identity_columns": 3,
            "legitimate": 3,
            "frauds": 1,
            "fraud_rate": 0.25,
            "imbalance_ratio": 3.0,
            "identity_coverage": 0.5,
            "transaction_span_days": 1.0,
            "transaction_amount_by_class": {
                0: {"median": 20.0},
                1: {"median": 100.0},
            },
            "card1_unique": 3,
            "device_info_unique": 2,
        }

        report = render_report(stats, registry)

        self.assertIn("não contém transações Pix", report)
        self.assertIn("features aprovadas pela dupla", report)
        self.assertIn("src/features/pix_features.py", report)
        self.assertIn("SMOTE", report)

    def test_monograph_chapters_use_existing_citations(self) -> None:
        bibliography = (PROJECT_ROOT / "monografia" / "referencias.bib").read_text(
            encoding="utf-8"
        )
        bibliography_keys = set(
            re.findall(r"@[a-zA-Z]+\{([^,]+),", bibliography)
        )
        chapter_paths = [
            PROJECT_ROOT / "monografia" / "capitulos" / "02_revisao_bibliografica.tex",
            PROJECT_ROOT / "monografia" / "capitulos" / "03_metodologia.tex",
        ]

        cited_keys: set[str] = set()
        for chapter_path in chapter_paths:
            chapter = chapter_path.read_text(encoding="utf-8")
            self.assertNotIn("`", chapter)
            for citation_group in re.findall(r"\\cite\{([^}]+)\}", chapter):
                cited_keys.update(key.strip() for key in citation_group.split(","))

        self.assertTrue(cited_keys)
        self.assertEqual(cited_keys - bibliography_keys, set())

        main_tex = (PROJECT_ROOT / "monografia" / "main.tex").read_text(
            encoding="utf-8"
        )
        self.assertIn(r"\include{capitulos/02_revisao_bibliografica}", main_tex)
        self.assertIn(r"\include{capitulos/03_metodologia}", main_tex)


if __name__ == "__main__":
    unittest.main()
