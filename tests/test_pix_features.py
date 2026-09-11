import unittest

import numpy as np
import pandas as pd

from src.features.pix_features import (
    dispositivo_raro_proxy,
    frequencia_recente_proxy,
    posicao_ciclo_diario_relativa,
    valor_atipico_proxy,
)


class PixFeaturesTest(unittest.TestCase):
    def test_frequency_handles_missing_card_and_excludes_equal_timestamp(self) -> None:
        frame = pd.DataFrame(
            {
                "card1": [1.0, 1.0, 1.0, np.nan],
                "TransactionDT": [0, 10, 10, 20],
            }
        )

        result = frequencia_recente_proxy(frame, janela_segundos=100)

        self.assertEqual(result.iloc[0], 0.0)
        self.assertEqual(result.iloc[1], 1.0)
        self.assertEqual(result.iloc[2], 1.0)
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_frequency_uses_open_window_boundary(self) -> None:
        frame = pd.DataFrame(
            {
                "card1": [1, 1, 1],
                "TransactionDT": [0, 10, 20],
            }
        )

        result = frequencia_recente_proxy(frame, janela_segundos=10)

        self.assertListEqual(result.tolist(), [0.0, 0.0, 0.0])

    def test_device_rarity_requires_card_and_excludes_equal_timestamp(self) -> None:
        frame = pd.DataFrame(
            {
                "card1": [1.0, 1.0, 1.0, np.nan],
                "DeviceInfo": ["A", "A", "A", "A"],
                "TransactionDT": [0, 10, 10, 20],
            }
        )

        result = dispositivo_raro_proxy(frame)

        self.assertEqual(result.iloc[0], 1.0)
        self.assertEqual(result.iloc[1], 0.5)
        self.assertEqual(result.iloc[2], 0.5)
        self.assertTrue(np.isnan(result.iloc[3]))

    def test_amount_feature_does_not_use_rows_from_equal_timestamp(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionAmt": [10.0, 20.0, 100.0, 30.0],
                "card1": [1, 1, 1, 1],
                "TransactionDT": [0, 10, 10, 20],
            }
        )

        result = valor_atipico_proxy(frame)

        self.assertTrue(np.isnan(result.iloc[1]))
        self.assertTrue(np.isnan(result.iloc[2]))
        self.assertFalse(np.isnan(result.iloc[3]))

    def test_cycle_encoding_has_expected_quadrants(self) -> None:
        frame = pd.DataFrame({"TransactionDT": [0, 21_600, 43_200]})

        result = posicao_ciclo_diario_relativa(frame)

        np.testing.assert_allclose(
            result["posicao_ciclo_diario_relativa"],
            [0.0, 0.25, 0.5],
        )
        np.testing.assert_allclose(
            result["posicao_ciclo_diario_sen"],
            [0.0, 1.0, 0.0],
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
