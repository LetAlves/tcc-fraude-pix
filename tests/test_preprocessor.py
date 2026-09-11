import unittest

import pandas as pd

from src.features.preprocessor import dividir_temporal


class DivisaoTemporalTest(unittest.TestCase):
    @staticmethod
    def _dataset(tempos: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TransactionID": range(1, len(tempos) + 1),
                "TransactionDT": tempos,
                "isFraud": [0] * len(tempos),
            }
        )

    def assert_sem_timestamp_compartilhado(self, *particoes: pd.DataFrame) -> None:
        conjuntos = [set(df["TransactionDT"]) for df in particoes]
        for indice, atual in enumerate(conjuntos):
            for seguinte in conjuntos[indice + 1 :]:
                self.assertTrue(atual.isdisjoint(seguinte))

    def test_preserva_proporcoes_quando_fronteiras_nao_tem_empate(self) -> None:
        df = self._dataset(list(range(10)))

        treino, validacao, teste = dividir_temporal(df)

        self.assertEqual((len(treino), len(validacao), len(teste)), (7, 1, 2))
        self.assert_sem_timestamp_compartilhado(treino, validacao, teste)

    def test_avanca_fronteira_do_treino_ate_fim_do_timestamp(self) -> None:
        df = self._dataset([1, 2, 3, 3, 3, 4, 5, 6, 7, 8])

        treino, validacao, teste = dividir_temporal(
            df, frac_treino=0.3, frac_val=0.3
        )

        self.assertEqual((len(treino), len(validacao), len(teste)), (5, 1, 4))
        self.assert_sem_timestamp_compartilhado(treino, validacao, teste)

    def test_avanca_fronteira_da_validacao_ate_fim_do_timestamp(self) -> None:
        df = self._dataset([1, 2, 3, 4, 5, 5, 5, 6, 7, 8])

        treino, validacao, teste = dividir_temporal(
            df, frac_treino=0.2, frac_val=0.3
        )

        self.assertEqual((len(treino), len(validacao), len(teste)), (2, 5, 3))
        self.assert_sem_timestamp_compartilhado(treino, validacao, teste)

    def test_preserva_todas_as_linhas_apos_deslocar_os_cortes(self) -> None:
        df = self._dataset([3, 1, 2, 3, 4, 5, 5, 6, 7, 8])

        particoes = dividir_temporal(df, frac_treino=0.3, frac_val=0.3)
        ids_resultantes = pd.concat(particoes)["TransactionID"].tolist()

        self.assertCountEqual(ids_resultantes, df["TransactionID"].tolist())
        self.assertEqual(len(ids_resultantes), len(set(ids_resultantes)))
        self.assert_sem_timestamp_compartilhado(*particoes)

    def test_reserva_grupo_de_validacao_quando_cortes_caem_no_mesmo_empate(self) -> None:
        df = self._dataset([1, 2, 2, 2, 2, 2, 2, 3, 4, 5])

        treino, validacao, teste = dividir_temporal(
            df, frac_treino=0.3, frac_val=0.3
        )

        self.assertEqual((len(treino), len(validacao), len(teste)), (7, 1, 2))
        self.assert_sem_timestamp_compartilhado(treino, validacao, teste)

    def test_rejeita_quando_empate_impede_tres_particoes(self) -> None:
        df = self._dataset([1] * 10)

        with self.assertRaisesRegex(ValueError, "sem dividir grupos"):
            dividir_temporal(df)

    def test_rejeita_tempo_ausente(self) -> None:
        df = self._dataset([1, 2, 3])
        df.loc[1, "TransactionDT"] = None

        with self.assertRaisesRegex(ValueError, "valores ausentes"):
            dividir_temporal(df, frac_treino=0.34, frac_val=0.33)


if __name__ == "__main__":
    unittest.main()
