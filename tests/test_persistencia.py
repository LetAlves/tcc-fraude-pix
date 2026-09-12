"""Testes da persistência de modelos.

O teste central é o de ida e volta: salvar, carregar e obter **exatamente** as
mesmas probabilidades. Um modelo persistido que prevê quase igual não serve —
os números da monografia precisam ser reproduzíveis a partir do artefato.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.features.preprocessor import construir_preprocessador
from src.models.persistencia import ARQUIVO_MANIFESTO, carregar, prever, salvar


def _dados(n=200, semente=0):
    rng = np.random.default_rng(semente)
    X = pd.DataFrame({
        "valor": rng.normal(100, 30, n),
        "canal": rng.choice(["A", "B", "C"], n),
        "dispositivo": rng.choice([f"d{i}" for i in range(20)], n),
    })
    y = (X["valor"] > 120).astype(int)
    return X, y


def _treinar():
    X, y = _dados()
    pre = construir_preprocessador(["valor"], ["canal"], ["dispositivo"])
    M = pre.fit_transform(X)
    modelo = RandomForestClassifier(n_estimators=5, random_state=0).fit(M, y)
    return pre, modelo, X


class PersistenciaTest(unittest.TestCase):

    def test_ida_e_volta_preserva_as_probabilidades(self) -> None:
        pre, modelo, X = _treinar()
        esperado = prever(pre, modelo, X)

        with tempfile.TemporaryDirectory() as pasta:
            salvar(pasta, pre, modelo)
            pre_lido, modelo_lido, _ = carregar(pasta)
            obtido = prever(pre_lido, modelo_lido, X)

        np.testing.assert_array_equal(esperado, obtido)

    def test_manifesto_e_legivel_sem_carregar_o_modelo(self) -> None:
        pre, modelo, _ = _treinar()

        with tempfile.TemporaryDirectory() as pasta:
            salvar(pasta, pre, modelo, metadados={"linhas_treino": 200})
            manifesto = json.loads(
                (Path(pasta) / ARQUIVO_MANIFESTO).read_text(encoding="utf-8"))

        self.assertEqual(manifesto["tipo_do_modelo"], "RandomForestClassifier")
        self.assertEqual(manifesto["metadados"]["linhas_treino"], 200)
        self.assertIn("scikit-learn", manifesto["versoes"])
        self.assertIn("gerado_em", manifesto)

    def test_arquivo_alterado_e_detectado(self) -> None:
        """Corrupção deve virar erro no carregamento, não previsão errada."""
        pre, modelo, _ = _treinar()

        with tempfile.TemporaryDirectory() as pasta:
            manifesto = salvar(pasta, pre, modelo)
            alvo = Path(pasta) / manifesto["arquivo_do_modelo"]
            alvo.write_bytes(alvo.read_bytes() + b"lixo")

            with self.assertRaisesRegex(ValueError, "hash de"):
                carregar(pasta)

    def test_verificacao_de_hash_pode_ser_dispensada(self) -> None:
        pre, modelo, _ = _treinar()
        with tempfile.TemporaryDirectory() as pasta:
            salvar(pasta, pre, modelo)
            pre_lido, modelo_lido, _ = carregar(pasta, verificar_hashes=False)
        self.assertIsNotNone(modelo_lido)

    def test_diretorio_e_criado_se_nao_existir(self) -> None:
        pre, modelo, _ = _treinar()
        with tempfile.TemporaryDirectory() as pasta:
            destino = Path(pasta) / "nivel1" / "nivel2"
            salvar(destino, pre, modelo)
            self.assertTrue((destino / ARQUIVO_MANIFESTO).exists())

    def test_prever_aceita_dados_crus_e_aplica_o_preprocessador(self) -> None:
        """O caminho curto tem de ser o correto: DataFrame cru entra, probabilidade sai."""
        pre, modelo, X = _treinar()
        probabilidades = prever(pre, modelo, X)

        self.assertEqual(len(probabilidades), len(X))
        self.assertTrue(((probabilidades >= 0) & (probabilidades <= 1)).all())


class PersistenciaXGBoostTest(unittest.TestCase):

    def test_xgboost_usa_formato_nativo_sem_pickle(self) -> None:
        from xgboost import XGBClassifier

        X, y = _dados()
        pre = construir_preprocessador(["valor"], ["canal"], ["dispositivo"])
        M = pre.fit_transform(X)
        modelo = XGBClassifier(n_estimators=5, tree_method="hist").fit(M, y)
        esperado = prever(pre, modelo, X)

        with tempfile.TemporaryDirectory() as pasta:
            manifesto = salvar(pasta, pre, modelo)
            self.assertEqual(manifesto["formato_do_modelo"], "xgboost_nativo")
            self.assertTrue((Path(pasta) / "modelo.json").exists())

            pre_lido, modelo_lido, _ = carregar(pasta)
            obtido = prever(pre_lido, modelo_lido, X)

        np.testing.assert_allclose(esperado, obtido, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
