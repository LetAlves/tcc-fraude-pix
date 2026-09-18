from __future__ import annotations

import json

import numpy as np
import pytest
from xgboost import XGBClassifier

from src.models.shap_xgboost import (
    amostrar_indices,
    erro_maximo_aditividade,
    gerar_relatorio_shap,
    importancia_global,
    selecionar_casos_representativos,
)


class PreprocessadorFalso:
    def get_feature_names_out(self):
        return np.asarray(["numericas__a", "numericas__b", "numericas__c"])


def test_amostragem_e_deterministica_e_sem_repeticao() -> None:
    primeira = amostrar_indices(100, 20, semente=7)
    segunda = amostrar_indices(100, 20, semente=7)

    np.testing.assert_array_equal(primeira, segunda)
    assert len(primeira) == len(np.unique(primeira)) == 20
    assert np.all(np.diff(primeira) > 0)
    assert len(amostrar_indices(4, 20, semente=7)) == 4


def test_seleciona_mediana_de_cada_quadrante() -> None:
    y = np.asarray([1, 1, 1, 0, 0, 0, 1, 0])
    proba = np.asarray([0.60, 0.80, 0.95, 0.55, 0.70, 0.90, 0.20, 0.10])

    casos = selecionar_casos_representativos(y, proba, limiar=0.5)

    assert casos == {
        "verdadeiro_positivo": 1,
        "falso_positivo": 4,
        "falso_negativo": 6,
        "verdadeiro_negativo": 7,
    }


def test_importancia_e_aditividade() -> None:
    valores = np.asarray([[0.1, -0.4], [-0.3, 0.2]])
    ranking = importancia_global(valores, ["fraca", "forte"])
    probabilidades = np.asarray([0.4, 0.5])
    bases = probabilidades - valores.sum(axis=1)

    assert [item["feature"] for item in ranking] == ["forte", "fraca"]
    assert erro_maximo_aditividade(valores, bases, probabilidades) < 1e-12


@pytest.mark.filterwarnings("ignore:.*distutils Version classes.*")
def test_relatorio_real_fica_na_escala_de_probabilidade(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    gerador = np.random.default_rng(42)
    X = gerador.normal(size=(180, 3)).astype(np.float32)
    y = (X[:, 0] + 0.7 * X[:, 1] - 0.2 * X[:, 2] > 0).astype(int)
    modelo = XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.2,
        random_state=42,
        enable_categorical=False,
    ).fit(X[:130], y[:130])

    relatorio = gerar_relatorio_shap(
        modelo,
        PreprocessadorFalso(),
        X[:130],
        X[130:],
        y[130:],
        tmp_path / "saida",
        tamanho_fundo=30,
        tamanho_amostra_global=40,
    )

    assert relatorio["metodologia"]["model_output"] == "probability"
    assert relatorio["fidelidade"]["aprovada"] is True
    assert relatorio["fidelidade"]["erro_maximo"] < 1e-5
    assert (tmp_path / "saida" / "shap_summary_xgboost.png").is_file()
    salvo = json.loads(
        (tmp_path / "saida" / "shap_xgboost.json").read_text(encoding="utf-8")
    )
    assert salvo["importancia_global"][0]["feature"] in {"a", "b", "c"}
