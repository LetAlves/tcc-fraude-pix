"""Avalia o artefato entregue: escolhe F1 na validação e só depois abre o teste.

Não treina, não reajusta o pré-processador e não sobrescreve evidências anteriores.
Execute como módulo, informando --data-dir com os dois CSVs IEEE-CIS originais.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    model_dir = ROOT / "models" / "xgboost"
    output = ROOT / "reports" / "avaliacao_xgboost_entregue.json"
    policy_path = model_dir / "politica_decisao.json"
    if output.exists() or policy_path.exists():
        raise FileExistsError("Avaliação ou política já registrada; não será sobrescrita.")
    os.environ["TCC_DATA_RAW"] = str(args.data_dir.resolve())

    from src.data_loader import carregar_dados
    from src.features.pix_features import criar_features_pix
    from src.features.preprocessor import COLUNA_ALVO, dividir_temporal, reduzir_precisao
    from src.models.evaluator import avaliar_probabilidades, escolher_limiar, matriz_confusao
    from src.models.persistencia import carregar, prever

    # Vincula o artefato ao relatório SHAP já publicado antes de desserializar.
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    shap = json.loads((ROOT / "reports/pessoa_1/julho/shap/shap_xgboost.json").read_text(encoding="utf-8"))
    expected = {
        "modelo.json": shap["modelo"]["hash_modelo"],
        "preprocessador.joblib": shap["modelo"]["hash_preprocessador"],
    }
    if manifest["hashes"] != expected:
        raise ValueError("O manifesto não identifica o modelo usado no SHAP publicado.")
    for filename, checksum in expected.items():
        if sha256(model_dir / filename) != checksum:
            raise ValueError(f"Artefato divergente: {filename}")
    preprocessor, model, _ = carregar(model_dir)
    dataset_hashes = {
        name: sha256(args.data_dir / name)
        for name in ("train_transaction.csv", "train_identity.csv")
    }
    print("Modelo verificado; preparando features causais e split temporal.", flush=True)
    data = carregar_dados(baixar_se_ausente=False, nrows=None)
    frame = reduzir_precisao(criar_features_pix(data[2]))
    del data
    gc.collect()
    train, validation, test = dividir_temporal(frame)
    del frame
    counts = {"treino": len(train), "validacao": len(validation), "teste": len(test)}
    if counts != {"treino": 413378, "validacao": 88581, "teste": 88581}:
        raise ValueError(f"Split difere do protocolo publicado: {counts}")
    del train
    gc.collect()

    p_validation = prever(preprocessor, model, validation.drop(columns=[COLUNA_ALVO]))
    y_validation = validation[COLUNA_ALVO].to_numpy(dtype=int)
    selection = escolher_limiar(y_validation, p_validation, criterio="f1")
    # Política congelada e persistida ANTES de prever ou medir o teste.
    policy = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "hashes_modelo": expected,
        "hashes_dataset": dataset_hashes,
        "criterio": "f1",
        "particao_selecao": "validacao",
        "desempate": "menor limiar entre os máximos de F1, conforme escolher_limiar",
        **selection,
    }
    save_new(policy_path, policy)
    print(f"Limiar F1 congelado: {selection['limiar']:.10f}; avaliando teste.", flush=True)
    del validation, p_validation, y_validation
    gc.collect()
    p_test = prever(preprocessor, model, test.drop(columns=[COLUNA_ALVO]))
    y_test = test[COLUNA_ALVO].to_numpy(dtype=int)
    threshold = selection["limiar"]
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "modelo_retreinado": False,
        "preprocessador_reajustado": False,
        "hashes_modelo": expected,
        "hashes_dataset": dataset_hashes,
        "split": {"protocolo": "temporal 70/15/15 sem dividir empates", **counts},
        "versoes": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "scikit-learn", "xgboost")},
        },
        "selecao_limiar_validacao": selection,
        "teste": avaliar_probabilidades(y_test, p_test, limiar=threshold),
        "matriz_confusao_teste": {
            "ordem": "linhas=reais [0,1]; colunas=previstas [0,1]",
            "valores": matriz_confusao(y_test, p_test, limiar=threshold).values.tolist(),
        },
    }
    if any(sha256(model_dir / name) != digest for name, digest in expected.items()):
        raise ValueError("Artefatos foram alterados durante a avaliação.")
    save_new(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
