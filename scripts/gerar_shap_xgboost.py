r"""Gera os artefatos SHAP do XGBoost final (m3_p1_4).

O script reutiliza o modelo persistido quando disponivel. Se o diretorio local
``models/xgboost`` estiver ausente, reconstrui apenas o modelo final com os
hiperparametros ja escolhidos em ``reports/tuning_xgboost.json``; a busca do
Optuna nao e repetida.

Uso:
    .\.venv\Scripts\python.exe -m scripts.gerar_shap_xgboost
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Matplotlib tenta escrever no perfil do Windows por padrao. Manter o cache no
# projeto evita falha em ambientes restritos; o diretorio permanece ignorado.
os.environ.setdefault("MPLCONFIGDIR", str(RAIZ / ".mplconfig"))

from src.data_loader import carregar_dados
from src.features.pix_features import criar_features_pix
from src.features.preprocessor import (
    COLUNA_ALVO,
    construir_preprocessador,
    dividir_temporal,
    identificar_colunas,
    reduzir_precisao,
)
from src.models.evaluator import avaliar_probabilidades
from src.models.persistencia import carregar, salvar
from src.models.shap_xgboost import (
    SEMENTE,
    TAMANHO_FUNDO,
    amostrar_indices,
    gerar_relatorio_shap,
)
from src.models.xgboost_tuning import criar_modelo, razao_entre_classes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DIRETORIO_MODELO = RAIZ / "models" / "xgboost"
RELATORIO_TUNING = RAIZ / "reports" / "tuning_xgboost.json"
DIRETORIO_SAIDA = RAIZ / "reports" / "pessoa_1" / "julho" / "shap"


def _modelo_persistido_existe() -> bool:
    return (DIRETORIO_MODELO / "manifest.json").is_file()


def _parametros_finais() -> tuple[dict, int]:
    relatorio = json.loads(RELATORIO_TUNING.read_text(encoding="utf-8"))
    parametros = relatorio.get("melhores_parametros")
    n_arvores = relatorio.get("n_arvores_da_melhor")
    if not isinstance(parametros, dict) or not isinstance(n_arvores, int):
        raise ValueError(
            "reports/tuning_xgboost.json nao contem os parametros finais esperados"
        )
    return parametros, n_arvores


def main() -> None:
    processo = psutil.Process(os.getpid())
    inicio = time.time()

    def marco(mensagem: str) -> None:
        logger.info(
            "[%.0fs | pico %.0f MB] %s",
            time.time() - inicio,
            processo.memory_info().peak_wset / 1e6,
            mensagem,
        )

    marco("carregando dataset completo")
    dados = carregar_dados(nrows=None)
    df = reduzir_precisao(criar_features_pix(dados[2]))
    del dados
    gc.collect()

    numericas, cat_baixa, cat_alta = identificar_colunas(df)
    treino, validacao, teste = dividir_temporal(df)
    del df, teste
    gc.collect()

    X_treino = treino.drop(columns=[COLUNA_ALVO])
    y_treino = treino[COLUNA_ALVO]
    X_validacao = validacao.drop(columns=[COLUNA_ALVO])
    y_validacao = validacao[COLUNA_ALVO].to_numpy(dtype=int)
    del treino, validacao
    gc.collect()
    marco("split temporal pronto")

    indices_fundo = amostrar_indices(len(X_treino), TAMANHO_FUNDO, SEMENTE)
    X_fundo = X_treino.iloc[indices_fundo].copy()

    if _modelo_persistido_existe():
        preprocessador, modelo, manifesto = carregar(DIRETORIO_MODELO)
        matriz_fundo = preprocessador.transform(X_fundo)
        matriz_validacao = preprocessador.transform(X_validacao)
        origem_modelo = "artefato local reutilizado"
        marco("modelo e pre-processador persistidos carregados")
    else:
        parametros, n_arvores = _parametros_finais()
        preprocessador = construir_preprocessador(numericas, cat_baixa, cat_alta)
        matriz_treino = preprocessador.fit_transform(X_treino)
        matriz_validacao = preprocessador.transform(X_validacao)
        matriz_fundo = matriz_treino[indices_fundo].copy()
        marco("pre-processamento ajustado somente no treino")

        razao = razao_entre_classes(y_treino)
        modelo = criar_modelo(parametros, razao, n_estimators=n_arvores)
        modelo.fit(matriz_treino, y_treino)
        metricas = avaliar_probabilidades(
            y_validacao,
            modelo.predict_proba(matriz_validacao)[:, 1],
        )
        manifesto = salvar(
            DIRETORIO_MODELO,
            preprocessador,
            modelo,
            metadados={
                "tarefa": "m3_p1_4",
                "origem": "reconstrucao do modelo final para SHAP; Optuna nao repetido",
                "hiperparametros": parametros,
                "n_arvores": n_arvores,
                "scale_pos_weight": razao,
                "linhas_treino": int(matriz_treino.shape[0]),
                "colunas": int(matriz_treino.shape[1]),
                "split": "temporal 70/15/15",
                "metricas_validacao": metricas,
                "metricas_validacao_reconstrucao": metricas,
            },
        )
        del matriz_treino
        gc.collect()
        origem_modelo = "modelo final reconstruido com parametros ja selecionados"
        marco("XGBoost final reconstruido e persistido")

    del X_treino, X_validacao, X_fundo, y_treino
    gc.collect()
    marco("calculando SHAP interventional em escala de probabilidade")

    relatorio = gerar_relatorio_shap(
        modelo,
        preprocessador,
        matriz_fundo,
        matriz_validacao,
        y_validacao,
        DIRETORIO_SAIDA,
    )
    relatorio["modelo"] = {
        "origem": origem_modelo,
        "tipo": manifesto["tipo_do_modelo"],
        "formato": manifesto["formato_do_modelo"],
        "hash_modelo": manifesto["hashes"][manifesto["arquivo_do_modelo"]],
        "hash_preprocessador": manifesto["hashes"]["preprocessador.joblib"],
        "metricas_validacao_reconstrucao": manifesto.get("metadados", {}).get(
            "metricas_validacao_reconstrucao"
        ),
    }
    relatorio["execucao"] = {
        "segundos_total": round(time.time() - inicio),
        "pico_memoria_mb": round(processo.memory_info().peak_wset / 1e6),
    }
    (DIRETORIO_SAIDA / "shap_xgboost.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    marco("artefatos gravados em %s" % DIRETORIO_SAIDA)
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
