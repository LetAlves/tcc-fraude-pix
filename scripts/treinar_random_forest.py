"""
Treina o Random Forest comparativo (m3_p1_2) e grava o resultado.

Roda as duas configurações definidas em `src/models/random_forest.py` sobre o
conjunto de treino completo, mede AUC-PR, F1 e tempo de treino — os três itens
que a tarefa pede — e grava `reports/random_forest.json`, consumido pelo
notebook comparativo da m3_p1_5.

Uso:
    .\\.venv\\Scripts\\python.exe scripts/treinar_random_forest.py
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import psutil

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data_loader import carregar_dados
from src.features.pix_features import criar_features_pix
from src.features.preprocessor import (COLUNA_ALVO, construir_preprocessador,
                                       dividir_temporal, identificar_colunas,
                                       reduzir_precisao)
from src.models.evaluator import avaliar_probabilidades
from src.models.random_forest import CONFIGURACOES, criar_modelo, descrever_arvores

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SAIDA = RAIZ / "reports" / "random_forest.json"


def main() -> None:
    processo = psutil.Process(os.getpid())
    inicio = time.time()

    def marco(mensagem: str) -> None:
        logger.info("[%.0fs | pico %.0f MB] %s", time.time() - inicio,
                    processo.memory_info().peak_wset / 1e6, mensagem)

    marco("carregando dataset completo...")
    dados = carregar_dados(nrows=None)
    df = reduzir_precisao(criar_features_pix(dados[2]))
    del dados
    gc.collect()

    numericas, cat_baixa, cat_alta = identificar_colunas(df)
    treino, validacao, _ = dividir_temporal(df)
    X_treino, y_treino = treino.drop(columns=[COLUNA_ALVO]), treino[COLUNA_ALVO]
    X_validacao, y_validacao = validacao.drop(columns=[COLUNA_ALVO]), validacao[COLUNA_ALVO]
    del df, treino, validacao
    gc.collect()

    preprocessador = construir_preprocessador(numericas, cat_baixa, cat_alta)
    M_treino = preprocessador.fit_transform(X_treino)
    M_validacao = preprocessador.transform(X_validacao)
    del X_treino, X_validacao
    gc.collect()
    marco("dados prontos — matriz de treino %s" % (M_treino.shape,))

    resultados = {}
    for nome, parametros in CONFIGURACOES.items():
        marco("treinando %s..." % nome)
        modelo = criar_modelo(**parametros)

        comeco = time.time()
        modelo.fit(M_treino, y_treino)
        segundos = time.time() - comeco

        metricas = avaliar_probabilidades(
            y_validacao, modelo.predict_proba(M_validacao)[:, 1])
        resultados[nome] = {
            **metricas,
            "segundos_treino": round(segundos, 1),
            "arvores": descrever_arvores(modelo),
            "max_depth": parametros["max_depth"],
        }
        marco("%s — auc_pr %.4f | f1 %.4f | %.0fs | profundidade média %.1f" % (
            nome, metricas["auc_pr"], metricas["f1"], segundos,
            resultados[nome]["arvores"]["profundidade_media"]))
        del modelo
        gc.collect()

    resultados["referencias_validacao"] = {
        "baseline_regressao_logistica": {"auc_pr": 0.3930, "f1": 0.2251},
        "xgboost_padrao": {"auc_pr": 0.5293},
    }
    resultados["_meta"] = {
        "linhas_treino": int(M_treino.shape[0]),
        "colunas": int(M_treino.shape[1]),
        "segundos": round(time.time() - inicio),
        "pico_mb": round(processo.memory_info().peak_wset / 1e6),
    }

    SAIDA.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    marco("resultado gravado em %s" % SAIDA)


if __name__ == "__main__":
    main()
