"""
Executa a busca de hiperparâmetros do XGBoost (m3_p1_1) e grava o resultado.

Roda fora do notebook de propósito: a busca leva dezenas de minutos e consome
memória, e o kernel do Jupyter segura tudo o que já foi executado. O notebook
comparativo (m3_p1_5) consome o JSON produzido aqui.

Uso:
    .\\.venv\\Scripts\\python.exe -m scripts.tunar_xgboost [n_tentativas]
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
from src.models.persistencia import salvar
from src.models.xgboost_tuning import buscar, criar_modelo, razao_entre_classes, resumo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SAIDA = RAIZ / "reports" / "tuning_xgboost.json"
DIRETORIO_MODELO = RAIZ / "models" / "xgboost"


def main(n_tentativas: int = 50) -> None:
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
    marco("dados prontos — treino %d x %d" % X_treino.shape)

    # Ajustado só no treino, uma vez. Não depende dos hiperparâmetros.
    preprocessador = construir_preprocessador(numericas, cat_baixa, cat_alta)
    M_treino = preprocessador.fit_transform(X_treino)
    M_validacao = preprocessador.transform(X_validacao)
    del X_treino, X_validacao
    gc.collect()
    marco("pré-processamento concluído — matriz %s" % (M_treino.shape,))

    marco("iniciando busca com %d tentativas..." % n_tentativas)
    estudo = buscar(M_treino, y_treino, M_validacao, y_validacao, n_tentativas=n_tentativas)
    relatorio = resumo(estudo)
    marco("busca concluída — melhor AUC-PR %.4f" % relatorio["melhor_auc_pr"])

    # Referência: o mesmo XGBoost sem ajuste nenhum, para isolar o ganho do Optuna.
    razao = razao_entre_classes(y_treino)
    padrao = criar_modelo({}, razao, n_estimators=100)
    padrao.fit(M_treino, y_treino)
    metricas_padrao = avaliar_probabilidades(
        y_validacao, padrao.predict_proba(M_validacao)[:, 1])
    marco("XGBoost padrão (referência) — AUC-PR %.4f" % metricas_padrao["auc_pr"])

    relatorio["referencia_xgboost_padrao"] = {
        chave: metricas_padrao[chave]
        for chave in ("auc_pr", "auc_roc", "recall", "precisao", "f1")
    }
    relatorio["baseline_junho_validacao"] = {
        "auc_pr": 0.3930, "auc_roc": 0.8410, "recall": 0.6772, "precisao": 0.1350,
    }
    relatorio["_meta"] = {
        "scale_pos_weight": razao,
        "linhas_treino": int(M_treino.shape[0]),
        "colunas": int(M_treino.shape[1]),
        "segundos": round(time.time() - inicio),
        "pico_mb": round(processo.memory_info().peak_wset / 1e6),
    }

    # Modelo final: os melhores hiperparametros, com o numero de arvores que a
    # parada antecipada selecionou naquela tentativa. Sem fixar esse numero, o
    # modelo final treinaria ate o teto e seria diferente do que foi avaliado.
    marco("treinando o modelo final com os melhores hiperparametros...")
    final = criar_modelo(
        relatorio["melhores_parametros"], razao,
        n_estimators=relatorio["n_arvores_da_melhor"],
    )
    final.fit(M_treino, y_treino)
    metricas_final = avaliar_probabilidades(
        y_validacao, final.predict_proba(M_validacao)[:, 1])
    relatorio["metricas_modelo_final_validacao"] = metricas_final
    marco("modelo final — AUC-PR %.4f" % metricas_final["auc_pr"])

    salvar(
        DIRETORIO_MODELO, preprocessador, final,
        metadados={
            "tarefa": "m3_p1_1",
            "papel": "modelo principal da proposta aprovada",
            "hiperparametros": relatorio["melhores_parametros"],
            "n_arvores": relatorio["n_arvores_da_melhor"],
            "n_tentativas": relatorio["n_tentativas"],
            "scale_pos_weight": razao,
            "linhas_treino": int(M_treino.shape[0]),
            "colunas": int(M_treino.shape[1]),
            "split": "temporal 70/15/15",
            "metricas_validacao": metricas_final,
        },
    )
    relatorio["modelo_salvo_em"] = str(DIRETORIO_MODELO)
    marco("modelo persistido em %s" % DIRETORIO_MODELO)

    SAIDA.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")
    marco("resultado gravado em %s" % SAIDA)
    print(json.dumps(relatorio, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
