"""
Executa a busca de hiperparâmetros do Random Forest e grava o resultado.

Experimento complementar de outubro: aplica ao Random Forest o protocolo de
busca usado no XGBoost (``src/models/random_forest_tuning.py``), para que a
comparação entre os dois não dependa de só um deles ter sido ajustado. Usa
apenas treino e validação; o conjunto de teste é descartado logo após o corte
temporal e nunca é previsto.

Grava ``reports/tuning_random_forest.json``. Nenhum modelo é persistido: o
artefato entregue continua sendo o XGBoost.

Uso:
    .venv/bin/python -m scripts.tunar_random_forest [n_tentativas]
"""

from __future__ import annotations

import gc
import json
import logging
import resource
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data_loader import carregar_dados  # noqa: E402
from src.features.pix_features import criar_features_pix  # noqa: E402
from src.features.preprocessor import (  # noqa: E402
    COLUNA_ALVO,
    construir_preprocessador,
    dividir_temporal,
    identificar_colunas,
    reduzir_precisao,
)
from src.models.evaluator import avaliar_probabilidades  # noqa: E402
from src.models.random_forest import descrever_arvores  # noqa: E402
from src.models.random_forest_tuning import buscar, criar_modelo, resumo  # noqa: E402
from src.models.xgboost_tuning import selecionar_cauda_treino  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SAIDA = RAIZ / "reports" / "tuning_random_forest.json"
STORAGE_OPTUNA = RAIZ / "data" / "processed" / "optuna_random_forest_outubro.db"
AVALIACAO_XGBOOST = RAIZ / "reports" / "avaliacao_xgboost_entregue.json"
LINHAS_BUSCA = 150_000


def _pico_mb() -> float:
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pico / 1e6 if sys.platform == "darwin" else pico / 1e3


def _metricas(metricas: dict) -> dict:
    return {chave: metricas[chave] for chave in ("auc_pr", "auc_roc", "recall", "precisao", "f1")}


def main(n_tentativas: int = 30) -> None:
    inicio = time.time()

    def marco(mensagem: str) -> None:
        logger.info("[%.0fs | pico %.0f MB] %s", time.time() - inicio, _pico_mb(), mensagem)

    marco("carregando dataset completo...")
    dados = carregar_dados(nrows=None)
    df = reduzir_precisao(criar_features_pix(dados[2]))
    del dados
    gc.collect()

    numericas, cat_baixa, cat_alta = identificar_colunas(df)
    treino, validacao, teste = dividir_temporal(df)
    del df, teste  # o teste não participa deste experimento
    gc.collect()
    X_treino, y_treino = treino.drop(columns=[COLUNA_ALVO]), treino[COLUNA_ALVO]
    X_validacao, y_validacao = validacao.drop(columns=[COLUNA_ALVO]), validacao[COLUNA_ALVO]
    del treino, validacao
    gc.collect()

    preprocessador = construir_preprocessador(numericas, cat_baixa, cat_alta)
    M_treino = preprocessador.fit_transform(X_treino)
    M_validacao = preprocessador.transform(X_validacao)
    del X_treino, X_validacao
    gc.collect()
    marco("pré-processamento concluído — treino %s, validação %s" % (M_treino.shape, M_validacao.shape))

    # Referência no mesmo ambiente: a configuração de julho (árvores completas).
    t0 = time.time()
    referencia = criar_modelo({"max_depth": None})
    referencia.fit(M_treino, y_treino)
    metricas_referencia = avaliar_probabilidades(y_validacao, referencia.predict_proba(M_validacao)[:, 1])
    arvores_referencia = descrever_arvores(referencia)
    segundos_referencia = round(time.time() - t0)
    del referencia
    gc.collect()
    marco("referência (árvores completas) — AUC-PR %.4f" % metricas_referencia["auc_pr"])

    M_busca, y_busca = selecionar_cauda_treino(M_treino, y_treino, LINHAS_BUSCA)
    marco("iniciando busca: %d tentativas na cauda de %d linhas" % (n_tentativas, len(y_busca)))
    estudo = buscar(M_busca, y_busca, M_validacao, y_validacao,
                    n_tentativas=n_tentativas, caminho_storage=STORAGE_OPTUNA)
    relatorio = resumo(estudo)
    marco("busca concluída — melhor AUC-PR na cauda %.4f" % relatorio["melhor_auc_pr_busca"])

    # A configuração padrão medida nas mesmas condições da busca: separa "a busca
    # não superou o padrão" de "a busca superou, mas não transferiu ao treino completo".
    padrao_cauda = criar_modelo({"max_depth": None})
    padrao_cauda.fit(M_busca, y_busca)
    auc_pr_padrao_cauda = avaliar_probabilidades(
        y_validacao, padrao_cauda.predict_proba(M_validacao)[:, 1])["auc_pr"]
    del padrao_cauda
    gc.collect()
    marco("referência (árvores completas) na cauda — AUC-PR %.4f" % auc_pr_padrao_cauda)

    # As três melhores da cauda retreinadas no treino completo. A seleção continua
    # na validação; o teste não é usado.
    concluidas = sorted(
        (t for t in relatorio["tentativas"] if t["estado"] == "COMPLETE"),
        key=lambda t: t["auc_pr"], reverse=True,
    )[:3]
    top3_completo = []
    for tentativa in concluidas:
        modelo = criar_modelo(tentativa["parametros"])
        modelo.fit(M_treino, y_treino)
        auc_pr = avaliar_probabilidades(y_validacao, modelo.predict_proba(M_validacao)[:, 1])["auc_pr"]
        top3_completo.append({"numero": tentativa["numero"], "auc_pr_cauda": tentativa["auc_pr"],
                              "auc_pr_treino_completo": auc_pr, "parametros": tentativa["parametros"]})
        del modelo
        gc.collect()
        marco("tentativa %d no treino completo — AUC-PR %.4f" % (tentativa["numero"], auc_pr))

    t0 = time.time()
    final = criar_modelo(relatorio["melhores_parametros"])
    final.fit(M_treino, y_treino)
    metricas_final = avaliar_probabilidades(y_validacao, final.predict_proba(M_validacao)[:, 1])
    segundos_final = round(time.time() - t0)
    marco("Random Forest ajustado, treino completo — AUC-PR %.4f" % metricas_final["auc_pr"])

    xgboost = json.loads(AVALIACAO_XGBOOST.read_text(encoding="utf-8"))
    relatorio.update({
        "referencia_arvores_completas_validacao": _metricas(metricas_referencia),
        "referencia_arvores_completas_estrutura": arvores_referencia,
        "referencia_segundos_treino": segundos_referencia,
        "referencia_arvores_completas_na_cauda_auc_pr": auc_pr_padrao_cauda,
        "top3_da_cauda_no_treino_completo": top3_completo,
        "random_forest_ajustado_validacao": _metricas(metricas_final),
        "random_forest_ajustado_estrutura": descrever_arvores(final),
        "random_forest_ajustado_segundos_treino": segundos_final,
        "xgboost_entregue_validacao_auc_pr": xgboost["selecao_limiar_validacao"]["metricas"]["auc_pr"],
        "_meta": {
            "protocolo": "AUC-PR na validação completa; busca na cauda de 150 mil linhas do treino; "
                         "class_weight='balanced'; 100 árvores; sem uso do teste",
            "linhas_treino": int(M_treino.shape[0]),
            "linhas_busca": int(len(y_busca)),
            "linhas_validacao": int(M_validacao.shape[0]),
            "colunas": int(M_treino.shape[1]),
            "storage_optuna_local": str(STORAGE_OPTUNA.relative_to(RAIZ)),
            "segundos_total": round(time.time() - inicio),
            "pico_memoria_mb": round(_pico_mb()),
        },
    })
    SAIDA.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    marco("relatório gravado em %s" % SAIDA.relative_to(RAIZ))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
