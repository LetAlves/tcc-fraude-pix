"""
Avaliação complementar dos modelos comparativos no conjunto de teste.

O teste foi reservado para avaliar uma única vez o modelo escolhido, o XGBoost
entregue, cuja avaliação está em ``reports/avaliacao_xgboost_entregue.json``.
Este script mede, depois dessa escolha, a AUC-PR e a AUC-ROC de teste dos
modelos usados só para comparação, para que a tabela de resultados não fique
com lacunas. A escolha do modelo entregue não depende destes números e não
será revista com base neles.

Validação e teste de cada modelo vêm da mesma execução, para que a linha da
tabela seja comparável. São métricas de ordenação, independentes de limiar.

Uso:
    .venv/bin/python -m scripts.avaliar_comparativos_teste
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
from src.models.random_forest_tuning import criar_modelo as criar_random_forest  # noqa: E402
from src.models.xgboost_tuning import criar_modelo as criar_xgboost  # noqa: E402
from src.models.xgboost_tuning import razao_entre_classes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SAIDA = RAIZ / "reports" / "avaliacao_comparativos_teste.json"
TUNING_RF = RAIZ / "reports" / "tuning_random_forest.json"


def _pico_mb() -> float:
    pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return pico / 1e6 if sys.platform == "darwin" else pico / 1e3


def main() -> None:
    inicio = time.time()

    def marco(mensagem: str) -> None:
        logger.info("[%.0fs | pico %.0f MB] %s", time.time() - inicio, _pico_mb(), mensagem)

    dados = carregar_dados(nrows=None)
    df = reduzir_precisao(criar_features_pix(dados[2]))
    del dados
    gc.collect()
    numericas, cat_baixa, cat_alta = identificar_colunas(df)
    treino, validacao, teste = dividir_temporal(df)
    del df
    gc.collect()

    preprocessador = construir_preprocessador(numericas, cat_baixa, cat_alta)
    M_treino = preprocessador.fit_transform(treino.drop(columns=[COLUNA_ALVO]))
    y_treino = treino[COLUNA_ALVO]
    M_validacao = preprocessador.transform(validacao.drop(columns=[COLUNA_ALVO]))
    y_validacao = validacao[COLUNA_ALVO]
    M_teste = preprocessador.transform(teste.drop(columns=[COLUNA_ALVO]))
    y_teste = teste[COLUNA_ALVO]
    del treino, validacao, teste
    gc.collect()
    marco("pré-processamento ajustado no treino — %d colunas" % M_treino.shape[1])

    parametros_rf = json.loads(TUNING_RF.read_text(encoding="utf-8"))["melhores_parametros"]
    modelos = {
        "random_forest_arvores_completas": lambda: criar_random_forest({"max_depth": None}),
        "random_forest_ajustado": lambda: criar_random_forest(parametros_rf),
        "xgboost_padrao": lambda: criar_xgboost({}, razao_entre_classes(y_treino), n_estimators=100),
    }

    resultados = {}
    for nome, fabrica in modelos.items():
        modelo = fabrica()
        modelo.fit(M_treino, y_treino)
        resultados[nome] = {}
        for particao, M, y in (("validacao", M_validacao, y_validacao), ("teste", M_teste, y_teste)):
            metricas = avaliar_probabilidades(y, modelo.predict_proba(M)[:, 1])
            resultados[nome][particao] = {"auc_pr": metricas["auc_pr"], "auc_roc": metricas["auc_roc"]}
        del modelo
        gc.collect()
        marco("%s — AUC-PR validação %.4f | teste %.4f" % (
            nome, resultados[nome]["validacao"]["auc_pr"], resultados[nome]["teste"]["auc_pr"]))

    relatorio = {
        "natureza": "avaliação complementar, posterior à escolha e à avaliação do modelo entregue; "
                    "não altera o modelo entregue nem o limiar",
        "metricas": "AUC-PR e AUC-ROC, independentes de limiar",
        "modelos": resultados,
        "_meta": {
            "linhas": {"treino": int(M_treino.shape[0]), "validacao": int(M_validacao.shape[0]),
                       "teste": int(M_teste.shape[0])},
            "parametros_random_forest_ajustado": parametros_rf,
            "segundos_total": round(time.time() - inicio),
            "pico_memoria_mb": round(_pico_mb()),
        },
    }
    SAIDA.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    marco("relatório gravado em %s" % SAIDA.relative_to(RAIZ))


if __name__ == "__main__":
    main()
