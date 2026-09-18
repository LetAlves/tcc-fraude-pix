"""
Exporta transações reais do conjunto de teste para a interface de demonstração.

A interface não pode pedir que alguém digite uma transação: o modelo recebe 439
colunas, e preencher meia dúzia completando o resto com valores inventados
produziria uma probabilidade que não corresponde a transação alguma. Selecionar
casos reais mantém a demonstração honesta.

A seleção é estratificada por quadrante da matriz de confusão — verdadeiro
positivo, falso positivo, falso negativo e verdadeiro negativo — porque uma
demonstração que só mostra acertos esconde o comportamento real do sistema, e o
falso positivo é o caso mais instrutivo: no conjunto de teste, cerca de 8 em
cada 10 transações marcadas não eram fraude.

O arquivo gerado fica em `data/`, ignorado pelo Git: são linhas do IEEE-CIS, e
as regras da competição não autorizam redistribuição.

Uso:
    .\\.venv\\Scripts\\python.exe scripts/exportar_exemplos_interface.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.data_loader import carregar_dados
from src.features.pix_features import criar_features_pix
from src.features.preprocessor import (COLUNA_ALVO, dividir_temporal,
                                       reduzir_precisao)
from src.models.persistencia import carregar

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SAIDA = RAIZ / "data" / "exemplos_interface.json"
POR_QUADRANTE = 3
SEMENTE = 42

# Campos mostrados na interface. Os demais viajam junto porque o modelo precisa
# deles, mas não são exibidos: não têm significado publicado.
CAMPOS_LEGIVEIS = [
    "TransactionAmt", "ProductCD", "card4", "card6", "P_emaildomain",
    "DeviceType", "DeviceInfo", "TransactionDT",
    "valor_atipico_proxy", "frequencia_recente_proxy",
    "dispositivo_raro_proxy", "posicao_ciclo_diario_relativa",
]


def main(diretorio_modelo: Path = RAIZ / "models" / "xgboost") -> None:
    if not diretorio_modelo.exists():
        alternativa = RAIZ / "models" / "random_forest"
        if not alternativa.exists():
            raise SystemExit(
                f"Nenhum modelo encontrado em {diretorio_modelo} nem em {alternativa}. "
                "Rode scripts/tunar_xgboost.py ou scripts/treinar_random_forest.py."
            )
        logger.warning("modelo XGBoost ausente; usando %s", alternativa.name)
        diretorio_modelo = alternativa

    logger.info("carregando dataset completo...")
    dados = carregar_dados(nrows=None)
    df = reduzir_precisao(criar_features_pix(dados[2]))
    del dados

    _, _, teste = dividir_temporal(df)
    X_teste, y_teste = teste.drop(columns=[COLUNA_ALVO]), teste[COLUNA_ALVO]
    del df, teste

    preprocessador, modelo, manifesto = carregar(diretorio_modelo)
    logger.info("modelo: %s", manifesto["tipo_do_modelo"])

    probabilidades = modelo.predict_proba(preprocessador.transform(X_teste))[:, 1]
    sinalizada = probabilidades >= 0.5
    real = y_teste.to_numpy().astype(bool)

    quadrantes = {
        "verdadeiro_positivo": sinalizada & real,
        "falso_positivo": sinalizada & ~real,
        "falso_negativo": ~sinalizada & real,
        "verdadeiro_negativo": ~sinalizada & ~real,
    }

    rng = np.random.default_rng(SEMENTE)
    exemplos = []
    for nome, mascara in quadrantes.items():
        posicoes = np.flatnonzero(mascara)
        if posicoes.size == 0:
            logger.warning("quadrante %s vazio", nome)
            continue
        escolhidas = rng.choice(posicoes, min(POR_QUADRANTE, posicoes.size), replace=False)
        for posicao in escolhidas:
            linha = X_teste.iloc[int(posicao)]
            exemplos.append({
                "quadrante": nome,
                "rotulo_real": int(real[int(posicao)]),
                "probabilidade_registrada": float(probabilidades[int(posicao)]),
                "resumo": {
                    campo: _json_seguro(linha.get(campo))
                    for campo in CAMPOS_LEGIVEIS if campo in linha.index
                },
                "transacao": {
                    coluna: _json_seguro(valor) for coluna, valor in linha.items()
                },
            })

    exemplos.sort(key=lambda e: -e["probabilidade_registrada"])
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        json.dumps(
            {
                "modelo": manifesto["tipo_do_modelo"],
                "gerado_em": manifesto["gerado_em"],
                "n_colunas": int(X_teste.shape[1]),
                "campos_legiveis": CAMPOS_LEGIVEIS,
                "exemplos": exemplos,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    logger.info("%d exemplos gravados em %s", len(exemplos), SAIDA)


def _json_seguro(valor):
    """Converte tipos do numpy/pandas para algo serializável, preservando nulos."""
    if valor is None:
        return None
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        numero = float(valor)
        return None if np.isnan(numero) else numero
    if isinstance(valor, (np.bool_, bool)):
        return bool(valor)
    texto = str(valor)
    return None if texto in ("nan", "NaT", "<NA>") else texto


if __name__ == "__main__":
    main()
