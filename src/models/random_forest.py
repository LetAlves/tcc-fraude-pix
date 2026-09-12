"""
Random Forest — modelo comparativo da proposta aprovada.

O papel dele no trabalho é servir de contraponto ao XGBoost, que é o modelo
principal. Não há busca de hiperparâmetros aqui: comparar um modelo ajustado
com outro sem ajuste mediria o esforço de ajuste, não os algoritmos.

Duas configurações são treinadas, e a razão é um achado da própria medição:
com árvores completas o Random Forest chega a **profundidade média 66**, o que
é memorização de transações individuais — e, ainda assim, a AUC-PR é
praticamente igual à de árvores limitadas a 20 níveis. Reportar as duas
transforma isso em evidência na tabela comparativa, em vez de uma afirmação.

`class_weight='balanced'` é o equivalente da ponderação de classe adotada em
junho, e é o mesmo tratamento de desbalanceamento usado pela regressão
logística e pelo XGBoost (lá, via `scale_pos_weight`). Assim a comparação mede
o algoritmo, não a estratégia de desbalanceamento.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

N_ARVORES = 100
SEMENTE = 42

# Nome legível -> parâmetros que diferem entre as configurações comparadas.
CONFIGURACOES = {
    "arvores_completas": {"max_depth": None},
    "profundidade_20": {"max_depth": 20},
}


def criar_modelo(max_depth: int | None = None, n_estimators: int = N_ARVORES) -> RandomForestClassifier:
    """Random Forest com os parâmetros fixos do projeto."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEMENTE,
    )


def descrever_arvores(modelo: RandomForestClassifier) -> dict:
    """
    Estatísticas da floresta treinada.

    A profundidade não é curiosidade: ela é o que revela se o modelo está
    generalizando ou memorizando, e é o número que justifica comparar duas
    configurações em vez de aceitar o padrão do scikit-learn sem discussão.
    """
    profundidades = [estimador.get_depth() for estimador in modelo.estimators_]
    folhas = [estimador.get_n_leaves() for estimador in modelo.estimators_]
    return {
        "n_arvores": len(modelo.estimators_),
        "profundidade_media": float(np.mean(profundidades)),
        "profundidade_maxima": int(np.max(profundidades)),
        "folhas_media": float(np.mean(folhas)),
    }
