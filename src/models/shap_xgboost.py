"""Explicabilidade global e local do XGBoost com SHAP.

O modulo implementa o protocolo definido para a Camada 2 do TCC:

* ``TreeExplainer`` no modo ``interventional``;
* fundo aleatorio de 500 linhas, retirado exclusivamente do treino;
* contribuicoes na escala de probabilidade;
* verificacao numerica da propriedade aditiva;
* importancia global e casos locais VP, FP, FN e VN.

O SHAP e importado apenas nas funcoes que geram os artefatos. Assim, as
funcoes puras de selecao e validacao continuam leves e testaveis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

SEMENTE = 42
TAMANHO_FUNDO = 500
TAMANHO_AMOSTRA_GLOBAL = 1_000
LIMIAR_PADRAO = 0.5
TOLERANCIA_ADITIVIDADE = 1e-5


def amostrar_indices(
    n_linhas: int,
    tamanho: int,
    semente: int = SEMENTE,
) -> np.ndarray:
    """Retorna indices aleatorios, unicos e ordenados de forma deterministica."""
    if n_linhas <= 0:
        raise ValueError("n_linhas deve ser maior que zero")
    if tamanho <= 0:
        raise ValueError("tamanho deve ser maior que zero")

    quantidade = min(n_linhas, tamanho)
    gerador = np.random.default_rng(semente)
    return np.sort(gerador.choice(n_linhas, size=quantidade, replace=False))


def nomes_das_features(preprocessador: Any, n_colunas: int) -> list[str]:
    """Extrai nomes do ``ColumnTransformer`` e remove prefixos tecnicos."""
    try:
        nomes = [str(nome) for nome in preprocessador.get_feature_names_out()]
    except (AttributeError, TypeError, ValueError):
        nomes = [f"feature_{indice}" for indice in range(n_colunas)]

    if len(nomes) != n_colunas:
        raise ValueError(
            "o pre-processador informou "
            f"{len(nomes)} features, mas a matriz possui {n_colunas} colunas"
        )

    # ``ColumnTransformer`` produz, por exemplo, ``numericas__TransactionAmt``.
    # O prefixo ajuda no codigo, mas atrapalha a leitura dos graficos.
    return [nome.split("__", maxsplit=1)[-1] for nome in nomes]


def selecionar_casos_representativos(
    y_verdadeiro,
    probabilidades,
    limiar: float = LIMIAR_PADRAO,
) -> dict[str, int]:
    """Seleciona o caso mais proximo da mediana de cada quadrante de erro.

    Usar a mediana evita escolher somente exemplos extremos. Quadrantes vazios
    sao omitidos, em vez de fabricar um caso que nao existe na amostra.
    """
    y = np.asarray(y_verdadeiro, dtype=int).reshape(-1)
    proba = np.asarray(probabilidades, dtype=float).reshape(-1)
    if len(y) != len(proba):
        raise ValueError("alvo e probabilidades precisam ter o mesmo tamanho")
    if not 0.0 < limiar < 1.0:
        raise ValueError("limiar deve estar entre zero e um")
    if not np.isfinite(proba).all() or np.any((proba < 0) | (proba > 1)):
        raise ValueError("probabilidades devem ser finitas e estar em [0, 1]")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("o alvo deve ser binario, com valores 0 e 1")

    previsto = (proba >= limiar).astype(int)
    mascaras = {
        "verdadeiro_positivo": (y == 1) & (previsto == 1),
        "falso_positivo": (y == 0) & (previsto == 1),
        "falso_negativo": (y == 1) & (previsto == 0),
        "verdadeiro_negativo": (y == 0) & (previsto == 0),
    }

    selecionados: dict[str, int] = {}
    for rotulo, mascara in mascaras.items():
        candidatos = np.flatnonzero(mascara)
        if candidatos.size == 0:
            continue
        mediana = float(np.median(proba[candidatos]))
        deslocamento = np.abs(proba[candidatos] - mediana)
        selecionados[rotulo] = int(candidatos[int(np.argmin(deslocamento))])
    return selecionados


def importancia_global(
    valores_shap,
    nomes_features: list[str],
) -> list[dict[str, float | int | str]]:
    """Ordena as features pela media do valor SHAP absoluto."""
    valores = np.asarray(valores_shap, dtype=float)
    if valores.ndim != 2:
        raise ValueError("valores_shap deve ser uma matriz 2D")
    if valores.shape[1] != len(nomes_features):
        raise ValueError("quantidade de nomes nao corresponde aos valores SHAP")
    if not np.isfinite(valores).all():
        raise ValueError("valores SHAP contem NaN ou infinito")

    medias = np.mean(np.abs(valores), axis=0)
    ordem = np.argsort(-medias, kind="stable")
    return [
        {
            "posicao": posicao,
            "feature": nomes_features[indice],
            "media_shap_absoluto": float(medias[indice]),
        }
        for posicao, indice in enumerate(ordem, start=1)
    ]


def erro_maximo_aditividade(
    valores_shap,
    valores_base,
    probabilidades,
) -> float:
    """Calcula o maior erro de ``base + soma(phi) == probabilidade``."""
    valores = np.asarray(valores_shap, dtype=float)
    base = np.asarray(valores_base, dtype=float).reshape(-1)
    proba = np.asarray(probabilidades, dtype=float).reshape(-1)
    if valores.ndim != 2 or valores.shape[0] != len(base) or len(base) != len(proba):
        raise ValueError("dimensoes incompativeis na verificacao de aditividade")
    return float(np.max(np.abs(base + valores.sum(axis=1) - proba)))


def _numero(valor: Any) -> float | int | None:
    """Converte escalares NumPy em JSON e representa nao finitos como nulo."""
    convertido = float(valor)
    if not np.isfinite(convertido):
        return None
    if convertido.is_integer():
        return int(convertido)
    return convertido


def _salvar_summary_plot(
    valores_shap: np.ndarray,
    dados: np.ndarray,
    nomes_features: list[str],
    destino: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import shap

    destino.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(
        valores_shap,
        dados,
        feature_names=nomes_features,
        max_display=20,
        show=False,
        plot_size=(11, 8),
    )
    figura = plt.gcf()
    figura.suptitle(
        "XGBoost — importancia global SHAP na validacao",
        fontsize=13,
        y=1.01,
    )
    figura.savefig(destino, dpi=180, bbox_inches="tight")
    plt.close(figura)


def _salvar_force_plots(
    valor_base: float,
    valores_shap: np.ndarray,
    dados: np.ndarray,
    nomes_features: list[str],
    destino_png: Path,
    destino_html: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import shap

    destino_png.parent.mkdir(parents=True, exist_ok=True)
    visualizacao = shap.force_plot(
        valor_base,
        valores_shap,
        dados,
        feature_names=nomes_features,
        link="identity",
        contribution_threshold=0.02,
        matplotlib=False,
        show=False,
    )
    # SHAP 0.52 aceita caminho somente como ``str`` (nao ``PathLike``).
    shap.save_html(str(destino_html), visualizacao, full_html=True)

    shap.force_plot(
        valor_base,
        valores_shap,
        dados,
        feature_names=nomes_features,
        link="identity",
        contribution_threshold=0.02,
        matplotlib=True,
        show=False,
        figsize=(18, 4),
        text_rotation=15,
    )
    figura = plt.gcf()
    figura.savefig(destino_png, dpi=180, bbox_inches="tight")
    plt.close(figura)


def gerar_relatorio_shap(
    modelo: Any,
    preprocessador: Any,
    matriz_treino,
    matriz_validacao,
    y_validacao,
    diretorio_saida: Path | str,
    *,
    tamanho_fundo: int = TAMANHO_FUNDO,
    tamanho_amostra_global: int = TAMANHO_AMOSTRA_GLOBAL,
    limiar: float = LIMIAR_PADRAO,
    semente: int = SEMENTE,
) -> dict[str, Any]:
    """Calcula SHAP e grava summary/force plots mais um relatorio JSON."""
    import shap
    import sklearn
    import xgboost

    treino = np.asarray(matriz_treino)
    validacao = np.asarray(matriz_validacao)
    y = np.asarray(y_validacao, dtype=int).reshape(-1)
    if treino.ndim != 2 or validacao.ndim != 2:
        raise ValueError("matrizes de treino e validacao devem ser bidimensionais")
    if treino.shape[1] != validacao.shape[1]:
        raise ValueError("treino e validacao devem ter as mesmas features")
    if len(validacao) != len(y):
        raise ValueError("matriz de validacao e alvo devem ter o mesmo tamanho")

    nomes_features = nomes_das_features(preprocessador, treino.shape[1])
    indices_fundo = amostrar_indices(len(treino), tamanho_fundo, semente)
    indices_globais = amostrar_indices(
        len(validacao), tamanho_amostra_global, semente + 1
    )

    probabilidades_validacao = np.asarray(
        modelo.predict_proba(validacao)[:, 1], dtype=float
    )
    casos = selecionar_casos_representativos(y, probabilidades_validacao, limiar)

    indices_exp = list(map(int, indices_globais))
    for indice in casos.values():
        if indice not in indices_exp:
            indices_exp.append(indice)
    matriz_exp = validacao[indices_exp]

    try:
        # Sem ``max_samples`` explicito, SHAP 0.52 reduz silenciosamente fundos
        # maiores para 100 linhas. O masker fixa as 500 linhas previstas no
        # protocolo metodologico e torna o tamanho auditavel.
        masker_fundo = shap.maskers.Independent(
            treino[indices_fundo], max_samples=len(indices_fundo)
        )
        explicador = shap.TreeExplainer(
            modelo,
            data=masker_fundo,
            feature_perturbation="interventional",
            model_output="probability",
        )
        if len(explicador.data) != len(indices_fundo):
            raise RuntimeError(
                "o SHAP alterou o tamanho do fundo: "
                f"esperado {len(indices_fundo)}, obtido {len(explicador.data)}"
            )
        explicacao = explicador(matriz_exp, check_additivity=False)
    except NotImplementedError as erro:
        raise RuntimeError(
            "A versao instalada do XGBoost nao e compativel com o "
            "TreeExplainer interventional do SHAP. Use XGBoost 3.0.5, "
            "versao validada por esta implementacao."
        ) from erro

    valores = np.asarray(explicacao.values, dtype=float)
    bases = np.asarray(explicacao.base_values, dtype=float).reshape(-1)
    probabilidades_exp = probabilidades_validacao[indices_exp]
    erro_aditividade = erro_maximo_aditividade(
        valores, bases, probabilidades_exp
    )
    if erro_aditividade > TOLERANCIA_ADITIVIDADE:
        raise ValueError(
            "falha na fidelidade SHAP: erro aditivo maximo "
            f"{erro_aditividade:.3e} excede {TOLERANCIA_ADITIVIDADE:.1e}"
        )

    saida = Path(diretorio_saida)
    saida.mkdir(parents=True, exist_ok=True)
    caminho_resumo = saida / "shap_summary_xgboost.png"
    quantidade_global = len(indices_globais)
    _salvar_summary_plot(
        valores[:quantidade_global],
        matriz_exp[:quantidade_global],
        nomes_features,
        caminho_resumo,
    )

    posicao_por_indice = {
        indice_validacao: posicao
        for posicao, indice_validacao in enumerate(indices_exp)
    }
    casos_relatorio: list[dict[str, Any]] = []
    for rotulo, indice_validacao in casos.items():
        posicao = posicao_por_indice[indice_validacao]
        valores_caso = valores[posicao]
        dados_caso = matriz_exp[posicao]
        caminho_png = saida / f"shap_force_{rotulo}.png"
        caminho_html = saida / f"shap_force_{rotulo}.html"
        _salvar_force_plots(
            float(bases[posicao]),
            valores_caso,
            dados_caso,
            nomes_features,
            caminho_png,
            caminho_html,
        )

        ordem = np.argsort(-np.abs(valores_caso), kind="stable")[:10]
        casos_relatorio.append(
            {
                "tipo": rotulo,
                "posicao_na_validacao": int(indice_validacao),
                "classe_real": int(y[indice_validacao]),
                "classe_predita": int(
                    probabilidades_validacao[indice_validacao] >= limiar
                ),
                "probabilidade_fraude": float(
                    probabilidades_validacao[indice_validacao]
                ),
                "valor_base": float(bases[posicao]),
                "force_plot_png": caminho_png.name,
                "force_plot_html": caminho_html.name,
                "principais_contribuicoes": [
                    {
                        "feature": nomes_features[indice_feature],
                        "valor_transformado": _numero(dados_caso[indice_feature]),
                        "shap": float(valores_caso[indice_feature]),
                        "direcao": (
                            "aumenta_risco"
                            if valores_caso[indice_feature] > 0
                            else "reduz_risco"
                        ),
                    }
                    for indice_feature in ordem
                ],
            }
        )

    relatorio: dict[str, Any] = {
        "schema_version": "1.0",
        "metodologia": {
            "explicador": "shap.TreeExplainer",
            "feature_perturbation": "interventional",
            "model_output": "probability",
            "tamanho_fundo_treino": int(len(indices_fundo)),
            "tamanho_amostra_global_validacao": int(len(indices_globais)),
            "semente": int(semente),
            "limiar": float(limiar),
            "selecao_local": "caso mais proximo da probabilidade mediana de cada quadrante",
        },
        "versoes": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "fidelidade": {
            "equacao": "valor_base + soma(SHAP) = probabilidade_fraude",
            "erro_maximo": erro_aditividade,
            "tolerancia": TOLERANCIA_ADITIVIDADE,
            "aprovada": True,
        },
        "importancia_global": importancia_global(
            valores[:quantidade_global], nomes_features
        ),
        "casos_locais": casos_relatorio,
        "artefatos": {
            "summary_plot": caminho_resumo.name,
            "force_plots": [
                {
                    "tipo": caso["tipo"],
                    "png": caso["force_plot_png"],
                    "html": caso["force_plot_html"],
                }
                for caso in casos_relatorio
            ],
        },
        "ressalvas": [
            "Contribuicao SHAP descreve influencia sobre o modelo, nao causalidade.",
            "Colunas V e outras features anonimizadas nao recebem significado Pix.",
            "O IEEE-CIS representa comercio eletronico/cartao, nao transacoes Pix reais.",
            "Uma explicacao fiel ao modelo nao garante que a classificacao esteja correta.",
        ],
    }
    (saida / "shap_xgboost.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return relatorio
