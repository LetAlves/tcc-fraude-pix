"""
Pipeline completo: transação → XGBoost → SHAP → RAG → explicação.

Orquestra as três camadas da arquitetura aprovada sem reimplementar nenhuma
delas. Cada etapa vem de um módulo que já existe e já tem teste próprio:

| Etapa | Módulo |
|---|---|
| pré-processamento e modelo | `src.models.persistencia` |
| nomes das features | `src.models.shap_xgboost` |
| recuperação documental | `src.rag.retriever` |
| explicação textual | `src.rag.explainer` |

## O formato do retorno não é escolha deste módulo

`app.py` valida o resultado do pipeline. O contrato é aninhado, e o detalhe
importa: `predicao` é um **objeto** com `classe`, `probabilidade` e `limiar`,
não um texto. Devolver a classe direto em `predicao` passa por uma verificação
superficial de chaves e falha na validação real da interface.

O contrato completo está documentado em `validar_resultado_pipeline`, em
`app.py`, e o teste deste módulo o executa de verdade em vez de reafirmar a
suposição de quem escreveu o pipeline.

## Por que o pré-processador vem junto do modelo

`carregar` devolve os dois porque o modelo recebe uma matriz de 460 colunas
construída por um `ColumnTransformer` ajustado no treino. Uma transação nova só
vira entrada válida passando por *aquele* ajuste. Aplicar o modelo a um
DataFrame cru não levanta exceção: devolve números errados.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.persistencia import carregar
from src.rag.explainer import explicar
from src.rag.retriever import RecuperadorDocumentos, consulta_a_partir_dos_fatores

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
DIRETORIO_MODELO_PADRAO = RAIZ / "models" / "xgboost"
LIMIAR_PADRAO = 0.5
TOP_FATORES = 3
TOP_DOCUMENTOS = 5


@dataclass
class Pipeline:
    """
    Componentes carregados uma vez e reutilizados a cada transação.

    Carregar modelo, explicador SHAP e índice vetorial custa segundos; fazer
    isso por transação inviabilizaria qualquer demonstração interativa.
    """

    preprocessador: Any
    modelo: Any
    explicador_shap: Any
    recuperador: RecuperadorDocumentos | None = None
    nomes_features: list[str] | None = None
    limiar: float = LIMIAR_PADRAO
    cliente_llm: Callable[[str], str] | None = None

    def processar(self, transacao: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Executa as três camadas para uma transação e devolve o resultado."""
        quadro = _como_quadro(transacao)
        matriz = self.preprocessador.transform(quadro)

        probabilidade = float(self.modelo.predict_proba(matriz)[0, 1])
        sinalizada = probabilidade >= self.limiar

        fatores = self._fatores_shap(matriz)

        documentos: list[Any] = []
        if self.recuperador is not None:
            consulta = consulta_a_partir_dos_fatores(fatores, sinalizada)
            documentos = self.recuperador.recuperar(consulta, top_k=TOP_DOCUMENTOS)

        explicacao = explicar(
            fatores_shap=fatores,
            documentos=documentos,
            probabilidade=probabilidade,
            sinalizada=sinalizada,
            cliente=self.cliente_llm,
        )

        return {
            "predicao": {
                "classe": "suspeita" if sinalizada else "não suspeita",
                "probabilidade": probabilidade,
                "limiar": self.limiar,
            },
            "fatores_shap": fatores,
            "documentos_recuperados": [
                {
                    "titulo": documento.fonte,
                    "trecho": documento.resumo(),
                    "fonte": documento.fonte,
                    "score": documento.score,
                    "metadados": documento.metadados,
                }
                for documento in documentos
            ],
            "explicacao_rag": explicacao,
        }

    def _fatores_shap(self, matriz: Any) -> list[dict[str, Any]]:
        """
        Os `TOP_FATORES` de maior contribuição absoluta, com sinal preservado.

        O formato de cada fator — `feature`, `valor`, `contribuicao`, `direcao` —
        é o que `app.py` renderiza na tabela; alterá-lo muda a interface.
        """
        valores = np.asarray(self.explicador_shap.shap_values(matriz), dtype=float)
        if valores.ndim == 3:  # (linhas, features, classes) — usa a classe positiva
            valores = valores[..., -1]
        contribuicoes = valores[0]

        nomes = self.nomes_features or [f"feature_{i}" for i in range(len(contribuicoes))]
        ordem = np.argsort(np.abs(contribuicoes))[::-1][:TOP_FATORES]

        valores_entrada = np.asarray(matriz)[0]
        return [
            {
                "feature": nomes[int(indice)],
                "valor": float(valores_entrada[int(indice)]),
                "contribuicao": float(contribuicoes[int(indice)]),
                "direcao": "aumenta" if contribuicoes[int(indice)] > 0 else "reduz",
            }
            for indice in ordem
        ]


def _como_quadro(transacao: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """Aceita dicionário ou DataFrame de uma linha; recusa o resto."""
    if isinstance(transacao, pd.DataFrame):
        if len(transacao) != 1:
            raise ValueError("processe uma transação por vez")
        return transacao
    if isinstance(transacao, dict):
        if not transacao:
            raise ValueError("a transação não pode ser vazia")
        return pd.DataFrame([transacao])
    raise TypeError("a transação deve ser um dicionário ou um DataFrame de uma linha")


def construir_pipeline(
    diretorio_modelo: Path | str = DIRETORIO_MODELO_PADRAO,
    matriz_fundo: Any | None = None,
    recuperador: RecuperadorDocumentos | None = None,
    cliente_llm: Callable[[str], str] | None = None,
    limiar: float = LIMIAR_PADRAO,
) -> Pipeline:
    """
    Carrega os componentes do disco e monta o pipeline.

    `matriz_fundo` são as linhas de referência do SHAP, amostradas **somente do
    treino**, conforme o protocolo em `reports/pessoa_2/julho/04_metodologia_shap.md`.
    Sem ela, cai-se no modo `tree_path_dependent`, que é outra configuração
    metodológica e produz contribuições em outra escala — os dois resultados não
    são comparáveis entre si.
    """
    import shap

    from src.models.shap_xgboost import nomes_das_features

    preprocessador, modelo, manifesto = carregar(Path(diretorio_modelo))
    logger.info("modelo carregado: %s", manifesto.get("tipo_do_modelo"))

    if matriz_fundo is not None:
        masker = shap.maskers.Independent(matriz_fundo, max_samples=len(matriz_fundo))
        explicador_shap = shap.TreeExplainer(
            modelo,
            data=masker,
            feature_perturbation="interventional",
            model_output="probability",
        )
    else:
        logger.warning(
            "sem matriz de fundo: usando tree_path_dependent, que não é a "
            "configuração registrada na metodologia"
        )
        explicador_shap = shap.TreeExplainer(modelo, feature_perturbation="tree_path_dependent")

    n_colunas = int(getattr(modelo, "n_features_in_", 0)) or None
    nomes = nomes_das_features(preprocessador, n_colunas) if n_colunas else None

    return Pipeline(
        preprocessador=preprocessador,
        modelo=modelo,
        explicador_shap=explicador_shap,
        recuperador=recuperador,
        nomes_features=nomes,
        limiar=limiar,
        cliente_llm=cliente_llm,
    )


def explicar_transacao(transacao: dict[str, Any]) -> dict[str, Any]:
    """
    Ponto de entrada usado pela interface Streamlit.

    `app.py` carrega esta função por `TCC_PIPELINE_MODULE=src.pipeline` e
    `TCC_PIPELINE_FUNCTION=explicar_transacao`. Constrói o pipeline a cada
    chamada, o que é caro: para uso repetido, construa uma vez com
    `construir_pipeline` e reutilize `Pipeline.processar`.

    {{PREENCHER}}: a matriz de fundo do SHAP e o índice vetorial precisam ser
    resolvidos aqui quando os caminhos de produção estiverem definidos. Enquanto
    isso, esta função exige que ambos venham prontos por `construir_pipeline`.
    """
    raise NotImplementedError(
        "{{PREENCHER}}: defina a origem da matriz de fundo do SHAP e do índice "
        "vetorial para uso pela interface. Use construir_pipeline(...) e "
        "Pipeline.processar(...) diretamente enquanto isso não estiver resolvido."
    )
