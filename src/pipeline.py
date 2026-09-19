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


ARQUIVO_FUNDO_SHAP = RAIZ / "data" / "fundo_shap.npy"
_PIPELINE_EM_CACHE: Pipeline | None = None


def _mensagem_sem_llm() -> str:
    """Texto usado quando não há cliente de LLM configurado.

    A interface exige `explicacao_rag` não vazio. Preencher com uma explicação
    inventada seria o pior desfecho possível num trabalho sobre explicabilidade,
    então o campo declara a ausência em vez de simulá-la, e o resultado traz
    `explicacao_disponivel=False` para a tela distinguir os dois casos.
    """
    return (
        "Explicação em linguagem natural indisponível: nenhum cliente de modelo "
        "de linguagem está configurado. Os fatores do SHAP e a probabilidade "
        "acima foram calculados normalmente e são resultados reais do modelo."
    )


def obter_pipeline(
    diretorio_modelo: Path | str = DIRETORIO_MODELO_PADRAO,
    cliente_llm: Callable[[str], str] | None = None,
) -> Pipeline:
    """
    Monta o pipeline com os artefatos disponíveis, reaproveitando entre chamadas.

    Cada componente é opcional de forma diferente:

    - **modelo**: obrigatório. Sem ele não há análise.
    - **matriz de fundo do SHAP**: exportada por
      `scripts/exportar_exemplos_interface.py`. Sem ela o explicador cai em
      `tree_path_dependent`, que é outra configuração metodológica — o pipeline
      avisa, mas não finge que é equivalente.
    - **índice vetorial**: sem ele não há recuperação documental, e a explicação
      fica apoiada apenas nos fatores do modelo.
    """
    global _PIPELINE_EM_CACHE
    if _PIPELINE_EM_CACHE is not None:
        return _PIPELINE_EM_CACHE

    fundo = None
    if ARQUIVO_FUNDO_SHAP.exists():
        fundo = np.load(ARQUIVO_FUNDO_SHAP)
        logger.info("matriz de fundo do SHAP carregada: %s", fundo.shape)
    else:
        logger.warning(
            "matriz de fundo ausente em %s; rode "
            "scripts/exportar_exemplos_interface.py para gerá-la",
            ARQUIVO_FUNDO_SHAP,
        )

    recuperador = None
    try:
        recuperador = RecuperadorDocumentos.a_partir_do_disco()
    except Exception as erro:  # índice ausente ou corrompido
        logger.warning("recuperação documental indisponível: %s", erro)

    _PIPELINE_EM_CACHE = construir_pipeline(
        diretorio_modelo=diretorio_modelo,
        matriz_fundo=fundo,
        recuperador=recuperador,
        cliente_llm=cliente_llm,
    )
    return _PIPELINE_EM_CACHE


def explicar_transacao(transacao: dict[str, Any]) -> dict[str, Any]:
    """
    Ponto de entrada usado pela interface Streamlit.

    `app.py` carrega esta função por `TCC_PIPELINE_MODULE=src.pipeline` e
    `TCC_PIPELINE_FUNCTION=explicar_transacao`.

    Devolve resultado parcial quando alguma camada não está disponível, sempre
    declarando o que faltou. Um resultado que omite a ausência é
    indistinguível de um resultado completo.
    """
    pipeline = obter_pipeline()

    sem_llm = pipeline.cliente_llm is None
    if sem_llm:
        # Cliente que devolve o próprio prompt: preserva o que seria enviado ao
        # modelo, sem inventar a resposta que ele daria.
        pipeline.cliente_llm = lambda prompt: prompt

    resultado = pipeline.processar(transacao)

    if sem_llm:
        resultado["prompt_montado"] = resultado["explicacao_rag"]
        resultado["explicacao_rag"] = _mensagem_sem_llm()

    resultado["explicacao_disponivel"] = not sem_llm
    resultado["rag_disponivel"] = pipeline.recuperador is not None
    resultado["shap_interventional"] = ARQUIVO_FUNDO_SHAP.exists()
    return resultado
