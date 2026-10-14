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

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.persistencia import carregar
from src.rag.explainer import (
    ErroLLM,
    criar_cliente_llm_de_ambiente,
    explicar,
    montar_prompt,
)
from src.rag.retriever import RecuperadorDocumentos, consulta_a_partir_dos_fatores

logger = logging.getLogger(__name__)

RAIZ = Path(__file__).resolve().parent.parent
DIRETORIO_MODELO_PADRAO = RAIZ / "models" / "xgboost"
ARQUIVO_POLITICA_DECISAO = "politica_decisao.json"
LIMIAR_PADRAO = 0.5
FEATURES_EXPLICAVEIS = (
    "valor_atipico_proxy",
    "frequencia_recente_proxy",
    "dispositivo_raro_proxy",
    "posicao_ciclo_diario_relativa",
)
TOP_FATORES = len(FEATURES_EXPLICAVEIS)
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

    def __post_init__(self) -> None:
        if (
            isinstance(self.limiar, bool)
            or not isinstance(self.limiar, Real)
            or not math.isfinite(float(self.limiar))
            or not 0.0 <= float(self.limiar) <= 1.0
        ):
            raise ValueError("o limiar deve ser um número finito entre 0 e 1")
        self.limiar = float(self.limiar)

    def processar(self, transacao: dict[str, Any] | pd.DataFrame) -> dict[str, Any]:
        """Executa as três camadas para uma transação e devolve o resultado."""
        quadro = _alinhar_ao_esquema_de_treino(_como_quadro(transacao), self.preprocessador)
        matriz = self.preprocessador.transform(quadro)

        probabilidades = np.asarray(self.modelo.predict_proba(matriz), dtype=float)
        if probabilidades.ndim != 2 or probabilidades.shape[0] != 1 or probabilidades.shape[1] < 2:
            raise ValueError("predict_proba deve devolver uma linha com duas classes")
        probabilidade = float(probabilidades[0, 1])
        if not math.isfinite(probabilidade) or not 0.0 <= probabilidade <= 1.0:
            raise ValueError("o modelo devolveu uma probabilidade inválida")
        sinalizada = probabilidade >= self.limiar

        fatores = self._fatores_shap(matriz)

        documentos: list[Any] = []
        if self.recuperador is not None:
            consulta = consulta_a_partir_dos_fatores(fatores, sinalizada)
            documentos = self.recuperador.recuperar(consulta, top_k=TOP_DOCUMENTOS)

        explicacao_disponivel = self.cliente_llm is not None
        prompt_montado: str | None = None
        erro_llm: str | None = None
        if self.cliente_llm is None:
            prompt_montado = montar_prompt(
                fatores, documentos, probabilidade, sinalizada
            )
            explicacao = _mensagem_sem_llm()
        else:
            try:
                explicacao = explicar(
                    fatores_shap=fatores,
                    documentos=documentos,
                    probabilidade=probabilidade,
                    sinalizada=sinalizada,
                    cliente=self.cliente_llm,
                )
            except ErroLLM as erro:
                logger.warning("explicação por LLM indisponível: %s", erro)
                explicacao_disponivel = False
                erro_llm = str(erro)
                prompt_montado = montar_prompt(
                    fatores, documentos, probabilidade, sinalizada
                )
                explicacao = _mensagem_erro_llm()

        resultado = {
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
            "explicacao_disponivel": explicacao_disponivel,
            "rag_disponivel": self.recuperador is not None,
        }
        if prompt_montado is not None:
            resultado["prompt_montado"] = prompt_montado
        if erro_llm is not None:
            resultado["erro_llm"] = erro_llm
        return resultado

    def _fatores_shap(self, matriz: Any) -> list[dict[str, Any]]:
        """
        Contribuições das quatro proxies explicáveis criadas no projeto.

        O classificador continua usando a matriz completa. Para a camada de
        explicação, porém, são selecionadas somente as features com semântica
        documentada pela dupla. Isso impede que o dashboard ou o RAG atribuam
        significado inventado a C1-C14, V1-V339, card1 e outros campos
        mascarados do IEEE-CIS.

        O formato de cada fator — `feature`, `valor`, `contribuicao`, `direcao` —
        é o que `app.py` renderiza na tabela; alterá-lo muda a interface.
        """
        valores_brutos = self.explicador_shap.shap_values(matriz)
        if isinstance(valores_brutos, (list, tuple)):
            valores = np.asarray(valores_brutos[-1], dtype=float)
        else:
            valores = np.asarray(valores_brutos, dtype=float)

        valores_entrada = _primeira_linha_densa(matriz)
        if valores.ndim == 1:
            contribuicoes = valores
        elif valores.ndim == 2 and valores.shape[0] == 1:
            contribuicoes = valores[0]
        elif valores.ndim == 3 and valores.shape[0] == 1:
            # SHAP recente: (linhas, features, classes).
            contribuicoes = valores[0, :, -1]
        elif valores.ndim == 3 and valores.shape[1] == 1:
            # Formato legado: (classes, linhas, features).
            contribuicoes = valores[-1, 0, :]
        else:
            raise ValueError(f"formato de saída SHAP não suportado: {valores.shape}")

        contribuicoes = np.asarray(contribuicoes, dtype=float).reshape(-1)
        if len(contribuicoes) != len(valores_entrada):
            raise ValueError("o SHAP devolveu quantidade de features incompatível")
        if not np.isfinite(contribuicoes).all():
            raise ValueError("o SHAP devolveu contribuição não finita")

        nomes = self.nomes_features or [f"feature_{i}" for i in range(len(contribuicoes))]
        if len(nomes) != len(contribuicoes):
            raise ValueError("a lista de nomes não corresponde às features do SHAP")

        indices_por_nome = {
            str(nome).rsplit("__", maxsplit=1)[-1]: indice
            for indice, nome in enumerate(nomes)
        }
        ausentes = [
            nome for nome in FEATURES_EXPLICAVEIS if nome not in indices_por_nome
        ]
        if ausentes:
            raise ValueError(
                "o artefato não contém as features explicáveis do projeto: "
                + ", ".join(ausentes)
            )
        ordem = sorted(
            (indices_por_nome[nome] for nome in FEATURES_EXPLICAVEIS),
            key=lambda indice: abs(contribuicoes[indice]),
            reverse=True,
        )

        return [
            {
                "feature": nomes[int(indice)],
                "valor": float(valores_entrada[int(indice)]),
                "contribuicao": float(contribuicoes[int(indice)]),
                "direcao": (
                    "aumenta"
                    if contribuicoes[int(indice)] > 0
                    else "reduz"
                    if contribuicoes[int(indice)] < 0
                    else "neutro"
                ),
            }
            for indice in ordem
        ]


GRUPOS_NUMERICOS = ("numericas",)
GRUPOS_CATEGORICOS = ("categoricas_baixa", "categoricas_alta")


def _alinhar_ao_esquema_de_treino(quadro: pd.DataFrame, preprocessador: Any) -> pd.DataFrame:
    """
    Converte a transação para os tipos com que o modelo foi treinado e avaliado.

    O treino e a avaliação usam `reduzir_precisao`, que deixa as colunas
    numéricas em `float32`. Uma transação reconstruída de JSON chega em
    `float64`, e as colunas com ausentes chegam como `object` com `None`. O
    XGBoost compara cada valor com cortes aprendidos em `float32`; em precisão
    diferente, valores próximos do corte mudam de ramo, e a probabilidade
    divergia muito da registrada (0,6059 virava 0,0872 num exemplo do teste).
    Com os tipos do treino, a probabilidade reproduz a avaliação.

    Sem os grupos de colunas esperados (por exemplo, um dublê de teste), o
    quadro é devolvido sem alteração.
    """
    grupos = getattr(preprocessador, "transformers_", None)
    if not isinstance(grupos, list):
        return quadro
    alinhado = quadro.copy()
    for nome, _, colunas in grupos:
        presentes = [c for c in colunas if c in alinhado.columns] if isinstance(colunas, list) else []
        if not presentes:
            continue
        if nome in GRUPOS_NUMERICOS:
            alinhado[presentes] = (
                alinhado[presentes].apply(pd.to_numeric, errors="coerce").astype("float32")
            )
        elif nome in GRUPOS_CATEGORICOS:
            for coluna in presentes:
                serie = alinhado[coluna].astype(object)
                alinhado[coluna] = serie.where(serie.notna(), np.nan)
    return alinhado


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


def _primeira_linha_densa(matriz: Any) -> np.ndarray:
    """Extrai uma linha numérica tanto de matrizes densas quanto esparsas."""
    linha = matriz[0]
    if hasattr(linha, "toarray"):
        linha = linha.toarray()
    valores = np.asarray(linha, dtype=float).reshape(-1)
    if not np.isfinite(valores).all():
        raise ValueError("o pré-processador devolveu valores não finitos")
    return valores


def carregar_limiar_politica(
    diretorio_modelo: Path | str,
    manifesto_modelo: dict[str, Any],
) -> float:
    """Lê a política versionada e verifica se ela pertence ao modelo carregado."""
    caminho = Path(diretorio_modelo) / ARQUIVO_POLITICA_DECISAO
    if not caminho.exists():
        logger.warning("política de decisão ausente; usando limiar padrão %.3f", LIMIAR_PADRAO)
        return LIMIAR_PADRAO
    try:
        politica = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as erro:
        raise ValueError("não foi possível ler a política de decisão") from erro
    if not isinstance(politica, dict):
        raise TypeError("política de decisão inválida")

    hashes_esperados = politica.get("hashes_modelo")
    hashes_atuais = manifesto_modelo.get("hashes")
    if hashes_esperados != hashes_atuais:
        raise ValueError("a política de decisão não corresponde ao modelo carregado")

    limiar = politica.get("limiar")
    if (
        isinstance(limiar, bool)
        or not isinstance(limiar, Real)
        or not math.isfinite(float(limiar))
        or not 0.0 <= float(limiar) <= 1.0
    ):
        raise ValueError("o limiar da política de decisão é inválido")
    return float(limiar)


def construir_pipeline(
    diretorio_modelo: Path | str = DIRETORIO_MODELO_PADRAO,
    matriz_fundo: Any | None = None,
    recuperador: RecuperadorDocumentos | None = None,
    cliente_llm: Callable[[str], str] | None = None,
    limiar: float | None = None,
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
    limiar_efetivo = (
        carregar_limiar_politica(diretorio_modelo, manifesto)
        if limiar is None
        else limiar
    )

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
        limiar=limiar_efetivo,
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


def _mensagem_erro_llm() -> str:
    return (
        "Explicação em linguagem natural temporariamente indisponível: o modelo "
        "de linguagem não concluiu a solicitação. A predição, os fatores SHAP e "
        "os documentos recuperados continuam disponíveis para inspeção."
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

    if cliente_llm is None:
        cliente_llm = criar_cliente_llm_de_ambiente()

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

    # O modelo é carregado ANTES do recuperador. No macOS, PyTorch, FAISS e
    # XGBoost trazem cópias distintas do OpenMP; carregar o XGBoost depois que o
    # modelo de embeddings iniciou as threads do OpenMP derruba o processo
    # (segmentation fault). A ordem inversa foi verificada em máquina limpa.
    pipeline = construir_pipeline(
        diretorio_modelo=diretorio_modelo,
        matriz_fundo=fundo,
        recuperador=None,
        cliente_llm=cliente_llm,
    )
    try:
        pipeline.recuperador = RecuperadorDocumentos.a_partir_do_disco()
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as erro:
        logger.warning("recuperação documental indisponível: %s", erro)

    _PIPELINE_EM_CACHE = pipeline
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

    resultado = pipeline.processar(transacao)
    resultado["shap_interventional"] = ARQUIVO_FUNDO_SHAP.exists()
    return resultado
