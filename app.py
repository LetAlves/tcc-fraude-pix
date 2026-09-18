"""Interface Streamlit do protótipo de explicação de transações suspeitas.

Este arquivo é somente a camada de apresentação. O pipeline integrado deve ser
informado por ``TCC_PIPELINE_MODULE`` e ``TCC_PIPELINE_FUNCTION`` e expor uma
função que receba uma transação como ``dict`` e devolva o contrato documentado
em ``validar_resultado_pipeline``.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from numbers import Real
from typing import Any

import streamlit as st


LOGGER = logging.getLogger(__name__)
MAX_TRANSACTION_JSON_CHARS = 100_000

# {{PREENCHER}}: definir o módulo e a função do orquestrador quando a integração
# ML -> SHAP -> RAG -> LLM for consolidada no projeto. As variáveis de ambiente
# permitem configurar a interface sem inserir credenciais ou caminhos no código.
PIPELINE_MODULE = os.getenv("TCC_PIPELINE_MODULE", "{{PREENCHER}}").strip()
PIPELINE_FUNCTION = os.getenv("TCC_PIPELINE_FUNCTION", "{{PREENCHER}}").strip()


class IntegracaoPendenteError(RuntimeError):
    """Indica que o ponto de integração ainda precisa ser configurado."""


def ler_transacao_json(conteudo: str) -> dict[str, Any]:
    """Valida a entrada textual e devolve uma única transação."""

    texto = conteudo.strip()
    if not texto:
        raise ValueError("Informe os dados da transação em formato JSON.")
    if len(texto) > MAX_TRANSACTION_JSON_CHARS:
        raise ValueError(
            f"A entrada excede o limite de {MAX_TRANSACTION_JSON_CHARS:,} caracteres."
        )

    try:
        transacao = json.loads(texto)
    except json.JSONDecodeError as erro:
        raise ValueError(
            f"JSON inválido na linha {erro.lineno}, coluna {erro.colno}."
        ) from erro

    if not isinstance(transacao, dict):
        raise ValueError("A entrada deve ser um objeto JSON com uma única transação.")
    if not transacao:
        raise ValueError("A transação não pode ser vazia.")
    return transacao


@st.cache_resource(show_spinner=False)
def carregar_pipeline(nome_modulo: str, nome_funcao: str) -> Callable[[dict[str, Any]], Any]:
    """Carrega o orquestrador existente sem duplicar suas regras no Streamlit."""

    if (
        not nome_modulo
        or not nome_funcao
        or "{{PREENCHER}}" in nome_modulo
        or "{{PREENCHER}}" in nome_funcao
    ):
        raise IntegracaoPendenteError(
            "{{PREENCHER}}: configure TCC_PIPELINE_MODULE e "
            "TCC_PIPELINE_FUNCTION com o orquestrador existente."
        )

    try:
        modulo = importlib.import_module(nome_modulo)
    except (ImportError, ModuleNotFoundError) as erro:
        raise IntegracaoPendenteError(
            f"Não foi possível importar o módulo configurado: {nome_modulo}."
        ) from erro

    funcao = getattr(modulo, nome_funcao, None)
    if not callable(funcao):
        raise IntegracaoPendenteError(
            f"A função configurada não foi encontrada: {nome_modulo}.{nome_funcao}."
        )
    return funcao


def validar_resultado_pipeline(resultado: Any) -> dict[str, Any]:
    """Valida o contrato mínimo esperado da integração.

    Contrato de saída::

        {
            "predicao": {
                "classe": str | int,
                "probabilidade": float,  # opcional
                "limiar": float,          # opcional
            },
            "fatores_shap": [
                {
                    "feature": str,
                    "valor": object,
                    "contribuicao": float,
                    "direcao": str,       # opcional
                }
            ],
            "explicacao_rag": str,
            "documentos_recuperados": [  # opcional
                {
                    "titulo": str,
                    "trecho": str,
                    "fonte": str,
                    "score": float,
                }
            ],
        }
    """

    if not isinstance(resultado, Mapping):
        raise ValueError("O pipeline deve devolver um objeto com os resultados.")

    campos_obrigatorios = {"predicao", "fatores_shap", "explicacao_rag"}
    ausentes = sorted(campos_obrigatorios.difference(resultado))
    if ausentes:
        raise ValueError(
            "Resposta incompleta do pipeline; campos ausentes: " + ", ".join(ausentes)
        )

    predicao = resultado["predicao"]
    if not isinstance(predicao, Mapping) or "classe" not in predicao:
        raise ValueError("O campo 'predicao' deve conter ao menos a chave 'classe'.")

    fatores_shap = resultado["fatores_shap"]
    if not isinstance(fatores_shap, list) or not all(
        isinstance(fator, Mapping) for fator in fatores_shap
    ):
        raise ValueError("O campo 'fatores_shap' deve ser uma lista de objetos.")

    explicacao = resultado["explicacao_rag"]
    if not isinstance(explicacao, str) or not explicacao.strip():
        raise ValueError("O campo 'explicacao_rag' deve conter texto não vazio.")

    documentos = resultado.get("documentos_recuperados", [])
    if documentos is None:
        documentos = []
    if not isinstance(documentos, list) or not all(
        isinstance(documento, Mapping) for documento in documentos
    ):
        raise ValueError("O campo 'documentos_recuperados' deve ser uma lista de objetos.")

    normalizado = dict(resultado)
    normalizado["predicao"] = dict(predicao)
    normalizado["fatores_shap"] = [dict(fator) for fator in fatores_shap]
    normalizado["explicacao_rag"] = explicacao.strip()
    normalizado["documentos_recuperados"] = [
        dict(documento) for documento in documentos
    ]
    return normalizado


def _formatar_proporcao(valor: Any) -> str:
    if isinstance(valor, Real) and not isinstance(valor, bool):
        numero = float(valor)
        if 0.0 <= numero <= 1.0:
            return f"{numero:.2%}"
        return f"{numero:.6g}"
    return str(valor) if valor is not None else "Não informado"


def _mostrar_predicao(predicao: Mapping[str, Any]) -> None:
    colunas = st.columns(3)
    colunas[0].metric("Classificação do ML", str(predicao["classe"]))
    colunas[1].metric(
        "Probabilidade",
        _formatar_proporcao(predicao.get("probabilidade")),
    )
    colunas[2].metric("Limiar", _formatar_proporcao(predicao.get("limiar")))


def _mostrar_documentos(documentos: list[dict[str, Any]]) -> None:
    if not documentos:
        st.info("O pipeline não retornou documentos recuperados para este caso.")
        return

    for posicao, documento in enumerate(documentos, start=1):
        titulo = documento.get("titulo") or documento.get("title") or "Documento sem título"
        with st.expander(f"{posicao}. {titulo}"):
            if documento.get("score") is not None:
                st.caption(f"Score de recuperação: {documento['score']}")
            trecho = documento.get("trecho") or documento.get("page_content")
            if trecho:
                st.write(trecho)
            else:
                st.caption("Trecho não informado pelo pipeline.")
            fonte = documento.get("fonte") or documento.get("source_url")
            if fonte:
                st.write("Fonte:", fonte)


def main() -> None:
    st.set_page_config(
        page_title="Explicação de transação suspeita",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Explicação de transação suspeita")
    st.write(
        "Informe uma transação para visualizar, em sequência, a classificação do "
        "modelo, os principais fatores SHAP e a explicação apoiada por documentos."
    )
    st.caption(
        "Protótipo acadêmico: a saída não comprova fraude e não deve produzir "
        "bloqueio, acusação ou outra decisão automática."
    )

    with st.sidebar:
        st.subheader("Fluxo demonstrado")
        st.write("Transação → ML → SHAP → RAG → explicação")
        st.subheader("Integração")
        if "{{PREENCHER}}" in {PIPELINE_MODULE, PIPELINE_FUNCTION}:
            st.warning(
                "{{PREENCHER}}: informe o módulo e a função do pipeline integrado."
            )
        else:
            st.success(f"Pipeline configurado: {PIPELINE_MODULE}.{PIPELINE_FUNCTION}")

    with st.form("formulario_transacao"):
        conteudo = st.text_area(
            "Dados da transação (JSON)",
            height=260,
            placeholder='{"campo_1": "valor", "campo_2": 0}',
            help="Envie um objeto JSON correspondente a uma única transação.",
        )
        executar = st.form_submit_button(
            "Analisar transação",
            type="primary",
            use_container_width=True,
        )

    if not executar:
        return

    try:
        transacao = ler_transacao_json(conteudo)
        pipeline = carregar_pipeline(PIPELINE_MODULE, PIPELINE_FUNCTION)
        with st.spinner("Executando ML, SHAP e RAG..."):
            resultado = validar_resultado_pipeline(pipeline(transacao))
    except (ValueError, IntegracaoPendenteError) as erro:
        st.error(str(erro))
        return
    except Exception:
        LOGGER.exception("Falha inesperada durante a execução do pipeline.")
        st.error(
            "Não foi possível concluir a análise. Consulte os logs locais e "
            "verifique os artefatos do modelo, o índice RAG e a configuração do LLM."
        )
        return

    st.divider()
    st.header("Resultado do modelo")
    _mostrar_predicao(resultado["predicao"])

    st.header("Principais fatores SHAP")
    if resultado["fatores_shap"]:
        st.dataframe(
            resultado["fatores_shap"],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("O pipeline não retornou fatores SHAP para este caso.")
    st.caption(
        "Os fatores SHAP descrevem a influência no modelo; não demonstram "
        "causalidade nem intenção."
    )

    st.header("Explicação gerada pelo RAG")
    st.write(resultado["explicacao_rag"])

    st.header("Documentos recuperados")
    _mostrar_documentos(resultado["documentos_recuperados"])


if __name__ == "__main__":
    main()
