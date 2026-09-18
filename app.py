"""
FraudGuard PIX — interface de demonstração do protótipo.

Consome o pipeline real do projeto (`src.pipeline`) e apenas apresenta o que ele
devolve. Nenhuma etapa de ML, SHAP ou RAG é executada aqui.

## Por que não há formulário de digitação

O modelo recebe 439 colunas por transação. Um formulário com meia dúzia de
campos, completando as outras 433 com valores inventados, produziria uma
probabilidade que não corresponde a transação alguma — e uma tela de
explicabilidade que fabrica sua própria entrada não explica nada.

A demonstração usa **transações reais do conjunto de teste**, exportadas por
`scripts/exportar_exemplos_interface.py` e estratificadas por quadrante da
matriz de confusão. Incluir falsos positivos é deliberado: no teste, cerca de 8
em cada 10 transações marcadas não eram fraude, e esconder isso venderia um
sistema que não existe.

A entrada por JSON permanece disponível para quem tenha uma transação completa.

## Contrato com o pipeline

`validar_resultado_pipeline` documenta e verifica o formato esperado. Ele é
importado pelos testes do pipeline, que executam esta validação de verdade em
vez de reafirmar a suposição de quem escreveu o produtor dos dados.
"""

from __future__ import annotations

import importlib
import json
import logging
import os
import time
from collections.abc import Callable, Mapping
from numbers import Real
from pathlib import Path
from typing import Any

import streamlit as st

from src.ui import components as ui

LOGGER = logging.getLogger(__name__)
MAX_TRANSACTION_JSON_CHARS = 100_000
RAIZ = Path(__file__).resolve().parent
ARQUIVO_EXEMPLOS = RAIZ / "data" / "exemplos_interface.json"

PIPELINE_MODULE = os.getenv("TCC_PIPELINE_MODULE", "src.pipeline").strip()
PIPELINE_FUNCTION = os.getenv("TCC_PIPELINE_FUNCTION", "explicar_transacao").strip()

PAGINAS = [
    "Visão Geral",
    "Analisar Transação",
    "Explicação da IA",
    "Evidências RAG",
    "Sobre o Projeto",
]

ROTULO_QUADRANTE = {
    "verdadeiro_positivo": "fraude que o modelo pegou",
    "falso_positivo": "alarme falso",
    "falso_negativo": "fraude que passou",
    "verdadeiro_negativo": "legítima, não marcada",
}


class IntegracaoPendenteError(RuntimeError):
    """Indica que o ponto de integração ainda precisa ser configurado."""


# ─── Contrato com o pipeline ────────────────────────────────────────────────

def ler_transacao_json(conteudo: str) -> dict[str, Any]:
    """Lê a transação enviada como JSON, recusando entrada implausível."""
    if not conteudo or not conteudo.strip():
        raise ValueError("Informe os dados da transação em formato JSON.")
    if len(conteudo) > MAX_TRANSACTION_JSON_CHARS:
        raise ValueError(
            f"Conteúdo acima do limite de {MAX_TRANSACTION_JSON_CHARS} caracteres."
        )
    try:
        transacao = json.loads(conteudo)
    except json.JSONDecodeError as erro:
        raise ValueError(f"JSON inválido: {erro.msg} (linha {erro.lineno}).") from erro
    if not isinstance(transacao, dict):
        raise ValueError("A entrada deve ser um objeto JSON com uma única transação.")
    if not transacao:
        raise ValueError("A transação não pode ser vazia.")
    return transacao


@st.cache_resource(show_spinner=False)
def carregar_pipeline(nome_modulo: str, nome_funcao: str) -> Callable[[dict[str, Any]], Any]:
    """
    Carrega o orquestrador existente sem duplicar suas regras na interface.

    Em cache porque carregar modelo, explicador SHAP e índice vetorial custa
    segundos: sem isso, cada clique na tela pagaria esse custo de novo.
    """
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
                {"titulo": str, "trecho": str, "fonte": str, "score": float}
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

    documentos = resultado.get("documentos_recuperados") or []
    if not isinstance(documentos, list) or not all(
        isinstance(documento, Mapping) for documento in documentos
    ):
        raise ValueError("O campo 'documentos_recuperados' deve ser uma lista de objetos.")

    normalizado = dict(resultado)
    normalizado["predicao"] = dict(predicao)
    normalizado["fatores_shap"] = [dict(fator) for fator in fatores_shap]
    normalizado["explicacao_rag"] = explicacao.strip()
    normalizado["documentos_recuperados"] = [dict(documento) for documento in documentos]
    return normalizado


def _formatar_proporcao(valor: Any) -> str:
    if isinstance(valor, Real) and not isinstance(valor, bool):
        numero = float(valor)
        return f"{numero:.2%}" if 0.0 <= numero <= 1.0 else f"{numero:.6g}"
    return str(valor) if valor is not None else "não informado"


# ─── Dados de demonstração ──────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def carregar_exemplos() -> dict[str, Any] | None:
    if not ARQUIVO_EXEMPLOS.exists():
        return None
    try:
        return json.loads(ARQUIVO_EXEMPLOS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Falha ao ler os exemplos da interface.")
        return None


def _executar(transacao: dict[str, Any]) -> dict[str, Any] | None:
    """
    Executa o pipeline mostrando as etapas reais.

    As etapas listadas correspondem ao que o pipeline faz de fato. Uma barra de
    progresso que anuncia etapas inexistentes é encenação, não informação.
    """
    etapas = [
        "Validando dados",
        "Executando modelo",
        "Calculando SHAP",
        "Consultando RAG",
        "Gerando explicação",
    ]
    barra = st.progress(0.0, text=etapas[0])
    inicio = time.perf_counter()
    try:
        funcao = carregar_pipeline(PIPELINE_MODULE, PIPELINE_FUNCTION)
        # O pipeline roda as etapas em sequência internamente; a barra avança
        # até a última antes da chamada porque não há callback de progresso.
        for posicao, etapa in enumerate(etapas[1:], start=1):
            barra.progress(posicao / len(etapas), text=etapa)
        resultado = validar_resultado_pipeline(funcao(transacao))
    except (ValueError, IntegracaoPendenteError) as erro:
        barra.empty()
        st.error(str(erro))
        return None
    except NotImplementedError as erro:
        barra.empty()
        st.error(
            "A integração do pipeline ainda não está completa. "
            "Detalhe técnico registrado nos logs."
        )
        LOGGER.warning("integração pendente: %s", erro)
        return None
    except FileNotFoundError:
        barra.empty()
        st.error(
            "Artefato do modelo não encontrado. Gere o modelo com "
            "`scripts/tunar_xgboost.py` antes de executar a análise."
        )
        return None
    except Exception:
        barra.empty()
        LOGGER.exception("Falha inesperada durante a execução do pipeline.")
        st.error(
            "Não foi possível concluir a análise. Verifique os artefatos do "
            "modelo, o índice RAG e a configuração do LLM."
        )
        return None

    barra.progress(1.0, text="Concluído")
    barra.empty()
    st.session_state["resultado"] = resultado
    st.session_state["duracao"] = time.perf_counter() - inicio
    return resultado


# ─── Páginas ────────────────────────────────────────────────────────────────

def pagina_visao_geral(exemplos: dict[str, Any] | None) -> None:
    ui.abrir_card("Visão geral", "O que este protótipo demonstra")
    st.markdown(
        "Este sistema classifica uma transação, identifica os fatores que mais "
        "influenciaram a decisão do modelo e recupera documentos regulatórios "
        "para compor uma explicação em linguagem natural."
    )
    ui.fechar_card()

    colunas = st.columns(3)
    colunas[0].metric("Modelo principal", "XGBoost")
    colunas[1].metric("Explicabilidade", "SHAP")
    colunas[2].metric("Contexto documental", "RAG + LLM")

    ui.abrir_card("Desempenho medido", "Conjunto de validação — 88.581 transações")
    st.markdown(
        "| Modelo | AUC-PR |\n|---|---|\n"
        "| Regressão logística (referência) | 0,3930 |\n"
        "| Random Forest (comparativo) | 0,5298 |\n"
        "| **XGBoost ajustado (principal)** | **0,5703** |"
    )
    st.caption(
        "A AUC-PR de um classificador aleatório equivale à proporção de fraude "
        "do conjunto (3,43%). No teste, o desempenho cai para cerca de metade — "
        "os padrões aprendidos envelhecem."
    )
    ui.fechar_card()

    if exemplos:
        ui.abrir_card("Dados da demonstração")
        st.write(
            f"{len(exemplos['exemplos'])} transações reais do conjunto de teste, "
            f"com {exemplos['n_colunas']} colunas cada."
        )
        ui.fechar_card()


def pagina_analisar(exemplos: dict[str, Any] | None) -> None:
    ui.abrir_card(
        "Nova análise PIX",
        "Selecione uma transação real do conjunto de teste ou informe uma em JSON.",
    )

    if exemplos is None:
        ui.fechar_card()
        ui.aviso(
            "<b>Exemplos indisponíveis.</b> Gere-os com "
            "<code>python scripts/exportar_exemplos_interface.py</code>. "
            "Enquanto isso, use a entrada por JSON abaixo."
        )
        transacao = _entrada_json()
    else:
        opcoes = {
            f"#{i + 1} — probabilidade {e['probabilidade_registrada']:.1%} "
            f"({ROTULO_QUADRANTE.get(e['quadrante'], e['quadrante'])})": i
            for i, e in enumerate(exemplos["exemplos"])
        }
        escolha = st.selectbox("Transação", list(opcoes), key="escolha_exemplo")
        exemplo = exemplos["exemplos"][opcoes[escolha]]

        st.caption("Campos legíveis desta transação (os demais não têm significado publicado):")
        resumo = {k: v for k, v in exemplo["resumo"].items() if v is not None}
        colunas = st.columns(3)
        for posicao, (campo, valor) in enumerate(resumo.items()):
            texto = f"{valor:.4g}" if isinstance(valor, float) else str(valor)
            colunas[posicao % 3].text_input(campo, value=texto, disabled=True,
                                            key=f"campo_{campo}")

        transacao = exemplo["transacao"]
        with st.expander("Informar outra transação em JSON"):
            alternativa = _entrada_json(chave="json_alternativo")
            if alternativa:
                transacao = alternativa

    ui.fechar_card()

    if st.button("🔍 Analisar transação", type="primary", use_container_width=True):
        if transacao:
            _executar(transacao)
        else:
            st.error("Nenhuma transação selecionada ou informada.")

    resultado = st.session_state.get("resultado")
    if resultado:
        _mostrar_resultado(resultado)


def _entrada_json(chave: str = "json_principal") -> dict[str, Any] | None:
    conteudo = st.text_area(
        "Dados da transação (JSON)",
        height=160,
        placeholder='{"TransactionAmt": 372.5, "ProductCD": "W", ...}',
        help="Objeto JSON com as colunas da transação.",
        key=chave,
    )
    if not conteudo or not conteudo.strip():
        return None
    try:
        return ler_transacao_json(conteudo)
    except ValueError as erro:
        st.error(str(erro))
        return None


def _mostrar_resultado(resultado: Mapping[str, Any]) -> None:
    predicao = resultado["predicao"]
    classe = str(predicao.get("classe", "")).lower()
    suspeita = "suspeita" in classe and "não" not in classe

    st.markdown("### Resultado da análise")
    esquerda, direita = st.columns([1.15, 1])
    with esquerda:
        ui.veredito(suspeita)
    with direita:
        ui.abrir_card("Probabilidade do modelo")
        ui.barra_probabilidade(predicao.get("probabilidade"), predicao.get("limiar"))
        ui.fechar_card()

    ui.abrir_card("Por que o modelo tomou essa decisão?",
                  "Fatores de maior contribuição, segundo o SHAP")
    ui.fatores_shap(resultado["fatores_shap"])
    ui.fechar_card()

    ui.abrir_card("✦ Explicação da IA", "Gerada a partir dos fatores e dos documentos recuperados")
    ui.explicacao(resultado["explicacao_rag"])
    ui.fechar_card()

    documentos = resultado.get("documentos_recuperados", [])
    st.markdown(f"### Evidências consultadas ({len(documentos)} documentos)")
    ui.documentos(documentos)

    duracao = st.session_state.get("duracao")
    ui.resumo_tecnico({
        "Predição": predicao.get("classe", "não informado"),
        "Probabilidade": _formatar_proporcao(predicao.get("probabilidade")),
        "Limiar de decisão": _formatar_proporcao(predicao.get("limiar")),
        "Fatores retornados": len(resultado["fatores_shap"]),
        "Documentos recuperados": len(documentos),
        "Tempo de processamento": f"{duracao:.2f} s" if duracao else None,
    })


def pagina_explicacao() -> None:
    resultado = st.session_state.get("resultado")
    if not resultado:
        ui.aviso("Nenhuma análise executada ainda. Vá para <b>Analisar Transação</b>.")
        return
    ui.abrir_card("✦ Explicação da IA")
    ui.explicacao(resultado["explicacao_rag"])
    ui.fechar_card()
    ui.abrir_card("Fatores que sustentam a explicação")
    ui.fatores_shap(resultado["fatores_shap"])
    ui.fechar_card()


def pagina_evidencias() -> None:
    resultado = st.session_state.get("resultado")
    if not resultado:
        ui.aviso("Nenhuma análise executada ainda. Vá para <b>Analisar Transação</b>.")
        return
    documentos = resultado.get("documentos_recuperados", [])
    ui.abrir_card("Evidências consultadas",
                  f"{len(documentos)} documentos recuperados pela busca vetorial")
    ui.documentos(documentos)
    ui.fechar_card()


def pagina_sobre() -> None:
    ui.abrir_card("Como o sistema funciona", "Quatro etapas, em sequência")
    st.markdown(
        "**1. Machine Learning — o que decide.** Um modelo XGBoost estima a "
        "probabilidade de a transação ser fraudulenta, a partir do histórico de "
        "transações anteriores.\n\n"
        "**2. SHAP — quais variáveis pesaram.** Para cada transação, calcula "
        "quanto cada variável contribuiu para aquela probabilidade. Isso indica "
        "**influência sobre a decisão do modelo**, não causa da fraude.\n\n"
        "**3. RAG — o contexto documental.** A partir dos fatores identificados, "
        "o sistema busca trechos relevantes em normas do Banco Central e "
        "relatórios do setor. É o que torna a explicação verificável.\n\n"
        "**4. LLM — a explicação em português.** Um modelo de linguagem combina "
        "os fatores e os documentos em um texto legível, instruído a não "
        "afirmar nada fora das evidências fornecidas."
    )
    ui.fechar_card()

    ui.abrir_card("O que este protótipo não faz")
    st.markdown(
        "- não usa dados reais do Pix: os experimentos usam o IEEE-CIS, de transações com cartão;\n"
        "- não prova fraude nem identifica responsáveis;\n"
        "- não deve produzir bloqueio ou qualquer decisão automática;\n"
        "- não atribui significado às colunas anônimas do dataset."
    )
    ui.fechar_card()


# ─── Aplicação ──────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="FraudGuard PIX — Detecção e Explicação de Fraudes",
                       page_icon="🛡", layout="wide")
    ui.aplicar_estilo()

    try:
        carregar_pipeline(PIPELINE_MODULE, PIPELINE_FUNCTION)
        integrado, detalhe = True, "Sistema operacional"
    except IntegracaoPendenteError:
        integrado, detalhe = False, "Integração pendente"

    ui.cabecalho(integrado, detalhe)
    exemplos = carregar_exemplos()

    with st.sidebar:
        st.markdown("#### Navegação")
        pagina = st.radio("Seções", PAGINAS, index=1, label_visibility="collapsed")
        st.divider()
        st.markdown("#### Integração")
        if integrado:
            st.success(f"Pipeline: `{PIPELINE_MODULE}.{PIPELINE_FUNCTION}`")
        else:
            st.warning(f"Pipeline não carregado: `{PIPELINE_MODULE}.{PIPELINE_FUNCTION}`")
        st.caption("Fluxo: Transação → ML → SHAP → RAG → explicação")

    if pagina == "Visão Geral":
        pagina_visao_geral(exemplos)
    elif pagina == "Analisar Transação":
        pagina_analisar(exemplos)
    elif pagina == "Explicação da IA":
        pagina_explicacao()
    elif pagina == "Evidências RAG":
        pagina_evidencias()
    else:
        pagina_sobre()

    ui.rodape()


if __name__ == "__main__":
    main()
