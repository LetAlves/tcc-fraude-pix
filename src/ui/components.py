"""
Componentes visuais da interface.

Cada função recebe dados já produzidos pelo pipeline e apenas os apresenta.
Nenhuma calcula, interpreta ou completa valor ausente: quando um campo não vem,
a interface diz que não veio. Uma tela que preenche lacunas com valor plausível
transforma ausência de resultado em resultado falso, que é o erro mais caro
possível num trabalho sobre explicabilidade.
"""

from __future__ import annotations

import html
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

import streamlit as st

from src.ui.styles import CSS

FEATURES_CONHECIDAS: dict[str, dict[str, str]] = {
    "valor_atipico_proxy": {
        "rotulo": "Valor fora do padrão",
        "tipo": "Entrada derivada do projeto",
        "descricao": (
            "Mede o quanto o valor da transação se afasta do histórico anterior "
            "disponível para comparação."
        ),
        "limitacao": (
            "O histórico utiliza um identificador anonimizado do experimento, não uma "
            "conta ou chave Pix."
        ),
    },
    "frequencia_recente_proxy": {
        "rotulo": "Frequência recente",
        "tipo": "Entrada derivada do projeto",
        "descricao": "Quantidade de transações anteriores em uma janela recente.",
        "limitacao": "A janela do dataset não representa necessariamente um dia civil Pix.",
    },
    "dispositivo_raro_proxy": {
        "rotulo": "Dispositivo pouco frequente",
        "tipo": "Entrada derivada do projeto",
        "descricao": "Indica baixa frequência histórica do dispositivo observado.",
        "limitacao": "Raridade estatística não comprova dispositivo novo ou comprometido.",
    },
    "posicao_ciclo_diario_relativa": {
        "rotulo": "Posição no ciclo diário",
        "tipo": "Entrada derivada do projeto",
        "descricao": "Posição relativa da transação dentro de um ciclo diário.",
        "limitacao": "TransactionDT não corresponde a um horário civil ou fuso conhecido.",
    },
}


def _nome_base_feature(nome: str) -> str:
    """Remove prefixos do transformador sem esconder o nome original da coluna."""
    return nome.rsplit("__", maxsplit=1)[-1]


def descrever_feature(nome: str) -> dict[str, str]:
    """Retorna metadados seguros para explicar uma feature na interface.

    O catálogo só interpreta variáveis cuja semântica está documentada no
    projeto. Para colunas mascaradas, informa categoria geral e limitação em
    vez de inferir uma história plausível a partir do nome.
    """
    tecnico = str(nome).strip() or "feature sem nome"
    base = _nome_base_feature(tecnico)

    if base in FEATURES_CONHECIDAS:
        return {"nome_tecnico": tecnico, **FEATURES_CONHECIDAS[base]}

    return {
        "nome_tecnico": tecnico,
        "rotulo": base,
        "tipo": "Variável técnica do modelo",
        "descricao": (
            "A interface não possui uma definição semântica validada para esta variável."
        ),
        "limitacao": (
            "Ela é apresentada pelo nome técnico para evitar uma interpretação inventada."
        ),
    }


def preparar_fatores_shap(
    fatores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normaliza fatores para apresentação e ignora contribuições inválidas."""
    preparados: list[dict[str, Any]] = []
    for fator in fatores:
        nome = str(fator.get("feature", "?"))
        if _nome_base_feature(nome) not in FEATURES_CONHECIDAS:
            continue
        contribuicao = fator.get("contribuicao", 0)
        if (
            isinstance(contribuicao, bool)
            or not isinstance(contribuicao, Real)
            or not math.isfinite(float(contribuicao))
        ):
            continue
        metadados = descrever_feature(nome)
        preparados.append(
            {
                **metadados,
                "valor": fator.get("valor"),
                "contribuicao": float(contribuicao),
            }
        )
    return preparados


def aplicar_estilo() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def cabecalho(operacional: bool, detalhe: str) -> None:
    ponto = "#6EE7A8" if operacional else "#FFD166"
    st.markdown(
        f"""
        <div class="fg-cabecalho">
          <div>
            <h1>Detecção e Explicação de Transações Suspeitas</h1>
            <p>Protótipo acadêmico — Machine Learning, SHAP e RAG</p>
          </div>
          <div class="fg-status">
            <span style="color:{ponto}">●</span> <b>{html.escape(detalhe)}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def marca_sidebar() -> None:
    st.markdown(
        """
        <div class="fg-marca-sidebar">
          <div class="fg-marca-icone">▥</div>
          <div>
            <strong>Análise de Transação</strong>
            <span>FraudGuard PIX</span>
          </div>
        </div>
        <p class="fg-sidebar-intro">
          Analise o risco, os fatores do modelo e as evidências documentais.
        </p>
        """,
        unsafe_allow_html=True,
    )


def abrir_card(titulo: str, subtitulo: str = "") -> None:
    sub = f'<div class="fg-sub">{html.escape(subtitulo)}</div>' if subtitulo else ""
    st.markdown(
        f'<div class="fg-card"><h3>{html.escape(titulo)}</h3>{sub}',
        unsafe_allow_html=True,
    )


def fechar_card() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def aviso(texto: str) -> None:
    st.markdown(f'<div class="fg-aviso">{texto}</div>', unsafe_allow_html=True)


def veredito(suspeita: bool) -> None:
    if suspeita:
        classe, titulo = "fg-suspeita", "⚠️ TRANSAÇÃO SUSPEITA"
        detalhe = "O modelo atribuiu probabilidade acima do limiar de decisão."
    else:
        classe, titulo = "fg-limpa", "✓ TRANSAÇÃO SEM INDÍCIOS DE FRAUDE"
        detalhe = "O modelo atribuiu probabilidade abaixo do limiar de decisão."
    st.markdown(
        f"""
        <div class="fg-veredito {classe}">
          <p class="fg-titulo">{titulo}</p>
          <p class="fg-detalhe">{detalhe}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metricas_resultado(
    suspeita: bool,
    probabilidade: float | None,
    limiar: float | None,
) -> None:
    """Apresenta decisão, probabilidade e limiar sem misturar os conceitos."""
    titulo = "SUSPEITA" if suspeita else "SEM INDÍCIOS"
    icone = "⚠" if suspeita else "✓"
    classe = "fg-metrica-alerta" if suspeita else "fg-metrica-segura"
    probabilidade_texto = (
        f"{float(probabilidade):.1%}" if isinstance(probabilidade, Real) else "não informado"
    )
    limiar_texto = f"{float(limiar):.2%}" if isinstance(limiar, Real) else "não informado"

    colunas = st.columns(3, gap="medium")
    colunas[0].markdown(
        f"""
        <div class="fg-metrica {classe}">
          <span class="fg-metrica-icone">{icone}</span>
          <div><small>Classificação</small><strong>{titulo}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    colunas[1].markdown(
        f"""
        <div class="fg-metrica">
          <span class="fg-metrica-icone fg-icone-azul">▥</span>
          <div><small>Risco estimado pelo modelo</small><strong>{probabilidade_texto}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    colunas[2].markdown(
        f"""
        <div class="fg-metrica">
          <span class="fg-metrica-icone fg-icone-azul">◇</span>
          <div><small>Limiar para emissão do alerta</small><strong>{limiar_texto}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def barra_probabilidade(probabilidade: float | None, limiar: float | None) -> None:
    """
    Probabilidade com o limiar marcado.

    O limiar aparece como um traço na barra porque 51% e 95% levam à mesma
    decisão e significam coisas diferentes — sem a referência, o número perde
    metade do sentido.
    """
    if probabilidade is None:
        st.caption("Probabilidade não informada pelo pipeline.")
        return

    percentual = max(0.0, min(1.0, float(probabilidade))) * 100
    acima = limiar is not None and probabilidade >= limiar
    cor = "#B3261E" if acima else "#1B6B3A"
    marca = ""
    if limiar is not None:
        marca = f'<div class="fg-barra-limiar" style="left:{max(0.0, min(1.0, float(limiar))) * 100:.1f}%"></div>'

    texto_limiar = (
        f"Limiar de decisão: {limiar:.0%}" if limiar is not None else "Limiar não informado"
    )
    st.markdown(
        f"""
        <div style="font-size:2.1rem;font-weight:700;color:{cor};line-height:1.1">
          {percentual:.1f}%
        </div>
        <div class="fg-legenda">Probabilidade estimada pelo modelo</div>
        <div class="fg-barra-fundo">
          <div class="fg-barra-preenchida" style="width:{percentual:.1f}%;background:{cor}"></div>
          {marca}
        </div>
        <div class="fg-legenda">{texto_limiar}</div>
        """,
        unsafe_allow_html=True,
    )


def _formatar_valor_feature(valor: Any) -> str:
    if isinstance(valor, Real) and not isinstance(valor, bool):
        numero = float(valor)
        if math.isfinite(numero):
            return f"{numero:,.4g}".replace(",", "X").replace(".", ",").replace("X", ".")
    if valor is None:
        return "não informado"
    return str(valor)


def fatores_shap(fatores: Sequence[Mapping[str, Any]]) -> None:
    """
    Contribuições como barras divergentes a partir do centro.

    Direita e vermelho aumentam a probabilidade; esquerda e verde reduzem. A
    escala é relativa ao maior valor absoluto do conjunto — comparar
    contribuições entre transações diferentes exigiria escala fixa, e esta tela
    explica uma transação por vez.
    """
    preparados = preparar_fatores_shap(fatores)
    if not preparados:
        st.info("O pipeline não retornou fatores SHAP para este caso.")
        return

    maximo = max(abs(fator["contribuicao"]) for fator in preparados)
    if maximo == 0:
        maximo = 1.0

    for fator in preparados:
        nome = html.escape(fator["rotulo"])
        contribuicao = fator["contribuicao"]
        largura = abs(contribuicao) / maximo * 100
        sobe = contribuicao > 0

        valor = _formatar_valor_feature(fator.get("valor"))
        sinal = "aumentou" if sobe else "reduziu" if contribuicao < 0 else "neutro"

        if sobe:
            metades = (
                '<div class="fg-fator-metade fg-fator-esq"></div>'
                f'<div class="fg-fator-metade"><div class="fg-fator-barra fg-sobe" style="width:{largura:.1f}%"></div></div>'
            )
        else:
            metades = (
                f'<div class="fg-fator-metade fg-fator-esq"><div class="fg-fator-barra fg-desce" style="width:{largura:.1f}%"></div></div>'
                '<div class="fg-fator-metade"></div>'
            )

        st.markdown(
            f"""
            <div class="fg-fator">
              <div class="fg-fator-topo">
                <span class="fg-fator-nome">{nome}</span>
                <span class="fg-fator-valor">valor {html.escape(valor)} &nbsp; {sinal} {contribuicao:+.4f}</span>
              </div>
              <div class="fg-fator-trilho">{metades}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="fg-legenda-shap">
          <span><i class="fg-bolinha fg-bolinha-sobe"></i>Aumentou o risco</span>
          <span><i class="fg-bolinha fg-bolinha-desce"></i>Reduziu o risco</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detalhe_fator_shap(fator: Mapping[str, Any]) -> None:
    """Explica um fator selecionado sem extrapolar a documentação disponível."""
    valor = html.escape(_formatar_valor_feature(fator.get("valor")))
    contribuicao = float(fator["contribuicao"])
    if contribuicao > 0:
        efeito, classe = "aumentou o risco estimado", "fg-impacto-sobe"
    elif contribuicao < 0:
        efeito, classe = "reduziu o risco estimado", "fg-impacto-desce"
    else:
        efeito, classe = "não alterou a previsão", ""

    st.markdown(
        f"""
        <div class="fg-detalhe-fator">
          <div class="fg-detalhe-linha"><span>Nome técnico</span><strong>{html.escape(fator['nome_tecnico'])}</strong></div>
          <div class="fg-detalhe-linha"><span>Tipo</span><strong>{html.escape(fator['tipo'])}</strong></div>
          <div class="fg-detalhe-linha"><span>Valor observado</span><strong>{valor}</strong></div>
          <div class="fg-detalhe-linha"><span>Impacto SHAP</span><strong class="{classe}">{contribuicao:+.4f}</strong></div>
          <div class="fg-detalhe-linha"><span>Efeito</span><strong class="{classe}">{efeito}</strong></div>
          <div class="fg-caixa-informacao">
            <b>O que representa?</b>
            <p>{html.escape(fator['descricao'])}</p>
          </div>
          <p class="fg-limitacao">ⓘ {html.escape(fator['limitacao'])}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def painel_fatores_shap(
    fatores: Sequence[Mapping[str, Any]],
    *,
    chave: str,
) -> None:
    """Combina gráfico divergente e explicação selecionável de cada fator."""
    preparados = preparar_fatores_shap(fatores)
    if not preparados:
        st.info("O pipeline não retornou fatores SHAP válidos para este caso.")
        return

    grafico, detalhes = st.columns([1.65, 1], gap="medium")
    with grafico, st.container(border=True):
        titulo, ajuda = st.columns([3, 1])
        titulo.markdown("#### Indicadores explicativos do projeto (SHAP)")
        with ajuda, st.popover("ⓘ Como interpretar", use_container_width=True):
            st.markdown(
                "Barras **vermelhas** aumentaram o risco estimado; barras "
                "**azuis** o reduziram. O valor SHAP mede influência na "
                "previsão desta transação — não é porcentagem, causa, intenção "
                "ou confirmação de fraude."
            )
        fatores_shap(fatores)
        st.caption(
            "Visualização restrita às variáveis derivadas e documentadas pela equipe."
        )

    with detalhes, st.container(border=True):
        st.markdown("#### Detalhes do indicador")
        indice = st.selectbox(
            "Indicador",
            options=range(len(preparados)),
            format_func=lambda posicao: preparados[posicao]["rotulo"],
            key=f"seletor_fator_{chave}",
            label_visibility="collapsed",
        )
        detalhe_fator_shap(preparados[indice])


def explicacao(texto: str) -> None:
    st.markdown(
        f'<div class="fg-explicacao">{html.escape(texto)}</div>',
        unsafe_allow_html=True,
    )


def documentos(lista: Sequence[Mapping[str, Any]]) -> None:
    if not lista:
        st.info(
            "Nenhum documento foi recuperado. A explicação acima, se existir, "
            "não está apoiada em fontes documentais."
        )
        return

    for posicao, documento in enumerate(lista, start=1):
        titulo = documento.get("titulo") or documento.get("fonte") or "Documento sem título"
        score = documento.get("score")
        rotulo = f"{posicao}. {titulo}"
        if isinstance(score, (int, float)):
            rotulo += f"  ·  similaridade {score:.3f}"

        with st.expander(rotulo):
            trecho = documento.get("trecho") or documento.get("page_content")
            st.write(trecho if trecho else "_Trecho não informado pelo pipeline._")
            fonte = documento.get("fonte") or documento.get("source_url")
            if fonte and fonte != titulo:
                st.caption(f"Fonte: {fonte}")


def resumo_tecnico(dados: Mapping[str, Any]) -> None:
    """Campos ausentes aparecem como 'não informado', nunca preenchidos."""
    with st.expander("Detalhes da análise"):
        linhas = []
        for rotulo, valor in dados.items():
            mostrado = "não informado" if valor is None else valor
            linhas.append(f"| {rotulo} | {mostrado} |")
        st.markdown(
            "| Item | Valor |\n|---|---|\n" + "\n".join(linhas)
        )


def rodape() -> None:
    st.markdown(
        '<div class="fg-rodape">Protótipo acadêmico — TCC em Ciência da Computação. '
        "Os dados são do IEEE-CIS Fraud Detection (transações de cartão), com variáveis "
        "inspiradas no Pix. <b>Não utiliza dados reais do Pix</b> e não deve produzir "
        "bloqueio, acusação ou decisão automática sobre pessoas.</div>",
        unsafe_allow_html=True,
    )
