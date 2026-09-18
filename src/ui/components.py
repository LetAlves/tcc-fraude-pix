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
from typing import Any, Mapping, Sequence

import streamlit as st

from src.ui.styles import CSS


def aplicar_estilo() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def cabecalho(operacional: bool, detalhe: str) -> None:
    ponto = "#6EE7A8" if operacional else "#FFD166"
    st.markdown(
        f"""
        <div class="fg-cabecalho">
          <div>
            <h1>🛡 FraudGuard PIX</h1>
            <p>Sistema Inteligente de Detecção e Explicação de Fraudes</p>
          </div>
          <div class="fg-status">
            <span style="color:{ponto}">●</span> <b>{html.escape(detalhe)}</b>
          </div>
        </div>
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


def fatores_shap(fatores: Sequence[Mapping[str, Any]]) -> None:
    """
    Contribuições como barras divergentes a partir do centro.

    Direita e vermelho aumentam a probabilidade; esquerda e verde reduzem. A
    escala é relativa ao maior valor absoluto do conjunto — comparar
    contribuições entre transações diferentes exigiria escala fixa, e esta tela
    explica uma transação por vez.
    """
    if not fatores:
        st.info("O pipeline não retornou fatores SHAP para este caso.")
        return

    maximo = max((abs(float(f.get("contribuicao", 0) or 0)) for f in fatores), default=0.0)
    if maximo == 0:
        maximo = 1.0

    for fator in fatores:
        nome = html.escape(str(fator.get("feature", "?")))
        contribuicao = float(fator.get("contribuicao", 0) or 0)
        largura = abs(contribuicao) / maximo * 100
        sobe = contribuicao > 0

        valor = fator.get("valor")
        texto_valor = f"valor {valor:.4g}" if isinstance(valor, (int, float)) else ""
        sinal = "↑ aumenta" if sobe else "↓ reduz"

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
                <span class="fg-fator-valor">{texto_valor} &nbsp; {sinal} {contribuicao:+.4f}</span>
              </div>
              <div class="fg-fator-trilho">{metades}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "As contribuições indicam influência sobre a saída do modelo — não "
        "demonstram causalidade, intenção nem culpa. Colunas com nomes não "
        "descritivos são anônimas no dataset original e não têm significado publicado."
    )


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
