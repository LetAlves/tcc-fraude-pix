"""
Camada 3 (parte 2): fatores do SHAP + documentos recuperados → explicação.

A construção do prompt é separada da chamada ao modelo de linguagem, e isso não
é organização estética: o prompt é a parte que carrega as restrições
metodológicas do trabalho, e precisa ser testável sem rede, sem credencial e sem
custo por chamada. `montar_prompt` é função pura; `explicar` é a única que fala
com o mundo externo.

## O que o prompt proíbe, e por quê

Três restrições estão escritas no prompt porque são as três formas de um modelo
de linguagem produzir texto convincente e errado neste trabalho:

1. **Não afirmar causalidade.** Uma contribuição do SHAP diz que a variável
   influenciou a saída do modelo, não que ela causou a fraude.
2. **Não atribuir significado às colunas anônimas.** O IEEE-CIS não publica o
   que `V257` ou `C13` representam. Um modelo de linguagem preenche essa lacuna
   com plausibilidade se não for impedido.
3. **Não afirmar nada que não esteja nos documentos recuperados.** É o motivo de
   existir RAG aqui: a explicação precisa ser rastreável até uma norma, e não
   até a memória do modelo.

## Ponto de integração pendente

A dupla decidiu em julho usar Claude Haiku 4.5 como principal e Ollama local
como alternativa (`reports/pessoa_2/julho/03_decisao_llm.md`), mas **nenhum
cliente foi implementado no projeto até agora**. Por isso `chamar_llm` é
`{{PREENCHER}}`: o contrato está definido — recebe texto, devolve texto — e a
implementação entra quando a credencial e a biblioteca forem decididas.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)

LIMITE_FATORES = 3
LIMITE_CARACTERES_DOCUMENTO = 400

INSTRUCOES = """Você explica, em português do Brasil, por que um sistema automatizado sinalizou uma transação.

Regras que você deve seguir sem exceção:
- Use somente as evidências fornecidas abaixo. Não acrescente informação de conhecimento próprio.
- Os fatores listados indicam influência sobre a decisão do modelo, não causa da fraude. Não afirme causalidade.
- Variáveis com nomes não descritivos (por exemplo C1, V257, card1) são colunas anônimas: não invente o que representam.
- Cite os documentos apenas para o que eles de fato dizem, mencionando a fonte.
- Se as evidências forem insuficientes para uma afirmação, diga que são insuficientes.
- A transação é de um conjunto de dados de pesquisa. Não trate ninguém como culpado.

Escreva de três a cinco frases, em linguagem simples, para alguém sem formação técnica."""


class ClienteLLM(Protocol):
    """Contrato mínimo do cliente: recebe um prompt, devolve texto."""

    def __call__(self, prompt: str) -> str: ...


def _formatar_fatores(fatores: list[dict[str, Any]]) -> str:
    linhas = []
    for posicao, fator in enumerate(fatores[:LIMITE_FATORES], start=1):
        nome = str(fator.get("feature", "?")).strip() or "?"
        contribuicao = fator.get("contribuicao", fator.get("shap"))
        if isinstance(contribuicao, (int, float)):
            direcao = "aumentou" if contribuicao > 0 else "reduziu"
            linhas.append(
                f"{posicao}. {nome} — {direcao} a probabilidade estimada "
                f"(contribuição {contribuicao:+.4f})"
            )
        else:
            linhas.append(f"{posicao}. {nome}")
    return "\n".join(linhas)


def _formatar_documentos(documentos: list[Any]) -> str:
    if not documentos:
        return "Nenhum documento foi recuperado para esta consulta."

    linhas = []
    for posicao, documento in enumerate(documentos, start=1):
        if hasattr(documento, "resumo"):
            fonte, trecho = documento.fonte, documento.resumo(LIMITE_CARACTERES_DOCUMENTO)
        else:  # aceita dicionários, para facilitar testes e reuso
            fonte = str(documento.get("fonte", "fonte não identificada"))
            texto = " ".join(str(documento.get("texto", "")).split())
            trecho = texto[:LIMITE_CARACTERES_DOCUMENTO]
        linhas.append(f"[{posicao}] Fonte: {fonte}\n{trecho}")
    return "\n\n".join(linhas)


def montar_prompt(
    fatores_shap: list[dict[str, Any]],
    documentos: list[Any],
    probabilidade: float | None = None,
    sinalizada: bool | None = None,
) -> str:
    """
    Monta o prompt completo. Função pura: mesma entrada, mesma saída, sem rede.

    Levanta `ValueError` se não houver fator algum — sem fatores não há o que
    explicar, e um prompt vazio produziria texto genérico convincente, que é
    exatamente o que este trabalho não pode aceitar.
    """
    if not fatores_shap:
        raise ValueError("é necessário ao menos um fator do SHAP para explicar")

    partes = [INSTRUCOES, ""]

    if sinalizada is not None or probabilidade is not None:
        estado = "sinalizada como suspeita" if sinalizada else "não sinalizada"
        if probabilidade is not None:
            partes.append(
                f"DECISÃO DO MODELO: {estado}, com probabilidade estimada de "
                f"{probabilidade:.1%}."
            )
        else:
            partes.append(f"DECISÃO DO MODELO: {estado}.")
        partes.append("")

    partes += [
        "FATORES DE MAIOR INFLUÊNCIA (SHAP):",
        _formatar_fatores(fatores_shap),
        "",
        "DOCUMENTOS RECUPERADOS:",
        _formatar_documentos(documentos),
        "",
        "EXPLICAÇÃO:",
    ]
    return "\n".join(partes)


def chamar_llm(prompt: str) -> str:
    """
    {{PREENCHER}} — cliente do modelo de linguagem.

    Decisão registrada em `reports/pessoa_2/julho/03_decisao_llm.md`: Claude
    Haiku 4.5 como principal, Ollama local como alternativa. Nenhum cliente foi
    implementado ainda, e a credencial deve vir do `.env`, nunca do código.

    Implementar aqui mantém o restante do módulo intacto: `montar_prompt` já é
    testável e `explicar` já aceita um cliente injetado.
    """
    raise NotImplementedError(
        "{{PREENCHER}}: configure o cliente do LLM (Claude Haiku 4.5 ou Ollama) "
        "ou injete um cliente em explicar(cliente=...)."
    )


def explicar(
    fatores_shap: list[dict[str, Any]],
    documentos: list[Any],
    probabilidade: float | None = None,
    sinalizada: bool | None = None,
    cliente: Callable[[str], str] | None = None,
) -> str:
    """
    Produz a explicação em português a partir dos fatores e dos documentos.

    `cliente` é injetável para que os testes usem um dublê e para que a troca
    entre Claude e Ollama não exija mexer neste módulo.
    """
    prompt = montar_prompt(fatores_shap, documentos, probabilidade, sinalizada)
    executor = cliente or chamar_llm

    texto = executor(prompt)
    if not isinstance(texto, str) or not texto.strip():
        raise RuntimeError("o cliente do LLM devolveu uma resposta vazia")
    return texto.strip()
