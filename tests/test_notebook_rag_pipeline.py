"""Validação estática do notebook ponta a ponta, sem dados ou chamadas pagas."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "04_rag_pipeline.ipynb"


def _carregar() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_e_json_valido_e_todas_as_celulas_compilam() -> None:
    notebook = _carregar()
    assert notebook["nbformat"] == 4
    for indice, celula in enumerate(notebook["cells"]):
        if celula["cell_type"] == "code":
            compile("".join(celula["source"]), f"celula-{indice}", "exec")


def test_notebook_configura_dez_exemplos_e_cliente_real() -> None:
    conteudo = NOTEBOOK.read_text(encoding="utf-8")
    assert "N_EXEMPLOS = 10" in conteudo
    assert "criar_cliente_llm_de_ambiente" in conteudo
    assert "cliente_inspecao" not in conteudo
    assert "{{PREENCHER}}" not in conteudo


def test_notebook_nao_contem_segredo() -> None:
    conteudo = NOTEBOOK.read_text(encoding="utf-8")
    assert "sk-ant-" not in conteudo
    assert "ANTHROPIC_API_KEY=" not in conteudo
