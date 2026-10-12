"""Gera as figuras finais do TCC em 300 dpi (m5_p1_3).

Reutiliza o XGBoost entregue em ``models/xgboost`` sem retreinar nem reajustar o
pré-processador, e o limiar congelado em ``politica_decisao.json``. As figuras
não têm título interno: pela ABNT, título e fonte ficam no texto da monografia.

Saída em ``reports/pessoa_1/setembro/figuras/``:

* ``fig_distribuicao_alvo.png`` e ``fig_top_correlacoes.png`` (EDA de maio);
* ``fig_curva_pr.png``: curvas de precisão-recall na validação e no teste;
* ``fig_matriz_confusao_teste.png``: matriz de confusão no teste;
* ``fig_arquitetura.png``: fluxo das camadas do protótipo;
* ``fig_shap_casos.png``: contribuições SHAP dos casos VP, FP, FN e VN, legível
  para a monografia;
* ``shap/``: summary plot em português e force plots originais (apoio);
* ``figuras_finais.json``: métricas recalculadas, conferidas com a avaliação publicada.

Execute a partir da raiz: ``python -m scripts.gerar_figuras_finais``.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.data_loader import carregar_dados  # noqa: E402
from src.features.pix_features import criar_features_pix  # noqa: E402
from src.features.preprocessor import (  # noqa: E402
    COLUNA_ALVO,
    dividir_temporal,
    reduzir_precisao,
)
from src.models.evaluator import (  # noqa: E402
    avaliar_probabilidades,
    matriz_confusao,
    plotar_curva_precisao_recall,
)
from src.models.persistencia import carregar, prever  # noqa: E402
from src.models.shap_xgboost import (  # noqa: E402
    SEMENTE,
    TAMANHO_FUNDO,
    amostrar_indices,
    gerar_relatorio_shap,
)

DPI = 300
DIRETORIO_MODELO = RAIZ / "models" / "xgboost"
DIRETORIO_SAIDA = RAIZ / "reports" / "pessoa_1" / "setembro" / "figuras"
AVALIACAO_PUBLICADA = RAIZ / "reports" / "avaliacao_xgboost_entregue.json"
SPLIT_ESPERADO = {"treino": 413378, "validacao": 88581, "teste": 88581}
TOLERANCIA_METRICA = 1e-6

AZUL, VERMELHO, CINZA = "#4878CF", "#E24A33", "#6B6B6B"


def sha256(caminho: Path) -> str:
    resumo = hashlib.sha256()
    with caminho.open("rb") as fluxo:
        for bloco in iter(lambda: fluxo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def numero_pt(valor: float, casas: int | None = None) -> str:
    """Formata número com vírgula decimal e ponto de milhar (padrão brasileiro)."""
    texto = f"{valor:,.{casas}f}" if casas is not None else f"{valor:,g}"
    return texto.replace(",", "_").replace(".", ",").replace("_", ".").replace("-", "\u2212")


def virgula(eixo, x: bool = True, y: bool = True) -> None:
    from matplotlib.ticker import FuncFormatter

    formatador = FuncFormatter(lambda valor, _: numero_pt(valor))
    if x:
        eixo.xaxis.set_major_formatter(formatador)
    if y:
        eixo.yaxis.set_major_formatter(formatador)


def salvar(figura, nome: str) -> Path:
    destino = DIRETORIO_SAIDA / nome
    figura.savefig(destino, dpi=DPI, bbox_inches="tight")
    plt.close(figura)
    return destino


def figura_distribuicao_alvo(alvo) -> None:
    contagens = alvo.value_counts().sort_index()
    percentuais = contagens / contagens.sum() * 100
    rotulos = ["Legítima (0)", "Fraude (1)"]
    figura, eixos = plt.subplots(1, 2, figsize=(12, 5))
    eixos[0].bar(rotulos, contagens.values, color=[AZUL, VERMELHO], edgecolor="white")
    eixos[0].set_ylabel("Número de transações")
    for posicao, valor in enumerate(contagens.values):
        eixos[0].text(posicao, valor * 1.01, f"{valor:,}".replace(",", "."),
                      ha="center", va="bottom", fontweight="bold")
    eixos[0].set_ylim(0, contagens.max() * 1.1)
    virgula(eixos[0], x=False)
    eixos[1].pie(
        percentuais.values,
        labels=[f"{r}\n{p:.2f}%".replace(".", ",") for r, p in zip(rotulos, percentuais.values)],
        colors=[AZUL, VERMELHO],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    figura.tight_layout()
    salvar(figura, "fig_distribuicao_alvo.png")


def figura_top_correlacoes(transacoes, top_n: int = 10) -> None:
    numericas = transacoes.select_dtypes(include="number").drop(
        columns=[COLUNA_ALVO], errors="ignore"
    ).astype(np.float32)
    correlacoes = numericas.corrwith(transacoes[COLUNA_ALVO].astype(np.float32)).dropna()
    top = correlacoes.abs().nlargest(top_n)
    sinais = correlacoes[top.index]
    cores = [VERMELHO if v > 0 else AZUL for v in sinais.values]
    figura, eixo = plt.subplots(figsize=(10, 6))
    barras = eixo.barh(top.index[::-1], top.values[::-1], color=cores[::-1], edgecolor="white")
    eixo.set_xlabel("|Correlação de Pearson| com isFraud")
    eixo.set_xlim(0, top.values.max() * 1.15)
    virgula(eixo, y=False)
    for barra, valor in zip(barras, top.values[::-1]):
        eixo.text(valor + 0.002, barra.get_y() + barra.get_height() / 2,
                  f"{valor:.3f}".replace(".", ","), va="center", fontsize=9)
    from matplotlib.patches import Patch

    legenda = []
    if (sinais > 0).any():
        legenda.append(Patch(color=VERMELHO, label="Correlação positiva"))
    if (sinais < 0).any():
        legenda.append(Patch(color=AZUL, label="Correlação negativa"))
    eixo.legend(handles=legenda, loc="lower right")
    figura.tight_layout()
    salvar(figura, "fig_top_correlacoes.png")


def figura_curva_pr(y_val, p_val, y_teste, p_teste, limiar: float) -> None:
    figura, eixos = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for eixo, y, p, nome in ((eixos[0], y_val, p_val, "Validação"),
                             (eixos[1], y_teste, p_teste, "Teste")):
        plotar_curva_precisao_recall(y, p, rotulo=f"XGBoost — {nome}", eixo=eixo)
        metricas = avaliar_probabilidades(y, p, limiar=limiar)
        eixo.scatter([metricas["recall"]], [metricas["precisao"]], color=VERMELHO,
                     zorder=5, label=f"limiar {limiar:.4f}".replace(".", ","))
        legenda = eixo.legend(loc="upper right", fontsize=9)
        for texto in legenda.get_texts():
            texto.set_text(texto.get_text().replace(".", ","))
        virgula(eixo)
        eixo.set_title(nome)
    figura.tight_layout()
    salvar(figura, "fig_curva_pr.png")


def figura_matriz_confusao(y_teste, p_teste, limiar: float) -> list[list[int]]:
    valores = np.asarray(matriz_confusao(y_teste, p_teste, limiar=limiar).values, dtype=int)
    figura, eixo = plt.subplots(figsize=(5.5, 4.5))
    eixo.imshow(valores, cmap="Blues")
    rotulos = ["Legítima", "Fraude"]
    eixo.set_xticks([0, 1], [f"Prevista\n{r}" for r in rotulos])
    eixo.set_yticks([0, 1], [f"Real\n{r}" for r in rotulos])
    limite = valores.max() / 2
    for i in range(2):
        for j in range(2):
            eixo.text(j, i, f"{valores[i, j]:,}".replace(",", "."), ha="center", va="center",
                      fontsize=13, fontweight="bold",
                      color="white" if valores[i, j] > limite else "black")
    figura.tight_layout()
    salvar(figura, "fig_matriz_confusao_teste.png")
    return valores.tolist()


NOMES_CASOS = {
    "verdadeiro_positivo": "Verdadeiro positivo",
    "falso_positivo": "Falso positivo",
    "falso_negativo": "Falso negativo",
    "verdadeiro_negativo": "Verdadeiro negativo",
}


def figura_shap_casos(casos_locais: list[dict], n_variaveis: int = 8) -> None:
    """Painel com as maiores contribuições SHAP de cada caso local (VP, FP, FN, VN).

    Substitui os force plots na monografia: valores arredondados, rótulos em
    português e sem sobreposição de texto.
    """
    figura, eixos = plt.subplots(2, 2, figsize=(13, 8.5))
    for eixo, caso in zip(eixos.ravel(), casos_locais):
        contribuicoes = sorted(caso["principais_contribuicoes"],
                               key=lambda c: abs(c["shap"]), reverse=True)[:n_variaveis]
        nomes = [c["feature"] for c in contribuicoes][::-1]
        valores = [c["shap"] for c in contribuicoes][::-1]
        cores = [VERMELHO if v > 0 else AZUL for v in valores]
        eixo.barh(nomes, valores, color=cores, edgecolor="white")
        eixo.axvline(0, color=CINZA, linewidth=0.8)
        limite = max(abs(v) for v in valores) * 1.35
        eixo.set_xlim(-limite, limite)
        for posicao, valor in enumerate(valores):
            eixo.text(valor + (0.01 * limite if valor > 0 else -0.01 * limite), posicao,
                      numero_pt(valor, 3), va="center", fontsize=8.5,
                      ha="left" if valor > 0 else "right")
        eixo.set_title(
            f"{NOMES_CASOS.get(caso['tipo'], caso['tipo'])}: probabilidade "
            f"{numero_pt(caso['probabilidade_fraude'], 2)} (base {numero_pt(caso['valor_base'], 2)})",
            fontsize=10.5,
        )
        virgula(eixo, y=False)
    for eixo in eixos[1]:
        eixo.set_xlabel("Contribuição SHAP para a probabilidade de fraude")
    from matplotlib.patches import Patch

    figura.legend(handles=[Patch(color=VERMELHO, label="Aumenta o risco"),
                           Patch(color=AZUL, label="Reduz o risco")],
                  loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    figura.tight_layout(rect=(0, 0.03, 1, 1))
    salvar(figura, "fig_shap_casos.png")


def figura_arquitetura(limiar: float) -> None:
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    etapas = [
        ("1. Entrada", "Transação informada\nna interface Streamlit"),
        ("2. Classificação", "XGBoost estima a\nprobabilidade de fraude\n"
                             f"(limiar {limiar:.4f})".replace(".", ",")),
        ("3. Explicabilidade", "SHAP (TreeExplainer)\nidentifica as variáveis\nque pesaram na decisão"),
        ("4. Recuperação (RAG)", "Embeddings + FAISS\nbuscam trechos de\nnormas BCB e FEBRABAN"),
        ("5. Geração", "LLM redige a explicação\nem português, citando\nos trechos recuperados"),
    ]
    figura, eixo = plt.subplots(figsize=(15, 3.0))
    largura, altura, espaco = 2.6, 1.55, 0.55
    for i, (titulo, texto) in enumerate(etapas):
        x = i * (largura + espaco)
        cor = VERMELHO if i == 1 else AZUL
        eixo.add_patch(FancyBboxPatch((x, 0), largura, altura,
                                      boxstyle="round,pad=0.04,rounding_size=0.12",
                                      facecolor="white", edgecolor=cor, linewidth=2))
        eixo.text(x + largura / 2, altura - 0.32, titulo,
                  ha="center", va="center", fontsize=11, fontweight="bold", color=cor)
        eixo.text(x + largura / 2, altura / 2 - 0.15, texto,
                  ha="center", va="center", fontsize=9.5)
        if i < len(etapas) - 1:
            eixo.add_patch(FancyArrowPatch((x + largura + 0.05, altura / 2),
                                           (x + largura + espaco - 0.05, altura / 2),
                                           arrowstyle="-|>", mutation_scale=18,
                                           color=CINZA, linewidth=1.6))
    eixo.text(1 * (largura + espaco) + largura / 2, -0.25,
              "transações abaixo do limiar seguem sem acionar o RAG",
              ha="center", va="center", fontsize=8.5, color=CINZA, style="italic")
    eixo.set_xlim(-0.15, len(etapas) * (largura + espaco) - espaco + 0.15)
    eixo.set_ylim(-0.45, altura + 0.1)
    eixo.axis("off")
    salvar(figura, "fig_arquitetura.png")


def main() -> None:
    inicio = time.time()

    def marco(mensagem: str) -> None:
        pico = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # bytes no macOS
        if sys.platform != "darwin":
            pico *= 1024  # Linux informa em KiB
        print(f"[{time.time() - inicio:5.0f}s | pico {pico:,.0f} MB] {mensagem}", flush=True)

    DIRETORIO_SAIDA.mkdir(parents=True, exist_ok=True)
    publicada = json.loads(AVALIACAO_PUBLICADA.read_text(encoding="utf-8"))
    politica = json.loads((DIRETORIO_MODELO / "politica_decisao.json").read_text(encoding="utf-8"))
    limiar = float(politica["limiar"])
    for nome, esperado in publicada["hashes_modelo"].items():
        if sha256(DIRETORIO_MODELO / nome) != esperado:
            raise ValueError(f"Artefato divergente da avaliação publicada: {nome}")
    preprocessador, modelo, _ = carregar(DIRETORIO_MODELO)
    figura_arquitetura(limiar)
    marco("modelo verificado; figura da arquitetura gravada")

    transacoes, _, mesclado = carregar_dados(baixar_se_ausente=False, nrows=None)
    figura_distribuicao_alvo(transacoes[COLUNA_ALVO])
    figura_top_correlacoes(transacoes)
    del transacoes
    gc.collect()
    marco("figuras do EDA gravadas")

    frame = reduzir_precisao(criar_features_pix(mesclado))
    del mesclado
    gc.collect()
    treino, validacao, teste = dividir_temporal(frame)
    del frame
    contagens = {"treino": len(treino), "validacao": len(validacao), "teste": len(teste)}
    if contagens != SPLIT_ESPERADO:
        raise ValueError(f"Split difere do protocolo publicado: {contagens}")
    X_fundo = treino.drop(columns=[COLUNA_ALVO]).iloc[
        amostrar_indices(len(treino), TAMANHO_FUNDO, SEMENTE)
    ].copy()
    del treino
    gc.collect()
    marco("split temporal conferido")

    X_val = validacao.drop(columns=[COLUNA_ALVO])
    y_val = validacao[COLUNA_ALVO].to_numpy(dtype=int)
    X_teste = teste.drop(columns=[COLUNA_ALVO])
    y_teste = teste[COLUNA_ALVO].to_numpy(dtype=int)
    del validacao, teste
    gc.collect()
    p_val = prever(preprocessador, modelo, X_val)
    p_teste = prever(preprocessador, modelo, X_teste)
    del X_teste
    gc.collect()

    metricas_val = avaliar_probabilidades(y_val, p_val, limiar=limiar)
    metricas_teste = avaliar_probabilidades(y_teste, p_teste, limiar=limiar)
    for chave in ("auc_pr", "auc_roc", "precisao", "recall", "f1"):
        if abs(metricas_teste[chave] - publicada["teste"][chave]) > TOLERANCIA_METRICA:
            raise ValueError(
                f"{chave} do teste ({metricas_teste[chave]}) difere da avaliação publicada "
                f"({publicada['teste'][chave]}); figuras não serão gravadas."
            )
    marco("métricas do teste conferem com a avaliação publicada")

    figura_curva_pr(y_val, p_val, y_teste, p_teste, limiar)
    matriz = figura_matriz_confusao(y_teste, p_teste, limiar)
    if matriz != publicada["matriz_confusao_teste"]["valores"]:
        raise ValueError("Matriz de confusão difere da avaliação publicada.")
    del p_teste, y_teste
    gc.collect()
    marco("curva PR e matriz de confusão gravadas")

    # Mesmo caminho do relatório de julho ("artefato local reutilizado"), mas em 300 dpi.
    relatorio_shap = gerar_relatorio_shap(
        modelo,
        preprocessador,
        preprocessador.transform(X_fundo),
        preprocessador.transform(X_val),
        y_val,
        DIRETORIO_SAIDA / "shap",
        dpi=DPI,
        rotulos_pt=True,
    )
    figura_shap_casos(relatorio_shap["casos_locais"])
    marco("SHAP gravado")

    julho = json.loads(
        (RAIZ / "reports/pessoa_1/julho/shap/shap_xgboost.json").read_text(encoding="utf-8")
    )
    ranking = [item["feature"] for item in relatorio_shap["importancia_global"][:10]]
    ranking_julho = [item["feature"] for item in julho["importancia_global"][:10]]

    resumo = {
        "tarefa": "m5_p1_3",
        "dpi": DPI,
        "limiar": limiar,
        "split": contagens,
        "metricas_validacao": metricas_val,
        "metricas_teste": metricas_teste,
        "matriz_confusao_teste": matriz,
        "confere_com": str(AVALIACAO_PUBLICADA.relative_to(RAIZ)),
        "shap_top10_igual_julho": ranking == ranking_julho,
        "shap_top10": ranking,
        "figuras": sorted(
            str(p.relative_to(DIRETORIO_SAIDA)) for p in DIRETORIO_SAIDA.rglob("*.png")
        ),
        "segundos_total": round(time.time() - inicio),
    }
    (DIRETORIO_SAIDA / "figuras_finais.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    marco("resumo gravado")
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
