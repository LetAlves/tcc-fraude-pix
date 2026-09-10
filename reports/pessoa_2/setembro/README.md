# Setembro de 2026 — fechamento das tarefas da Pessoa 2

Esta pasta registra a revalidação realizada em **10/09/2026** sobre as entregas de código, pesquisa e escrita da Pessoa 2.

## Situação

| Frente | Resultado | Evidência |
|---|---|---|
| Código RAG | Aprovado e atualizado para preservar datas de vigência | 30 testes, construção offline e consulta semântica |
| Pesquisa documental | Quatro fontes revalidadas; Guia MED atualizado de 4.3 para 4.4 | [`01_revalidacao_corpus_rag.md`](01_revalidacao_corpus_rag.md) e snapshot de hashes |
| Escrita SHAP | Protocolo detalhado com configuração explícita e escala comparável | [`../julho/04_metodologia_shap.md`](../julho/04_metodologia_shap.md) e capítulo 3 da monografia |

Os PDFs, JSON normativos, chunks e vetores continuam ignorados pelo Git. Somente código, catálogo, testes, documentação e hashes foram preparados para versionamento.

## Resultado reprodutível

```text
4 fontes → 106 unidades → 290 chunks pais (500/50)
         → 1.195 janelas de embedding (126/24)
         → 1.195 vetores de 384 dimensões no FAISS IndexFlatIP
```

O Guia MED 4.4 possui vigência escalonada: a alteração de prazo vale desde **01/09/2026**, enquanto a informação de camada no grafo de rastreamento está indicada para **26/10/2026**. Essa distinção foi incluída nos metadados do corpus.
