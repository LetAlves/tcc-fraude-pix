# Julho de 2026 — Pessoa 2 (RAG + escrita)

Este diretório reúne investigação, implementação e texto metodológico das tarefas de julho atribuídas à Pessoa 2. Data da verificação local: **30/08/2026**.

> Atualização: o corpus foi revalidado em **10/09/2026**, quando o Guia MED 4.3 foi substituído pela versão 4.4. Consulte a [revalidação de setembro](../setembro/README.md). Os números abaixo foram atualizados para a execução mais recente.

## Situação das tarefas

| ID | Tarefa | Situação | Evidência |
|---|---|---|---|
| `m3_p2_1` | Coletar e versionar documentos para o RAG | Concluída localmente | [`01_investigacao_corpus_rag.md`](01_investigacao_corpus_rag.md), catálogo e snapshot de hashes |
| `m3_p2_2` | Implementar chunking em `knowledge_base.py` | Concluída e testada | chunks pais 500/50, extração e rastreabilidade |
| `m3_p2_3` | Gerar embeddings multilíngues | Concluída e executada | MiniLM de 384 dimensões, com janelas compatíveis com 128 tokens |
| `m3_p2_4` | Criar banco vetorial FAISS | Concluída e executada | `IndexFlatIP` com 1.195 vetores e hashes de integridade |
| `m3_p2_5` | Escrever a metodologia da Camada 2 — SHAP | Rascunho concluído para revisão | [`04_metodologia_shap.md`](04_metodologia_shap.md) e capítulo 3 da monografia |

## Tarefas conjuntas e limites do escopo

| ID | Situação | Motivo |
|---|---|---|
| `m3_ab_1` — escolher LLM | Recomendação pronta; decisão conjunta pendente | exige aprovação de Letícia e Lucas sobre modelo, privacidade e custo |
| `m3_ab_2` — entregar resultados dos modelos ao orientador | Pendente | depende dos experimentos e do envio liderados pela Pessoa 1 |

As tarefas `m3_p1_1` a `m3_p1_5` — XGBoost/Optuna, Random Forest, métricas, SHAP executado e notebook `03_models` — pertencem à Pessoa 1 e não foram marcadas como concluídas nesta entrega.

## Resultados principais

- quatro fontes oficiais catalogadas, coletadas e verificadas por SHA-256;
- 106 unidades de fonte convertidas em 290 chunks pais;
- 1.195 janelas compatíveis com o MiniLM indexadas em FAISS;
- consulta de fumaça recuperando primeiro a Resolução BCB nº 103 e o Guia MED;
- 30 testes automatizados aprovados na revalidação de 10/09/2026;
- dados brutos, chunks e índice confirmados como ignorados pelo Git;
- proposta de Claude Haiku 4.5 como principal e Ollama como alternativa local, ainda sem decisão conjunta.

## Artefatos de implementação

- [`../../../config/rag_corpus_sources.json`](../../../config/rag_corpus_sources.json)
- [`../../../scripts/download_rag_corpus.py`](../../../scripts/download_rag_corpus.py)
- [`../../../scripts/build_rag_index.py`](../../../scripts/build_rag_index.py)
- [`../../../scripts/query_rag_index.py`](../../../scripts/query_rag_index.py)
- [`../../../src/rag/knowledge_base.py`](../../../src/rag/knowledge_base.py)
- [`../../../src/rag/embeddings.py`](../../../src/rag/embeddings.py)
- [`../../../src/rag/vector_store.py`](../../../src/rag/vector_store.py)

## Critério de pronto

Uma tarefa técnica é considerada concluída quando possui código, teste, execução real e limites documentados. Texto acadêmico é “rascunho concluído” até revisão da dupla e do orientador. Decisão conjunta só é concluída após concordância humana registrada; uma recomendação técnica não substitui a aprovação.
