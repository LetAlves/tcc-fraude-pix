# Junho de 2026 — Pessoa 2 (RAG + escrita)

Este diretório reúne as evidências das tarefas de junho atribuídas à Pessoa 2 e das tarefas conjuntas já verificadas. O mês conecta a revisão bibliográfica à metodologia e transforma o mapeamento IEEE-CIS→Pix aprovado em um registro rastreável de features implementadas.

Data da revisão: **18/08/2026**.

## Situação das tarefas da Pessoa 2

| ID | Tarefa | Situação | Evidência |
|---|---|---|---|
| `m2_p2_1` | Capítulo 2 — fraudes no Pix e pagamentos instantâneos | Concluída como rascunho para revisão | [`02_notas_revisao_bibliografica.md`](02_notas_revisao_bibliografica.md) e [`../../../monografia/capitulos/02_revisao_bibliografica.tex`](../../../monografia/capitulos/02_revisao_bibliografica.tex) |
| `m2_p2_2` | Capítulo 2 — ML para detecção de fraude | Concluída como rascunho para revisão | mesmos artefatos acima |
| `m2_p2_3` | Capítulo 2 — RAG e LLMs | Concluída como rascunho para revisão | mesmos artefatos acima |
| `m2_p2_4` | Estudar LangChain: chains, retrievers e prompt templates | Concluída com laboratório sem API externa | [`01_estudo_langchain.md`](01_estudo_langchain.md), [`../../../src/rag/langchain_basics.py`](../../../src/rag/langchain_basics.py) e testes |
| `m2_p2_5` | Visão geral das três camadas na Metodologia | Concluída como desenho planejado | [`03_metodologia_tres_camadas.md`](03_metodologia_tres_camadas.md) e [`../../../monografia/capitulos/03_metodologia.tex`](../../../monografia/capitulos/03_metodologia.tex) |

“Concluída como rascunho” significa que o texto tem estrutura, fontes e limites explícitos, mas ainda requer revisão da dupla e do orientador antes de ser tratado como versão final da monografia.

## Situação das tarefas conjuntas

| ID | Tarefa | Situação verificável | Próxima ação humana |
|---|---|---|---|
| `m2_ab_1` | Definir features Pix simuladas | **Concluída:** quatro features aprovadas em 16/08/2026, registradas e implementadas | Revisar somente se o experimento indicar necessidade de mudança |
| `m2_ab_2` | Entrega parcial 1: EDA + features | EDA executada e relatório reproduzido dos CSVs; envio ainda não comprovado | Anexar resultados do baseline da Pessoa 1 e registrar o envio ao orientador |

Evidências conjuntas:

- [`../../../config/pix_feature_registry.json`](../../../config/pix_feature_registry.json): fórmulas, janelas, nulos, controle de vazamento e advertências;
- [`../../../reports/reunioes/2026-08-16_mapeamento_ieee_cis_pix.md`](../../../reports/reunioes/2026-08-16_mapeamento_ieee_cis_pix.md): decisão confirmada por Letícia e Lucas;
- [`../../../src/features/pix_features.py`](../../../src/features/pix_features.py): implementação causal das quatro features;
- [`04_entrega_parcial_eda_features.md`](04_entrega_parcial_eda_features.md): estatísticas reais e pauta de decisão;
- [`../../../scripts/gerar_entrega_junho.py`](../../../scripts/gerar_entrega_junho.py): geração reproduzível com leitura seletiva dos CSVs.

`m2_ab_1` pode ser marcada como concluída porque há decisão conjunta e implementação versionadas. `m2_ab_2` permanece aberta até existirem baseline e registro do envio ao orientador.

## Principais entendimentos de junho

1. O IEEE-CIS é um proxy de comércio eletrônico/cartão, não um dataset Pix.
2. Uma chain é uma composição de etapas; no laboratório atual, recuperação e formatação do prompt são `Runnable`s encadeados.
3. Um retriever recebe uma consulta e devolve documentos; ele não precisa ser um banco vetorial e não gera a resposta.
4. Um prompt template padroniza entradas, contexto e restrições; ele não garante fidelidade sozinho.
5. SMOTE deve ficar dentro do treino. Aplicá-lo antes da divisão dos dados produz vazamento.
6. O texto metodológico descreve o protocolo planejado; métricas só entram no Capítulo 4 depois de experimentos executados.

## Como verificar

```powershell
.venv\Scripts\python.exe -m unittest tests.test_langchain_basics tests.test_june_deliverables -v
.venv\Scripts\python.exe -m src.rag.langchain_basics
.venv\Scripts\python.exe scripts\gerar_entrega_junho.py
```

## Critério de pronto

Uma tarefa de escrita precisa conter fontes rastreáveis, separar plano de resultado e declarar as limitações do domínio. Um estudo técnico precisa executar sem chave externa e ter teste. Uma decisão conjunta precisa de aprovação humana registrada; um documento preparado não substitui reunião ou envio ao orientador.
