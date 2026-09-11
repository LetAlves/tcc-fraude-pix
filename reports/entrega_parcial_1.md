# Entrega parcial 1 — EDA, features Pix simuladas e baseline de referência

**TCC:** Protótipo Híbrido para Detecção e Explicação de Fraudes em Transações Pix Simuladas Utilizando Machine Learning, SHAP e RAG
**Autores:** Letícia Alves (ML e dados) · Lucas Nogueira (RAG e escrita)
**Data:** 10/09/2026

---

## Declaração sobre os dados

Este trabalho **não utiliza dados reais do Pix** nem de qualquer instituição financeira. Os experimentos usam o dataset público IEEE-CIS Fraud Detection (Kaggle), que contém transações de comércio eletrônico com cartão. As variáveis inspiradas no Pix são **proxies analíticos documentados**, construídos a partir das colunas do IEEE-CIS, conforme a seção 4.1 da proposta aprovada. Os resultados aqui não comprovam desempenho operacional no Pix, e o protótipo não deve ser usado para bloquear transações ou acusar pessoas.

---

## 1. O que esta entrega cobre

| Item | Situação |
|---|---|
| Análise exploratória do dataset | Concluída |
| Features Pix simuladas (4 conceitos) | Implementadas e aprovadas pela dupla |
| Pré-processamento completo | Implementado |
| Divisão treino/validação/teste | Implementada e verificada |
| Baseline (Regressão Logística) treinado e avaliado | Concluído |

O escopo previsto para esta entrega era EDA e features. O baseline foi incluído porque já está concluído e porque produziu um resultado que muda a leitura do projeto (seção 5).

---

## 2. Dados

O IEEE-CIS traz duas tabelas, unidas pelo `TransactionID`:

| Medida | Valor |
|---|---|
| Transações | 590.540 |
| Fraudes marcadas | 20.663 (3,499%) |
| Razão legítima:fraude | 27,6 : 1 |
| Colunas após a junção | 434 |
| Cobertura da tabela de identidade | 24,4% das transações |
| Extensão relativa de `TransactionDT` | ~182 dias |

O desbalanceamento de 27,6:1 é o fato que organiza todas as decisões seguintes: um classificador que sempre responda "não é fraude" acerta 96,5% dos casos e é inútil. Por isso a acurácia não é reportada em nenhum momento deste trabalho.

Detalhamento completo da EDA, incluindo padrão de nulos e correlações, em `reports/pessoa_2/junho/04_entrega_parcial_eda_features.md` e `reports/eda_summary.txt`.

---

## 3. Features Pix simuladas

Quatro conceitos implementados em `src/features/pix_features.py`, aprovados pela dupla em 16/08/2026 (ata em `reports/reunioes/`), gerando seis colunas:

| Feature | Conceito da proposta | Ressalva obrigatória |
|---|---|---|
| `valor_atipico_proxy` | valor atípico / mudança brusca de comportamento | `card1` é identificador mascarado, não conta nem chave Pix |
| `frequencia_recente_proxy` | frequência de transações recentes | a janela usa o relógio relativo do dataset, não um dia civil |
| `dispositivo_raro_proxy` | dispositivo conhecido ou desconhecido | raridade estatística não equivale a dispositivo comprometido |
| `posicao_ciclo_diario_relativa` | horário da transação | origem e fuso de `TransactionDT` não são publicados |

Todas são **causais**: usam apenas transações anteriores à linha avaliada, sem consultar o alvo nem eventos futuros.

Três itens da proposta permanecem **sem feature correspondente**, por não terem equivalente no IEEE-CIS: tipo de chave Pix, perfil do destinatário e idade da conta. Serão declarados como limitação no Capítulo 4.

---

## 4. Pré-processamento e divisão dos dados

**Tratamento por grupo de coluna** (`src/features/preprocessor.py`): 406 numéricas recebem imputação por mediana e padronização; 11 categóricas de baixa cardinalidade viram one-hot; 18 de alta cardinalidade recebem codificação por frequência. Todo ajuste estatístico é aprendido **somente no conjunto de treino**.

`card4` (bandeira do cartão) e `card6` (crédito/débito) foram **excluídas**: são conceitos exclusivos do domínio de cartão, sem equivalente possível no Pix, e não constam da lista de variáveis da proposta aprovada.

**A divisão é temporal, não aleatória**: as transações mais antigas formam o treino, as mais recentes o teste. Em detecção de fraude, só se tem o passado para prever o futuro; um sorteio aleatório permitiria ao modelo aprender padrões de datas posteriores às que ele tenta prever.

A preocupação natural é que, sem estratificar, os conjuntos fiquem desbalanceados entre si. Verificamos no dataset completo que isso não ocorre:

| Conjunto | Transações | Fraudes | Taxa |
|---|---|---|---|
| Treino | 413.378 | 14.538 | 3,517% |
| Validação | 88.581 | 3.042 | 3,434% |
| Teste | 88.581 | 3.083 | 3,480% |

As três taxas ficam dentro de 0,08 ponto percentual. O realismo temporal foi obtido **sem custo de equilíbrio de classe**.

---

## 5. Resultados do baseline

Regressão Logística, limiar de decisão 0,5, dataset completo. Duas estratégias de desbalanceamento comparadas em pipelines separados:

**Validação** — 3.042 fraudes, taxa base 3,434%:

| Métrica | Ponderação de classe | SMOTE |
|---|---|---|
| AUC-ROC | 0,8414 | 0,8392 |
| AUC-PR | 0,3934 | 0,3967 |
| Recall | 0,6785 | 0,6831 |
| Precisão | 0,1347 | 0,1316 |
| F1 | 0,2248 | 0,2206 |
| Tempo de treino | 5min43s | 42min |

Como a AUC-PR de um classificador aleatório é igual à taxa base, **os dois modelos estão cerca de 11,5× acima do acaso**.

**As duas estratégias empataram.** A diferença de 0,0033 é menor que a incerteza esperada para 3.042 fraudes — e o sinal se inverte conforme o recorte: em uma amostra de 50.000 linhas, a ponderação de classe vencia pela mesma margem. A escolha entre elas será feita por critérios secundários (não fabricar dados sintéticos, não interpolar variáveis categóricas, custo computacional), com o critério declarado antes da avaliação final.

**Teste** — 3.083 fraudes, taxa base 3,480%:

| AUC-ROC | AUC-PR | Recall | Precisão | F1 |
|---|---|---|---|---|
| 0,8234 | 0,1840 | 0,7071 | 0,1213 | 0,2071 |

Em termos operacionais: das 3.083 fraudes do período de teste, o modelo recupera cerca de 2.180, marcando aproximadamente 18.000 das 88.581 transações como suspeitas — 20% do total, com 8 de cada 10 acusações sendo alarme falso. É o comportamento esperado de um baseline linear, e é o piso que os modelos de julho precisam superar.

---

## 6. Achado principal: degradação temporal

**A AUC-PR cai de 0,3967 na validação para 0,1840 no teste — menos da metade.** A queda não se explica por desbalanceamento:

- as taxas base dos dois conjuntos são praticamente iguais (3,434% e 3,480%);
- a AUC-ROC quase não se move (0,8392 → 0,8234), ou seja, o modelo continua ordenando;
- o recall inclusive sobe (0,6831 → 0,7071).

O que se degrada é a **pureza das previsões de maior confiança** no período mais recente. Os padrões aprendidos no passado envelhecem.

Esse resultado **só é observável por causa da divisão temporal**. Uma divisão aleatória teria misturado os períodos e reportado ~0,39 como desempenho do modelo — errando por um fator de dois. É evidência interna, produzida pelo próprio experimento, para a escolha metodológica adotada, e indica que um sistema como este exigiria retreino periódico em operação.

---

## 7. Limitações assumidas nesta etapa

1. **O limiar de decisão permanece em 0,5**, valor padrão e não escolhido. Ajustá-lo na validação é a próxima melhoria de menor custo.
2. **O SMOTE interpola colunas já codificadas em one-hot**, gerando linhas com categorias fracionárias que não existem no domínio. O `SMOTENC` trataria dado misto, mas exigiria reordenar as etapas do pipeline.
3. **A execução não é reprodutível bit a bit.** Para caber na memória disponível, o caminho de modelagem usa `float32`; combinado com operações matriciais paralelas, isso faz os resultados variarem a partir da terceira casa decimal entre execuções. As conclusões não dependem dessa casa, mas números citados devem vir de uma execução identificada.
4. **Três variáveis da proposta não têm equivalente no dataset**: tipo de chave Pix, perfil do destinatário e idade da conta.
5. **O dataset é de cartão, não de Pix.** Nenhum resultado aqui pode ser lido como desempenho no Pix real.

---

## 8. Observação sobre a base regulatória

A proposta aprovada cita a **Resolução BCB nº 403/2023** como base regulatória de prevenção a fraudes no Pix. Ao levantar as fontes para a base documental do RAG, identificamos que o **Mecanismo Especial de Devolução (MED) foi criado pela Resolução BCB nº 103/2021**; a Resolução nº 403 é de 22/07/2024 e trata de outro ajuste. A referência será corrigida na monografia.

---

## 9. Próxima etapa

Conforme a proposta aprovada, julho prevê:

- treinar **XGBoost** (modelo principal) e **Random Forest** (comparativo);
- implementar o módulo de avaliação com todas as métricas, matriz de confusão e curva de precisão-recall;
- aplicar **SHAP** sobre o modelo principal, para identificar as variáveis mais influentes por transação.

Os modelos serão avaliados no **mesmo corte temporal**, para que a queda entre validação e teste seja comparável com a do baseline. O critério de sucesso é melhorar a **precisão sem perder recall**.

---

## 10. Ponto para alinhamento

<!-- Letícia: ajuste este parágrafo com o que você quiser combinar sobre o calendário. -->

Esta entrega está sendo enviada em setembro, e não no prazo originalmente previsto. Gostaríamos de alinhar com o senhor o calendário das etapas seguintes antes de avançar, para que as próximas entregas tenham prazos realistas.

---

**Artefatos de referência:** `notebooks/01_eda.ipynb` (EDA), `notebooks/02_preprocessing.ipynb` (pré-processamento e baseline, com resultados), `src/features/pix_features.py` (features), `src/features/preprocessor.py` (pré-processamento), `reports/anotacoes_metodologia.md` (registro de decisões metodológicas).
