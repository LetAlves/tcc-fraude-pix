# Entrega parcial 1 — EDA e features Pix simuladas

Data de geração: **18/08/2026**.  
Fonte dos números: CSVs locais do treino IEEE-CIS, lidos por `scripts/gerar_entrega_junho.py`.

## Estado da entrega

Este documento consolida a **EDA executada e as features aprovadas pela dupla em 16/08/2026**. O IEEE-CIS não contém transações Pix: os atributos abaixo são proxies analíticos, implementados em `src/features/pix_features.py`, e não equivalências de campos Pix reais. O envio desta entrega parcial ao orientador ainda deve ser registrado.

## Evidência reproduzida dos dados

| Medida | Resultado |
|---|---:|
| `train_transaction.csv` | 590.540 linhas × 394 colunas |
| `train_identity.csv` | 144.233 linhas × 41 colunas |
| transações legítimas | 569.877 |
| fraudes marcadas | 20.663 |
| prevalência de `isFraud=1` | 3.499% |
| razão legítima:fraude | 27.6:1 |
| cobertura de identidade no treino | 24.424% |
| extensão relativa de `TransactionDT` | 182.0 dias |
| valores distintos de `card1` | 13.553 |
| valores distintos não nulos de `DeviceInfo` | 1.786 |

`TransactionAmt` apresenta mediana de **68.50** nas linhas legítimas e **75.00** nas linhas marcadas como fraude. Essa diferença descritiva não estabelece causalidade nem define sozinha um limiar de risco.

## Implicações para o baseline

1. A prevalência próxima de 3,5% torna acurácia inadequada como métrica principal; AUC-PR, precisão, recall, F1 e matriz de confusão devem ser relatados.
2. SMOTE, se comparado, deve ser ajustado **somente nos folds/partição de treino** por meio de pipeline do `imbalanced-learn`.
3. `TransactionDT` preserva ordem relativa, mas não revela data, fuso ou hora civil; a avaliação deve respeitar a ordem sem chamar o ciclo de “horário Pix”.
4. A identidade cobre apenas parte das transações; ausência de `DeviceInfo` precisa ser tratada explicitamente e não pode ser interpretada como fraude.
5. Estatísticas históricas por `card1` devem usar apenas eventos anteriores. Cálculo global antes da divisão de treino/validação/teste causaria vazamento.

## Protocolo temporal aprovado

A avaliação principal usa corte temporal 70/15/15, com as transações mais antigas no treino e as mais recentes no teste. As fronteiras são deslocadas quando necessário para manter todas as transações com o mesmo `TransactionDT` na mesma partição.

| Conjunto | Linhas | Fraudes | Taxa de fraude |
|---|---:|---:|---:|
| treino | 413.378 | 14.538 | 3.517% |
| validação | 88.581 | 3.042 | 3.434% |
| teste | 88.581 | 3.083 | 3.480% |

A diferença máxima entre as taxas é de **0.083 ponto percentual**. Isso representa proporções semelhantes entre as partições, não balanceamento entre as classes: fraude continua sendo rara. Ocorrências repetidas de `TransactionDT`, desconsiderando a primeira de cada valor, correspondem a **2.9%** das linhas; ao contar todas as linhas pertencentes a grupos de timestamps repetidos, a proporção é **5.7%**. Fronteiras sem timestamps compartilhados: **sim**. Qualquer divisão aleatória estratificada será apenas uma análise complementar.

## Registro aprovado de features

Status do registro: `approved_and_implemented`.

| ID | Conceito controlado | Colunas-fonte | Advertência obrigatória |
|---|---|---|---|
| `valor_atipico_proxy` | desvio robusto do valor no histórico proxy | `TransactionAmt`, `card1`, `TransactionDT` | card1 é um identificador mascarado do domínio original, não uma conta ou chave Pix. |
| `frequencia_recente_proxy` | frequência recente no identificador proxy | `card1`, `TransactionDT` | A janela usa o relógio relativo do dataset e não representa necessariamente um dia civil Pix. |
| `dispositivo_raro_proxy` | raridade do dispositivo no histórico proxy | `card1`, `DeviceInfo`, `TransactionDT` | Raridade estatística não equivale a dispositivo comprometido, novo ou associado a fraude. |
| `posicao_ciclo_diario_relativa` | posição cíclica temporal relativa | `TransactionDT` | A origem e o fuso de TransactionDT não são publicados; a feature não representa horário local. |

O arquivo-fonte completo, incluindo fórmula, janela, tratamento de nulos, controle de vazamento e tags do RAG, está em `config/pix_feature_registry.json`.

## O que esta entrega não afirma

- não afirma que `card1` é conta, chave ou destinatário Pix;
- não converte `TransactionDT` em horário local;
- não transforma raridade de dispositivo em “dispositivo comprometido”;
- não usa colunas `C*`, `D*`, `M*`, `V*` ou `id_*` como conceitos Pix individuais;
- não considera o baseline concluído antes da execução dos modelos, publicação das métricas e revisão da dupla.

## Pendências para concluir a entrega ao orientador

- [ ] decidir comparação entre peso de classe e SMOTE sem vazamento;
- [x] definir split temporal 70/15/15, preservando empates de `TransactionDT`;
- [ ] registrar seeds do experimento;
- [ ] executar e revisar o baseline da Pessoa 1;
- [ ] registrar a data de apresentação desta entrega ao orientador.
