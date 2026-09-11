# Validação investigativa das features Pix simuladas

Data da validação: **30/08/2026**. Escopo: tarefa conjunta `m2_ab_1` — “Definir quais features Pix simuladas fazer: mapear colunas IEEE-CIS → conceitos Pix”.

## Decisão apoiada

Definir quais sinais podem ser derivados do IEEE-CIS para uma prova de conceito relacionada ao contexto Pix, sem apresentar colunas do domínio de comércio eletrônico/cartão como campos reais do Pix.

## Veredito

**ACHADO — confiança alta:** as quatro features registradas são tecnicamente defensáveis como **proxies analíticos**, desde que seus nomes, tags e explicações preservem as advertências de domínio. Nenhuma delas constitui equivalência semântica com conta, chave, usuário, dispositivo cadastrado ou horário real do Pix.

Features mantidas:

1. `valor_atipico_proxy`;
2. `frequencia_recente_proxy`;
3. `dispositivo_raro_proxy`;
4. `posicao_ciclo_diario_relativa`.

O registro oficial da decisão está em `config/pix_feature_registry.json` e a implementação em `src/features/pix_features.py`.

## Evidências primárias e conexões

### IEEE-CIS

A [descrição oficial da competição no Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) afirma que o alvo é a probabilidade de fraude em uma transação online, separa os dados em arquivos de transação e identidade, informa que nem toda transação possui identidade e define `TransactionDT` como um delta em relação a uma referência, não como timestamp real. Também lista `card1`–`card6`, `DeviceType` e `DeviceInfo`, mas não autoriza interpretá-los como entidades do Pix.

### Segurança do Pix

A página oficial [Segurança no Pix](https://www.bcb.gov.br/estabilidadefinanceira/pix-seguranca) descreve análise de risco, marcações de fraude e controles para dispositivos cadastrados. A página de [limites de valor para transações Pix](https://www.bcb.gov.br/meubc/faqs/p/limites-de-valor-para-as-transacoes-pix) registra que limites podem depender do perfil de risco e de comportamento e variar por período. Essas fontes sustentam a relevância conceitual de valor, comportamento, dispositivo e tempo no domínio Pix; elas não provam que as colunas do IEEE-CIS representem esses campos.

### Literatura de fraude transacional

Jurgovsky et al. (2018), [Sequence classification for credit-card fraud detection](https://doi.org/10.1016/j.eswa.2018.01.037), sustenta a utilidade de sequência e histórico de transações. Dal Pozzolo et al. (2018), [Credit Card Fraud Detection: A Realistic Modeling and a Novel Learning Strategy](https://doi.org/10.1109/TNNLS.2017.2736643), reforça a necessidade de avaliação temporal realista e cuidado com mudança de distribuição.

### SINAL FORTE

Três caminhos independentes convergem:

1. o dataset oferece valor, ordem relativa, identificador mascarado e informação parcial de dispositivo;
2. o Banco Central reconhece valor, perfil comportamental, período e dispositivo como elementos relevantes à segurança do Pix;
3. a literatura sustenta histórico e sequência como sinais úteis em fraude transacional.

A convergência apoia a criação de proxies. A diferença entre os domínios impede chamá-los de features Pix reais.

## Matriz final de decisão

| Feature | Papel analítico permitido | Evidência | Confiança | Limite obrigatório |
|---|---|---|---|---|
| `valor_atipico_proxy` | medir desvio robusto do valor diante do histórico anterior de `card1` | `TransactionAmt` oficial + relevância de valor/perfil no Pix + histórico na literatura | alta como proxy; média para transferência ao Pix | `card1` não é conta nem chave Pix |
| `frequencia_recente_proxy` | contar eventos anteriores em janela relativa | ordem de `TransactionDT` + evidência de sequência/histórico | alta como sinal transacional; média no Pix | não representa frequência real de uma conta Pix |
| `dispositivo_raro_proxy` | medir raridade histórica do par `card1`/`DeviceInfo` | `DeviceInfo` oficial + controle de dispositivos no Pix | média-alta como proxy | não equivale a dispositivo cadastrado, novo ou comprometido |
| `posicao_ciclo_diario_relativa` | representar periodicidade no relógio relativo | `TransactionDT` + limites Pix por período | média técnica; baixa interpretação Pix | não pode ser chamada de hora local, período noturno ou horário Pix |

## Mapeamentos rejeitados

| Mapeamento proposto | Decisão | Motivo |
|---|---|---|
| `card1` → conta ou chave Pix | rejeitado | identificador mascarado do domínio original |
| `addr*` → pagador ou recebedor Pix | rejeitado | descrição oficial não fornece essa equivalência |
| `P_emaildomain`/`R_emaildomain` → chave Pix de e-mail | rejeitado | são atributos do comércio eletrônico, não chaves Pix documentadas |
| `ProductCD` → tipo de transação Pix | rejeitado | código de produto do domínio original |
| `C*`, `D*`, `M*`, `V*`, `id_*` → conceitos Pix individuais | rejeitado | significado individual total ou parcialmente anonimizado |
| `TransactionDT mod 86400` → horário local | rejeitado | origem, data e fuso não são publicados |

## Controles contra vazamento

- estatísticas históricas usam apenas eventos com tempo estritamente menor;
- eventos no mesmo `TransactionDT` não são usados como histórico uns dos outros;
- a janela de frequência usa o intervalo aberto `(t - janela, t)`;
- linhas sem `card1` não formam um falso grupo de histórico;
- nenhuma feature consulta `isFraud`;
- SMOTE, se usado, deve permanecer somente no pipeline de treino.

## Correções produzidas durante a validação

A investigação executável revelou que a frequência falhava quando `card1` estava ausente e que empates de `TransactionDT` podiam introduzir histórico não estritamente anterior. Foram aplicadas as seguintes correções:

- frequência retorna `NaN` quando o identificador ou o tempo está ausente;
- frequência exclui o limite da janela e todos os eventos do mesmo instante;
- valor atípico usa a mesma estatística prévia para linhas empatadas no tempo;
- raridade do dispositivo exige `card1`, `DeviceInfo` e tempo válidos;
- raridade exclui ocorrências do mesmo instante;
- cinco testes específicos foram adicionados.

O smoke test processou **50.000 linhas reais** do IEEE-CIS e produziu um DataFrame com 440 colunas, incluindo as seis saídas derivadas, sem erro.

A suíte completa do projeto foi executada após as correções: **30 testes aprovados**, incluindo os cinco novos testes das features.

## Critério de pronto

A tarefa pode ser considerada concluída quando:

- [x] as quatro features possuem fórmula, fontes, população, janela e tratamento de nulos;
- [x] cada feature possui advertência de domínio;
- [x] mapeamentos semanticamente indevidos estão explicitamente rejeitados;
- [x] o cálculo é causal e possui testes para nulos, empates e limites temporais;
- [x] a implementação executa sobre amostra real;
- [x] a decisão da dupla está registrada em ata.

## Limites

- A validação confirma coerência e implementação; não confirma ganho preditivo.
- A utilidade de cada feature deve ser medida no baseline e em análise de ablação.
- Desempenho no IEEE-CIS não demonstra generalização para transações Pix reais.
- Uma futura base Pix permitiria substituir proxies por atributos definidos pelo domínio e pela instituição.
