# Anotações da Metodologia — TCC Fraude Pix

Registro compartilhado de entendimentos importantes discutidos ao longo do cronograma.
Objetivo: virar base direta de parágrafos do TCC (principalmente Capítulo 3 — Metodologia) sem precisar reconstruir o raciocínio depois. Preencher juntos (Letícia e Lucas) conforme as tarefas do [cronograma](https://lunogueira-67.github.io/tcc-organizador-tarefas/) forem concluídas.

---

## Maio — Fundação do projeto

### Tarefa: Ler a proposta e entender a metodologia (m1_p1_0a)

- **SHAP explica o quê**: aponta quais variáveis da transação mais influenciaram a decisão do modelo. É a explicação técnica, específica de cada transação.
- **RAG contextualiza com fontes**: recupera trechos regulatórios, científicos e setoriais relacionados a conceitos com semântica validada e os fornece ao LLM para redigir uma explicação rastreável. O RAG não descobre o significado de uma coluna anônima.
- **Frase-chave para o Capítulo 3**: SHAP = *o que influenciou o modelo*; RAG = *qual contexto documental verificável ajuda a interpretar a evidência*. A ponte exige um registro de atributos semânticos; nomes anônimos como `V258` não devem virar consultas Pix por suposição.

### Tarefa: Estudar o contexto Pix — golpes típicos (m1_p1_0e)

- **Golpe do falso funcionário**: fraudador se passa por atendente do banco e convence a vítima a fazer o Pix ou a passar a senha/código. A vítima pode iniciar a transação sob manipulação. O Pix não possui chargeback automático equivalente ao cartão; casos de fundada suspeita de fraude podem seguir o **MED (Mecanismo Especial de Devolução)**, introduzido pela Resolução BCB nº 103/2021. Procedimentos e prazos devem ser citados a partir do Regulamento do Pix e do Guia do MED vigentes na data analisada.
- **Engenharia social (categoria ampla)**: inclui golpe do parente/emergência, falso vendedor/comprador, QR Code adulterado e falsas promoções ou investimentos. Esses cenários motivam atributos comportamentais explícitos, como desvio de valor e frequência recente. Os grupos `C*` e `D*` podem carregar sinais estatísticos, mas seu significado individual não foi divulgado e não prova um tipo de golpe.
- **Clonagem de dados/conta**: envolve acesso não autorizado, por exemplo após phishing, malware ou comprometimento de credenciais. `DeviceType` e `DeviceInfo` podem apoiar uma análise de contexto do dispositivo quando presentes; não é válido afirmar que cada `id_*` ou `M*` mede dispositivo novo, conta antiga ou identidade Pix.

**Por que importa pro TCC**: orienta hipóteses para atributos comportamentais e a seleção do corpus. Os documentos do BCB sustentam regras e procedimentos; o relatório FEBRABAN fornece contexto setorial. Nenhum deles valida sozinho a correspondência entre uma coluna anônima do IEEE-CIS e um padrão Pix.

### Tarefa: Entender as duas tabelas do dataset IEEE-CIS (m1_p1_0b)

- **`train_transaction.csv`** — 394 colunas, ~590k linhas. Uma linha = uma transação. Principais: `TransactionID` (chave), `isFraud` (alvo, 0/1 — fortemente desbalanceado, ~3,5% fraude), `TransactionDT` (segundos desde um ponto de referência arbitrário, não é data real), `TransactionAmt`, `ProductCD`, `card1-card6` (dados do cartão mascarados), grupos `C1-C14`/`D1-D15`/`M1-M9`/`V1-V339`.
- **`train_identity.csv`** — 41 colunas, bem menor (nem toda transação tem linha de identidade). Chave `TransactionID` faz o *left join* com a tabela principal. Colunas: `id_01-id_38` (anônimas, rede/dispositivo), `DeviceType` (mobile/desktop), `DeviceInfo` (modelo/SO — confirmado no preview: pode vir `NaN` mesmo com `DeviceType` preenchido, ou seja, dado incompleto é esperado e precisa ser tratado no pré-processamento).
- Preview real rodado localmente confirmou a estrutura: linhas de `train_transaction` majoritariamente `isFraud=0`; join por `TransactionID` funciona como esperado entre as duas tabelas.

### Tarefa: Estudar os grupos de colunas C/D/M/V (m1_p1_0c)

Contagem real confirmada no `train_transaction.csv`: **C1–C14** (14 colunas), **D1–D15** (15), **M1–M9** (9), **V1–V339** (339 — ~86% de todas as 394 colunas). Todas anonimizadas de propósito pela Vesta, sem dicionário oficial — só a categoria geral é conhecida:

- **C1–C14 (contagens)**: a categoria geral é de contagens. O evento contado por cada coluna não foi publicado; exemplos encontrados na comunidade são hipóteses, não dicionário oficial.
- **D1–D15 (deltas de tempo)**: a categoria geral é de deltas temporais. Não é seguro atribuir a cada coluna um evento como "abertura da conta" sem fonte.
- **M1–M9 (matches)**: indicadores de correspondência do domínio original, com significado individual não divulgado. Não equivalem automaticamente a conferências de identidade Pix.
- **V1–V339 (features Vesta)**: atributos numéricos engenheirados e anonimizados. O modelo e o SHAP indicam influência estatística, mas não recuperam o significado oculto nem estabelecem causalidade.

**Por que importa pro TCC**: sem significado semântico, o pré-processamento deve ser estatístico e documentado. O SHAP mostra quais colunas pesaram na decisão; a explicação em linguagem natural só pode usar conceitos definidos por atributos explícitos ou derivados validados.

### Revisão da Pessoa 2 — correções e ponte SHAP → RAG (m1_p2_0a–0d)

- O MED foi introduzido pela **Resolução BCB nº 103/2021**. A Resolução nº 403 é de 22/07/2024 e não criou o mecanismo.
- No treino local, `train_identity.csv` cobre **144.233 de 590.540 transações (24,4%)**, e não aproximadamente 60%.
- O IEEE-CIS é um dataset de comércio eletrônico/cartão usado como proxy técnico; resultados não comprovam desempenho em Pix real.
- A investigação, referências e regras para o corpus estão em `reports/pessoa_2/maio/`.

### Tarefa: Entender as features de identidade (m1_p1_0d)

- **`id_01` a `id_11`**: atributos numéricos anonimizados. A descrição oficial agrupa as features de identidade, rede e assinatura digital, mas não revela o significado individual de cada `id_*`.
- **`id_12` a `id_38`**: atributos categóricos ou códigos categóricos anonimizados. Alguns valores observáveis sugerem browser ou resolução, porém não é seguro atribuir uma definição individual sem fonte oficial.
- **`DeviceType`**: mobile ou desktop.
- **`DeviceInfo`**: texto livre (modelo/SO), alta cardinalidade — precisa de limpeza/agrupamento antes de virar feature categórica.

**Números reais do `train_identity.csv`** (144.233 linhas):
- `DeviceType`: 85.165 desktop / 55.645 mobile / 3.423 nulos (~2,4%).
- `DeviceInfo`: 1.786 valores distintos; top 5 = Windows (47.722), nulo (25.567 = ~17,7%), iOS Device (19.782), MacOS (12.573), Trident/7.0 = IE11 (7.440).
- Das 38 colunas `id_`, a taxa de nulos varia muito: `id_01` tem 0% nulo, enquanto `id_07`/`id_08` têm **96,4% de nulos**. Alta ausência não prova inutilidade; a presença do dado pode carregar sinal e precisa ser avaliada.

**Por que importa pro TCC**: colunas com alta ausência são candidatas a comparação entre descarte, indicador de presença e tratamento compatível com o modelo. A decisão deve ser tomada por validação dentro do treino, sem limiar arbitrário e sem consultar o conjunto de teste.

### Tarefa: Rodar o EDA e documentar achados (m1_p1_3 / m1_p1_4)

- **Desbalanceamento confirmado no dataset completo**: 590.540 transações, 20.663 fraudes (3,50%), razão 27,6:1. Justifica a escolha de F1/AUC-PR sobre acurácia (já registrado nas dicas do cronograma) e a necessidade de SMOTE/ajuste de peso de classe em junho.
- **Nulos concentrados nas colunas `id_*`**: `id_24`, `id_25`, `id_07`, `id_08`, `id_21`, `id_26`, `id_27`, `id_23`, `id_22` — todas com ~99% de nulo no dataset *merged* (bem acima dos ~96,4% vistos isoladamente em `train_identity.csv`, porque o merge herda os ~75,6% de transações sem linha de identidade). `dist2` também aparece no topo, com 93,63% de nulo.
- **Top correlações de Pearson com `isFraud`**: dominadas pelo grupo `V*` — `V257` (0,383), `V246` (0,367), `V244` (0,364), `V242` (0,361), `V201`, `V200`, `V189`, `V188`, `V258`, `V45`. Como o significado individual dessas colunas não é público, o SHAP em julho vai ser o que permite falar sobre *influência* sem inventar semântica — a correlação aqui só orienta prioridade de investigação, não causa.
- Evidência completa em `reports/eda_summary.txt`, `reports/fig_distribuicao_alvo.png` e `reports/fig_top_correlacoes.png` (gerados por `notebooks/01_eda.ipynb`, execução de 16/08/2026).

**Por que importa pro TCC**: esses três achados (desbalanceamento, padrão de nulos ligado à ausência de identidade, e concentração de sinal no grupo V) formam a ponte direta entre o Capítulo 3 (Metodologia — por que SMOTE, por que AUC-PR) e o Capítulo 4 (Resultados — de onde vêm as features mais fortes do modelo).

---

## Junho — Dataset proxy, baseline planejado e desenho das três camadas

### Evidência reproduzida para a entrega parcial

- `train_transaction.csv`: 590.540 linhas, 394 colunas e 20.663 fraudes marcadas (3,499%);
- razão entre linhas legítimas e fraude: 27,6:1;
- `train_identity.csv`: 144.233 linhas e cobertura de 24,424% das transações;
- `TransactionDT`: aproximadamente 182 dias de extensão relativa, sem data civil publicada;
- mediana de `TransactionAmt`: 68,50 nas linhas legítimas e 75,00 nas linhas marcadas como fraude; diferença descritiva, não causal.

### Decisão metodológica e registro aprovado

- usar corte temporal 70/15/15 como avaliação principal, com as transações mais antigas no treino e as mais recentes no teste;
- manter todas as transações com o mesmo `TransactionDT` na mesma partição e registrar qualquer análise estratificada aleatória apenas como complemento;
- ajustar toda transformação e reamostragem somente no treino;
- comparar ponderação de classe e SMOTE em pipelines separados;
- escolher limiar na validação, nunca no teste;
- usar somente features do registro versionado como ponte semântica SHAP→RAG;
- manter somente as quatro features aprovadas pela dupla em 16/08/2026 como ponte semântica: valor atípico, frequência recente, dispositivo raro e ciclo diário relativo, todas explicitamente qualificadas como proxies do IEEE-CIS;
- usar a implementação causal de `src/features/pix_features.py`, sem consultar alvo ou eventos futuros.

### Validação do corte temporal no dataset completo

| Conjunto | Linhas | Fraudes | Taxa de fraude |
|---|---:|---:|---:|
| treino | 413.378 | 14.538 | 3,517% |
| validação | 88.581 | 3.042 | 3,434% |
| teste | 88.581 | 3.083 | 3,480% |

As proporções de fraude diferem no máximo 0,083 ponto percentual entre as partições. Isso não torna o problema balanceado — a classe de fraude permanece rara —, mas mostra que o corte temporal preservou proporções semelhantes sem consultar o alvo para estratificar. Ocorrências repetidas de `TransactionDT`, desconsiderando a primeira de cada valor, correspondem a 2,9% das linhas; ao contar todas as linhas pertencentes a grupos de timestamps repetidos, a proporção é 5,7%. Nenhuma ocorrência do mesmo timestamp atravessa as fronteiras entre treino, validação e teste.

### Decisão: excluir `card4` e `card6` do pré-processamento

- **`card1`–`card6`** são "payment card information" (descrição oficial da Vesta). Dessas, `card1`, `card2`, `card3` e `card5` são numéricas mascaradas, sem significado individual revelado — entram na mesma categoria dos grupos anônimos (`C`/`D`/`M`/`V`), utilizáveis como sinal estatístico com a ressalva de "papel analítico análogo".
- **`card4`** (bandeira: visa/mastercard/amex/discover) e **`card6`** (crédito/débito) são diferentes: o significado é conhecido, e é **especificamente e exclusivamente do domínio de cartão** — bandeira e tipo crédito/débito não têm conceito equivalente no Pix (que não tem bandeira nem distinção crédito/débito). Diferente de `TransactionAmt` (valor) ou `TransactionDT` (tempo), que são conceitos genéricos de pagamento válidos por analogia, `card4`/`card6` não têm analogia possível — são artefatos do produto "cartão", não do "pagamento" em geral.
- **Confirmação pela proposta aprovada**: a proposta de TCC aprovada pelo orientador (seção 4.1 "Dataset Experimental") lista explicitamente as features que devem inspirar a adaptação Pix: valor, horário, frequência de transações recentes, tipo de chave Pix, perfil do destinatário, idade da conta, dispositivo conhecido/desconhecido. Bandeira/tipo de cartão não consta nessa lista — `card4`/`card6` estão fora do escopo de features aprovado, não é só uma escolha de conveniência.

**Decisão**: `card4` e `card6` serão excluídos das features usadas no `preprocessor.py`/baseline de junho. `card1`, `card2`, `card3`, `card5` permanecem como candidatos válidos (sinal estatístico anônimo).

**Por que importa pro TCC**: dá uma justificativa metodológica defensável pro Capítulo 3 sobre por que colunas de cartão explícitas foram descartadas, coerente com a proposta aprovada e com a regra de "papel analítico análogo" já validada com o Lucas.

### Decisão: manter `C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339` no baseline

Considerou-se excluir os 4 grupos anônimos por serem "muita coluna com dado que não sabemos o que é". Decisão: **manter todos por enquanto**, porque o critério de exclusão do `card4`/`card6` não se aplica aqui.

- Significado desconhecido não é o mesmo problema que significado conhecido sem equivalente em Pix (caso `card4`/`card6`). Colunas anônimas não fazem nenhuma afirmação semântica errada — só carregam sinal estatístico sem narrativa.
- Evidência concreta contra excluir: as maiores correlações com `isFraud` encontradas no EDA de maio são todas do grupo `V` (`V257`=0,383; `V246`=0,367; `V244`=0,364; `V242`=0,361...). Excluir `V1`–`V339` removeria o sinal preditivo mais forte já identificado no dataset.
- É exatamente para esse cenário que a Camada 2 (SHAP) existe na arquitetura aprovada: aponta quais variáveis anônimas pesaram na decisão sem exigir que se saiba o que elas significam. A regra de escrita já registrada (maio, m1_p1_0c) continua valendo: usar essas colunas no modelo é permitido; **narrar significado para elas não é**.

**Plano**: treinar o baseline com todas; em julho, usar a importância do SHAP para decidir se vale reduzir o conjunto (ex.: manter só as top-N mais influentes) — corte guiado por evidência do próprio modelo, não por desconforto com a quantidade de colunas.

### Decisão: ponderação de classe como estratégia principal de desbalanceamento (m2_p1_3)

Execução de 05/09/2026, `notebooks/02_preprocessing.ipynb`, **amostra de teste de 50.000 transações** (`N_LINHAS = 50_000`), split temporal 35.000 / 7.500 / 7.500, taxa de fraude de 2,81% na validação — cerca de 210 fraudes. Regressão logística, limiar padrão de 0,5. **Números provisórios**: servem para escolher a estratégia, não como resultado do trabalho. Os valores oficiais saem da execução com o dataset completo (m2_p1_6).

| Métrica na validação | `class_weight='balanced'` | SMOTE |
|---|---|---|
| AUC-PR | **0,1395** | 0,1332 |
| AUC-ROC | 0,7496 | 0,7489 |
| Recall | 0,6209 | **0,6303** |
| F1 | 0,1509 | **0,1539** |
| Precisão (derivada de F1 e recall) | ~8,6% | ~8,8% |

**As duas estratégias empataram.** Pelo critério registrado (maior AUC-PR na validação) vence a ponderação de classe, mas a diferença é de 0,006 em AUC-PR sobre ~210 fraudes — dentro da incerteza esperada para esse número de positivos. A redação do relatório e da monografia não deve afirmar superioridade de uma sobre a outra; o que a evidência sustenta é equivalência de desempenho, com a escolha decidida por critérios secundários.

Os três critérios secundários apontam todos para a ponderação de classe:

- **não fabrica dado**: a ponderação altera apenas o custo do erro na função de perda, enquanto o SMOTE cria transações de fraude sintéticas que nunca ocorreram;
- **não distorce as categóricas**: o SMOTE roda depois do pré-processamento, quando as categóricas já viraram colunas one-hot, e interpola entre elas — produz linhas com `ProductCD_W = 0,37`, combinação que não existe no domínio. O `SMOTENC` trata dado misto, mas exigiria conhecer as colunas categóricas antes do encoding, invertendo a ordem das etapas do pipeline. Limitação declarada, não corrigida nesta etapa;
- **custo de memória**: o SMOTE equilibra as classes duplicando o conjunto de treino em memória, o que inviabiliza a execução com o dataset completo na máquina usada (8 GB de RAM).

Referência para leitura das métricas: com 2,81% de fraude na validação, um classificador aleatório teria AUC-PR ≈ 0,028. O baseline fica ~4,7× acima disso. A precisão de ~8,6% significa que, a cada 100 transações marcadas como fraude, cerca de 9 são fraude e 91 são alarme falso — com recall de 62%. Esse é o piso que XGBoost e Random Forest precisam superar em julho (m3_p1_1 / m3_p1_2), e a melhora esperada é em precisão sem perda de recall.

**Validade da comparação**: `montar_pipeline_modelo` clona o pré-processador recebido. Sem o clone, os dois pipelines compartilhariam o mesmo objeto e o ajuste do segundo sobrescreveria o do primeiro — os dois números da tabela acima descreveriam um estado que só um dos modelos de fato usou. Correção registrada no commit `ddda837`.

**Por que importa pro TCC**: dá ao Capítulo 3 uma justificativa de escolha que não depende de um decimal indefensável, e ao Capítulo 4 a leitura correta do baseline (alto recall, baixa precisão). Também é o registro de que o SMOTE foi de fato aplicado e avaliado, e não descartado por conveniência.

### Evidência: o corte temporal dispensa a estratificação (m2_p1_4)

Verificação de 05/09/2026 sobre o **dataset completo** (590.540 transações), lendo apenas `TransactionDT` e `isFraud` — 9 MB, sem carregar as 434 colunas.

| Conjunto | Linhas | Fraudes | Taxa |
|---|---|---|---|
| treino | 413.378 | 14.538 | 3,517% |
| validação | 88.581 | 3.042 | 3,434% |
| teste | 88.581 | 3.083 | 3,480% |

O cronograma descrevia esta tarefa como "split estratificado (70/15/15)", enquanto o implementado é corte temporal, conforme a decisão metodológica registrada acima. A verificação mostra que **não houve troca**: neste dataset, as proporções 70/15/15 são exatas e as três taxas de fraude ficam dentro de 0,08 ponto percentual entre si. O corte cronológico produziu conjuntos balanceados sem estratificar, então o realismo temporal foi obtido sem custo de equilíbrio de classe. O texto da tarefa no organizador foi atualizado para refletir a metodologia e a garantia de fronteira por instante.

**Integridade da fronteira**: 5,7% das linhas do dataset (33.932) pertencem a grupos com `TransactionDT` repetido, o que abriria a possibilidade de um bloco de transações simultâneas ser partido entre dois conjuntos. Nos dois cortes efetivos (`TransactionDT` 10.437.996 e 13.151.840), nenhuma transação do lado direito repete o último timestamp do lado esquerdo — zero vazamento de instante entre treino, validação e teste.

> **Correção (12/09/2026).** Estas notas diziam "2,9% das linhas compartilham timestamp com outra transação". Os 2,9% (17.191 linhas) são as *repetições*, excluindo a primeira ocorrência de cada valor; as linhas que de fato pertencem a um grupo de timestamp repetido são 5,7% (33.932). A frase descrevia o segundo número e citava o primeiro. Erro apontado pela revisão do Lucas no PR #5.

**Ordenação estável**: `dividir_temporal` passou a ordenar com `kind="mergesort"`. Como o IEEE-CIS já chega ordenado por `TransactionDT`, o pandas detecta que a ordem pedida é a existente e não reordena nada — a mudança **não altera nenhum resultado atual**. Ela protege o caso de entrada fora de ordem (amostra embaralhada, arquivos concatenados): entre linhas de mesmo timestamp o mergesort preserva a ordem de chegada, enquanto o quicksort padrão a reorganiza de forma arbitrária e dependente da versão do numpy.

**Alerta sobre números de amostra**: na execução de teste com 50.000 linhas, as taxas de fraude saíram 2,87% / 2,81% / 1,91%. O 1,91% do teste é artefato do recorte — 50.000 linhas cobrem cerca de 9 dias iniciais, e a fraude não se distribui uniformemente nesse intervalo. No dataset completo a distorção desaparece. Reforça que nenhuma métrica de amostra deve ser citada como resultado.

**Onde não está o gargalo de memória**: medido com `psutil` em um DataFrame de 538 MB (150.000 linhas × 440 colunas), `dividir_temporal` custa +8 MB de RSS. A divisão não é a origem do consumo que ameaça a execução com o dataset completo; procurar nas cópias criadas no notebook (`X_treino = treino.drop(...)`) e na duplicação do conjunto de treino pelo SMOTE.

**Por que importa pro TCC**: sustenta no Capítulo 3 a escolha do corte temporal com evidência própria, em vez de apenas citar a literatura, e antecipa a pergunta óbvia de banca — "sem estratificar, os conjuntos não ficam desbalanceados?" — com número medido.

### Execução oficial do baseline no dataset completo (m2_p1_5)

Execução de 05/09/2026 sobre as **590.540 transações**, via script dedicado (não pelo notebook — a m2_p1_6 segue pendente). Regressão logística, limiar 0,5, corte temporal 70/15/15. Duração 49 minutos, pico de 4.394 MB de memória. Ambas as estratégias convergiram com `max_iter=1000`.

**Validação** — 88.581 transações, 3.042 fraudes, taxa base 3,434%:

| Métrica | ponderação de classe | SMOTE |
|---|---|---|
| AUC-ROC | 0,8414 | 0,8393 |
| AUC-PR | 0,3936 | **0,3962** |
| AUC-PR ÷ taxa base | 11,46× | 11,54× |
| Recall | 0,6792 | 0,6834 |
| Precisão | **0,1349** | 0,1314 |
| F1 | 0,2250 | 0,2204 |
| Tempo de treino | **5min43s** | 42min |

**Teste** — 88.581 transações, 3.083 fraudes, taxa base 3,480%, pipeline SMOTE:

| AUC-ROC | AUC-PR | AUC-PR ÷ taxa base | Recall | Precisão | F1 |
|---|---|---|---|---|---|
| 0,8239 | 0,1863 | 5,35× | 0,7074 | 0,1215 | 0,2074 |

#### Achado principal: degradação temporal

A AUC-PR cai de **0,3962 na validação para 0,1863 no teste** — menos da metade — enquanto as taxas base são praticamente iguais (3,434% e 3,480%), a AUC-ROC quase não se move (0,839 para 0,824) e o recall até sobe (0,683 para 0,707). A queda não é efeito de desbalanceamento: é perda de pureza nas previsões de maior confiança no período mais recente. Os padrões aprendidos no passado envelhecem.

O achado só é observável por causa do corte temporal — afirmação agora verificada por controle complementar, registrado em seção própria abaixo. É evidência interna, do próprio experimento, para a escolha metodológica do Capítulo 3, e limitação a declarar no Capítulo 5: um protótipo assim exigiria retreino periódico.

Em termos operacionais no teste: das 3.083 fraudes o modelo recupera ~2.181, marcando cerca de 18.000 das 88.581 transações como suspeitas — 20% do total, com 8 de cada 10 acusações sendo falso alarme.

#### A inversão confirma o empate

Na amostra de 50.000 linhas a ponderação de classe vencia por 0,006 de AUC-PR; no dataset completo o SMOTE vence por 0,0026. **As duas margens são mínimas e apontam para lados opostos** — confirmação empírica da leitura já registrada de que as estratégias empatam e que qualquer vencedor nessa casa decimal é ruído.

Dado novo desta escala: o SMOTE levou **7,4× mais tempo** de treino para entregar desempenho equivalente, o que reforça os critérios secundários já registrados (não fabricar dado sintético, não interpolar colunas one-hot).

#### Decisão em aberto — regra de seleção da estratégia

A nota anterior registrou a ponderação de classe como estratégia principal, por critérios secundários. O script de execução usa a regra automática (maior AUC-PR na validação), que no dataset completo seleciona o SMOTE — por isso a avaliação de teste acima é a do pipeline SMOTE. **As duas decisões precisam ser conciliadas em uma redação única antes do Capítulo 4.**

Restrição a respeitar na conciliação: avaliar também a ponderação de classe no teste e então escolher o melhor dos dois **contaminaria o conjunto de teste** — a seleção tem de ser feita na validação, com critério declarado antes de olhar o teste. Reportar os dois números de teste é aceitável desde que o texto declare qual critério de seleção valia previamente.

#### Alterações que viabilizaram a execução

A tentativa anterior falhou com `ArrayMemoryError` ao pedir 1,76 GiB para um bloco de 399 × 590.540 em `float64`. Duas correções, ambas verificadas como neutras em resultado:

- `frequencia_recente_proxy` ordenava e copiava o DataFrame inteiro (434 colunas) para usar duas. Passou a ordenar apenas as colunas-fonte: saída idêntica (`equals` verdadeiro, diferença máxima 0,0) com o custo caindo de 66 MB para 4 MB em 200.000 linhas;
- `reduzir_precisao` converte `float64` para `float32` após a engenharia de features, e o `OneHotEncoder` e o `CodificadorFrequencia` deixaram de emitir `float64`. Em 300.000 linhas: pico de 3.686 MB para 2.603 MB (menos 29%), tempo 15% menor, métricas diferindo apenas na quarta casa decimal (AUC-PR 0,4472 contra 0,4476; precisão idêntica).

**Nota sobre comparação entre tamanhos de amostra**: a AUC-PR medida em 50.000, 150.000 e 300.000 linhas (0,1395 / 0,2881 / 0,4472) não forma uma curva de aprendizado interpretável, porque cada tamanho usa uma janela de validação diferente, com dificuldade e taxa base próprias. Apenas os números do dataset completo devem ser citados.

**Por que importa pro TCC**: fixa os valores oficiais do baseline de junho — o piso que XGBoost e Random Forest precisam superar em julho (m3_p1_1 / m3_p1_2), com a melhora esperada em precisão sem perda de recall — e entrega o primeiro resultado experimental próprio do trabalho, a degradação temporal.

### Entregável executado e números canônicos (m2_p1_6)

Execução de 06/09/2026 do `notebooks/02_preprocessing.ipynb` no dataset completo, via `nbconvert --execute --inplace`, com as saídas gravadas no próprio notebook. Nove células de código, zero erros. **Estes são os números a citar na monografia** — a execução via script registrada na seção anterior (m2_p1_5) fica como verificação cruzada, não como fonte.

**Validação** — 88.581 transações, 3.042 fraudes, taxa base 3,434%:

| Métrica | ponderação de classe | SMOTE |
|---|---|---|
| AUC-ROC | 0,8414 | 0,8392 |
| AUC-PR | 0,3934 | **0,3967** |
| AUC-PR ÷ taxa base | 11,5× | 11,6× |
| Recall | 0,6785 | 0,6831 |
| Precisão | **0,1347** | 0,1316 |
| F1 | 0,2248 | 0,2206 |

**Teste** — 88.581 transações, 3.083 fraudes, taxa base 3,480%, pipeline SMOTE:

| AUC-ROC | AUC-PR | AUC-PR ÷ taxa base | Recall | Precisão | F1 |
|---|---|---|---|---|---|
| 0,8234 | 0,1840 | 5,3× | 0,7071 | 0,1213 | 0,2071 |

Todas as conclusões da seção anterior se mantêm: empate entre as estratégias (agora 0,0033 de diferença, ainda com inversão de sinal em relação à amostra) e queda da AUC-PR pela metade entre validação e teste, comportamento compatível com degradação temporal.

#### Limitação de reprodutibilidade

As duas execuções do mesmo código, com a mesma semente, divergem a partir da terceira casa decimal:

| | script (m2_p1_5) | notebook (m2_p1_6) |
|---|---|---|
| AUC-PR validação, ponderação | 0,3936 | 0,3934 |
| AUC-PR validação, SMOTE | 0,3962 | 0,3967 |
| AUC-PR teste | 0,1863 | 0,1840 |

A causa é a combinação de `float32` com as operações matriciais paralelas do BLAS: a ordem das somas varia entre execuções e o resultado muda nos últimos dígitos. A divergência foi introduzida pela redução de precisão adotada em m2_p1_5 — em `float64` o efeito seria menor, mas a execução completa não caberia na máquina disponível. É um custo assumido conscientemente, e a contrapartida é que **nenhum número citado pode misturar as duas execuções**; a fonte tem de ser identificada.

Observação relacionada: as métricas de limiar (recall, F1, precisão) são mais sensíveis a isso do que as de ordenação (AUC-ROC, AUC-PR), porque pequenas variações de arredondamento fazem transações cruzarem o corte de 0,5. Em teste com 50.000 linhas, o recall variou 0,0142 entre `float64` e `float32`, contra 0,0010 na AUC-PR.

#### Custo de execução e condições de medição

O notebook levou **1h25**, contra 49 minutos do script equivalente. O código é o mesmo; a diferença é pressão de memória — com o kernel do Jupyter e o restante do ambiente carregados, a máquina passou a paginar para o disco. A engenharia de features levou 17 minutos no notebook contra cerca de 40 segundos no script, e a própria redução de precisão levou 4,7 minutos contra tempo desprezível. Qualquer tempo de execução citado na monografia precisa declarar em que condições foi medido, sob pena de descrever a memória disponível da máquina em vez do custo do método.

**Por que importa pro TCC**: fixa a fonte única dos números de junho e documenta duas limitações que só aparecem em execução real — a não reprodutibilidade bit a bit introduzida pelo `float32` e a dependência do tempo de execução em relação à memória livre. As duas são candidatas a menção no Capítulo 5.

### Controle complementar: corte temporal contra divisão aleatória (m2_p1_5)

Execução de 10/09/2026 no dataset completo. Mesmo modelo (regressão logística, `class_weight='balanced'`, `max_iter=1000`, semente 42) e mesmo pré-processamento nos dois braços; muda apenas a forma de dividir os dados. Duração 23 minutos, pico de 3.489 MB. Ambos convergiram.

**Motivação**: as notas anteriores afirmavam que uma divisão aleatória "teria reportado ~0,39, errando por um fator de dois". Isso era inferência, não medição — nenhum split aleatório havia sido executado. A revisão do Lucas (10/09/2026) apontou corretamente o problema. Este experimento é o complemento aleatório que a decisão metodológica de junho já previa ("usar corte temporal como avaliação principal e registrar qualquer análise estratificada aleatória apenas como complemento") e que não tinha sido feito.

| | Corte temporal | Divisão aleatória estratificada |
|---|---|---|
| AUC-PR validação | 0,3936 | 0,4220 |
| **AUC-PR teste** | **0,1858** | **0,4250** |
| Variação validação → teste | **−52,8%** | +0,7% |
| AUC-ROC teste | 0,8289 | 0,8604 |
| Recall teste | 0,7032 | 0,7289 |
| Precisão teste | 0,1236 | 0,1374 |

**Resultado**: a divisão aleatória reporta **2,3× a AUC-PR** do corte temporal no conjunto de teste (0,4250 contra 0,1858) e **não mostra degradação alguma** — validação e teste praticamente idênticos. Descreveria um modelo estável e cerca de duas vezes melhor do que ele de fato é no período mais recente.

A inferência original estava na direção certa e **subestimava** o efeito: o fator medido é 2,3 e não 2, e o valor aleatório (0,4250) supera até a validação temporal. Mas a correção do Lucas procede: afirmação sem medição não é resultado, e a redação anterior não podia ficar.

**Ressalva obrigatória na redação**: a divisão aleatória coloca transações do **mesmo `card1`** no treino e no teste simultaneamente, além de misturar os períodos. A inflação observada combina os dois vazamentos — período compartilhado e identificador compartilhado — e **não pode ser atribuída inteiramente ao tempo**. Separar as duas contribuições exigiria um terceiro braço com divisão por grupo (`GroupShuffleSplit` por `card1`), não executado nesta etapa.

**Número novo relevante**: este experimento produziu também o teste do corte temporal com ponderação de classe — AUC-PR **0,1858**, contra 0,1840 do SMOTE. O empate entre as duas estratégias, já observado na validação, **se mantém no teste** (diferença de 0,0018). Adotar a ponderação de classe pelos critérios secundários não custa desempenho, e os dois números de teste podem ser reportados lado a lado sem contaminar a seleção.

**Por que importa pro TCC**: transforma a justificativa do corte temporal de argumento citado em evidência medida, com o número do contrafactual. É o parágrafo mais forte disponível para o Capítulo 3.

### Garantia de fronteira por instante no corte temporal (m2_p1_4)

Revisão do Lucas (10/09/2026) observou que o cronograma passou a descrever o corte temporal como "mantendo transações com o mesmo timestamp no mesmo conjunto", garantia que `dividir_temporal` não oferecia: o corte era feito por posição. A verificação de 05/09 mostrava que, no 70/15/15 do IEEE-CIS, nenhum bloco de empate era atravessado — mas isso era propriedade daquelas frações com aqueles dados, não do algoritmo.

A função passa a empurrar cada fronteira até a próxima mudança de instante, com três consequências registradas:

- **as frações tornam-se aproximadas**, com desvio máximo limitado ao tamanho do bloco de empate atravessado. No IEEE-CIS a 70/15/15 o desvio é zero (ver abaixo); em um caso construído com blocos irregulares, 70/15/15 resulta em 75/15/10;
- **três períodos não vazios deixam de ser garantidos** quando há poucos instantes distintos. O caso degenerado (todas as linhas no mesmo instante) agora levanta `ValueError` com mensagem explicativa, em vez de devolver conjuntos vazios silenciosamente;
- **a ordem interna de cada conjunto depende da ordem de entrada**, porque o mergesort é estável em relação a ela. O que a função garante é quais linhas caem em cada conjunto, não a ordem dentro deles.

**Impacto nos resultados já medidos: nenhum.** Verificado no dataset completo após a mudança — o split continua 413.378 / 88.581 / 88.581, com zero instantes compartilhados entre conjuntos e as mesmas taxas de fraude (3,517% / 3,434% / 3,480%). Todos os números de junho seguem válidos; a garantia passou de coincidência a propriedade.

Cobertura em `tests/test_split_temporal.py`, treze testes em `unittest` (o projeto não tem `pytest` instalado na venv): empates nas duas fronteiras, duas fronteiras no mesmo bloco, preservação de todas as linhas sem duplicação, ordem cronológica entre conjuntos, estabilidade frente a entrada desordenada, proporções aproximadas, e casos de erro para poucos instantes, frações inválidas, coluna ausente, dataset vazio e timestamp ausente. Um teste verifica a própria premissa do cenário, confirmando que o corte ingênuo por posição de fato partiria um bloco.

**Por que importa pro TCC**: a monografia pode afirmar que nenhuma transação simultânea foi dividida entre treino, validação e teste, e apontar o teste que garante isso — em vez de depender de uma verificação pontual que valeria só para aquele recorte.

### Reconfirmação dos resultados sobre as features corrigidas (m2_p1_5)

Em 11/09/2026 a branch `feat/preprocessor-junho` foi atualizada com a `main`, trazendo dois merges do Lucas feitos em paralelo: `c5d0ee3` (validação das features Pix) e `156d643` (corpus RAG). O primeiro **corrige um vazamento causal em `src/features/pix_features.py`**: transações do mesmo instante usavam uma à outra como histórico. Todos os números de junho haviam sido produzidos antes dessa correção.

**Resolução do conflito**: o merge conflitou em `pix_features.py`, onde as duas linhas de trabalho mexeram na mesma função. Prevaleceu a versão da `main`. Ela corrige o vazamento e, além disso, já resolve por outro caminho o consumo de memória atacado pelo commit `d8ef6c9` desta branch — seleciona apenas as colunas-fonte (`df.loc[validos, [card1, TransactionDT]]`) e conta por grupo com `searchsorted`, em vez de `groupby().rolling()` sobre uma coluna de data sintética. O commit `d8ef6c9` fica superado.

**População afetada pelo vazamento**: 312 linhas em 590.540 (0,053%), contendo 24 fraudes. `card1` não tem valores ausentes neste dataset, então a mudança de 0 para `NaN` em identificador ausente não altera nada aqui.

**Reexecução do controle sobre as features corrigidas:**

| AUC-PR | Antes | Depois | Δ |
|---|---|---|---|
| Temporal, validação | 0,3936 | 0,3933 | −0,0003 |
| Temporal, teste | 0,1858 | 0,1860 | +0,0002 |
| Aleatório, validação | 0,4220 | 0,4213 | −0,0007 |
| Aleatório, teste | 0,4250 | 0,4243 | −0,0007 |

Todas as diferenças ficam em 0,0007 ou menos, dentro da faixa de ruído já documentada para `float32` entre execuções. As conclusões não se movem: fator aleatório/temporal de **2,28×** (antes 2,29×) e queda de **−52,7%** entre validação e teste (antes −52,8%). O pico total medido subiu de 3.489 MB para 3.836 MB, portanto não se atribui redução ao processo inteiro; isoladamente, a preparação das features consumiu cerca de 70 MB a menos com a implementação por grupo.

**Testes diretamente relacionados após a reconciliação**: 18 testes passam — os 13 de `tests/test_split_temporal.py` e os 5 de `tests/test_pix_features.py`, escritos pelo Lucas para as features. O split temporal e as features corrigidas convivem sem regressão em nenhuma direção.

**Pendência registrada**: o `notebooks/02_preprocessing.ipynb`, declarado fonte canônica em m2_p1_6, foi executado **antes** da correção das features. Os valores nele diferem dos atuais na quarta casa decimal. Reexecutá-lo alinha o entregável ao código final; enquanto isso não acontece, esta nota é o registro de que a diferença é conhecida e medida.

**Por que importa pro TCC**: documenta que uma correção de causalidade feita depois dos experimentos foi verificada, e não presumida inofensiva. O tamanho da população afetada (0,053%) explica por que o efeito é nulo na prática, sem que isso sirva de desculpa para não medir.

### Números canônicos definitivos de junho (m2_p1_6)

Execução de 11/09/2026 do `notebooks/02_preprocessing.ipynb` no dataset completo, já sobre as features corrigidas e usando `src/models/evaluator.py`. Duração 45 minutos, zero erros, onze células de código. **Estes substituem todos os valores registrados antes** — as execuções anteriores usavam as features com o vazamento causal e uma função de avaliação definida dentro do notebook.

| Métrica | Validação (ponderação) | Validação (SMOTE) | Teste (ponderação) | Teste (SMOTE) |
|---|---|---|---|---|
| AUC-PR | 0,3930 | 0,3962 | **0,1850** | 0,1852 |
| AUC-ROC | 0,8410 | 0,8391 | 0,8286 | 0,8238 |
| Recall | 0,6772 | 0,6831 | 0,7026 | 0,7078 |
| Precisão | 0,1350 | 0,1315 | 0,1237 | 0,1216 |
| F1 | 0,2251 | 0,2205 | 0,2103 | 0,2076 |

Leitura operacional no teste, com a estratégia adotada: 2.166 fraudes recuperadas de 3.083, 917 perdidas, 17.513 transações marcadas (19,8% do total) das quais 15.347 são alarme falso.

**O empate foi medido quatro vezes, com o vencedor alternando:** amostra de 50.000 (ponderação por 0,006), dataset completo na validação (SMOTE por 0,003), teste do controle (ponderação por 0,002) e agora o teste do notebook (SMOTE por 0,0002). A margem sempre menor que o ruído entre execuções encerra a discussão de desempenho: **a decisão pela ponderação de classe se sustenta exclusivamente nos critérios secundários**, e a seção 8 do notebook passou a declarar o critério em vez de selecionar pelo maior AUC-PR.

#### Três cópias do DataFrame eliminadas

A execução abortou duas vezes com `MemoryError` pedindo 1,76 GiB para um bloco de 399 × 590.540 em `float64`. Três pontos distintos operavam o DataFrame inteiro para usar poucas colunas, cada um encoberto pelo anterior:

| Ponto | Efeito medido |
|---|---|
| `valor_atipico_proxy` ordenava 434 colunas por (card1, tempo) | 15 min → 18 s |
| `reduzir_precisao` atribuía coluna a coluna, forçando reconsolidação | 4,7 min → 1,1 s |
| `criar_features_pix` usava `pd.concat` para acrescentar 6 colunas | destravou a execução |

As três correções foram verificadas como neutras em resultado: saída idêntica em 200.000 linhas reais, com zero linhas divergentes, e os 30 testes da suíte passando. A execução caiu de 1h25 para 45 minutos.

**Regra prática que resume os três casos, e que vale para julho**: selecionar as colunas **antes** de ordenar, copiar ou transformar, nunca depois. Um bloco `float64` de 399 colunas × 590 mil linhas exige 1,76 GiB em alocação única — se não houver esse espaço contíguo, aborta, por mais memória total que a máquina tenha.

**Por que importa pro TCC**: fixa a fonte única dos números de junho, produzida pelo mesmo código de avaliação que julho vai usar, e documenta o padrão de consumo que limita o que é viável executar na máquina disponível.

## Julho — Modelos principais e explicabilidade

### Random Forest comparativo: duas configurações (m3_p1_2)

Execução de 11/09/2026 no conjunto de treino completo (413.378 × 460), 100 árvores, `class_weight='balanced'` — o mesmo tratamento de desbalanceamento da regressão logística e do `scale_pos_weight` do XGBoost, para que a comparação meça o algoritmo e não a estratégia de desbalanceamento. Sem busca de hiperparâmetros: a proposta aprovada define o Random Forest como modelo comparativo, e isso precisa estar dito no texto para que a comparação com um XGBoost ajustado não pareça enviesada.

| Configuração | AUC-PR | AUC-ROC | Recall | Precisão | F1 | Tempo | Profundidade média | Folhas/árvore |
|---|---|---|---|---|---|---|---|---|
| Árvores completas | **0,5298** | **0,9078** | 0,3192 | **0,8018** | 0,4566 | 176 s | 64,3 | 16.430 |
| Profundidade ≤ 20 | 0,4573 | 0,8751 | 0,4757 | 0,3988 | 0,4339 | 140 s | 20,0 | 4.965 |

#### Por que duas configurações, e a correção de uma conclusão precipitada

A medição preliminar de custo usou **10 árvores** e mostrou AUC-PR praticamente idêntica entre as duas configurações (0,4189 contra 0,4187), o que levou à conclusão de que a profundidade extra não comprava desempenho. **Com 100 árvores a conclusão se inverte**: a diferença é de 0,0725 a favor das árvores completas.

O motivo está no mecanismo do próprio algoritmo. Árvores profundas decoram individualmente, e é a média de muitas delas que converte capacidade em generalização. Com 10 árvores a média não tem de quem tirar e as duas configurações empatam por baixo; com 100, a floresta profunda aproveita a capacidade. **Extrapolar de 10 para 100 falhou porque a propriedade em questão depende justamente do número de árvores.**

Registro da lição: extrapolação linear vale para custo (tempo e memória cresceram como previsto), não para qualidade.

#### O XGBoost só se separa depois do ajuste

| Modelo (validação) | AUC-PR |
|---|---|
| Regressão logística (baseline de junho) | 0,3930 |
| Random Forest, profundidade ≤ 20 | 0,4573 |
| XGBoost padrão | 0,5293 |
| **Random Forest, árvores completas** | **0,5298** |
| XGBoost, melhor tentativa do Optuna até agora | 0,5794 |

O Random Forest **empata com o XGBoost sem ajuste**. A monografia não pode afirmar superioridade do XGBoost sobre o comparativo de forma genérica: o que a evidência sustenta é que o XGBoost se separa **depois da busca de hiperparâmetros**.

#### Pontos de operação incomparáveis no limiar padrão

No limiar 0,5, o Random Forest com árvores completas opera em **precisão 0,80 com recall 0,32**; o XGBoost padrão, em **precisão 0,27 com recall 0,67**. Marcam quantidades muito diferentes de transações e acertam em proporções muito diferentes.

Comparar F1, precisão ou recall entre modelos nesse limiar compara **calibração de probabilidade**, não qualidade de modelo — cada modelo distribui suas probabilidades de forma própria, e o corte de 0,5 cai em lugares distintos de cada curva. A comparação legítima é a AUC-PR, que independe do limiar. **Isso precisa estar escrito no Capítulo 4**, senão a tabela sugere que o Random Forest tem precisão muito superior à do XGBoost, o que é falso.

Em termos de trabalho, os dois pontos de operação são produtos diferentes: "acusa pouco e quase sempre acerta" contra "acusa muito e recupera mais fraude". A escolha entre eles é decisão de negócio, não de métrica, e cabe como discussão no Capítulo 5.

**Por que importa pro TCC**: entrega o modelo comparativo exigido pela proposta com número próprio, e produz duas ressalvas que sustentam a honestidade da tabela comparativa — o empate com o XGBoost não ajustado e a incomparabilidade das métricas de limiar.

### Registro de versões do ambiente

Criado `requirements-lock.txt` em 12/09/2026, com as versões exatas do ambiente que produziu todos os resultados registrados até aqui. As principais:

| Biblioteca | Versão |
|---|---|
| Python | 3.13.9 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| xgboost | 3.4.1 |
| imbalanced-learn | 0.14.2 |
| optuna | 4.9.0 |
| shap | 0.52.0 |

**Efeito sobre a limitação de reprodutibilidade já declarada.** Havia duas fontes de variação: a ordem das somas em `float32` com operações matriciais paralelas, que é inerente e permanece, e o desconhecimento de quais versões produziram os números, que este arquivo elimina. A redação do Capítulo 5 deve distinguir as duas — dizer apenas "não é reprodutível" é mais fraco e menos preciso do que a situação real.

Correção registrada de passagem: o README afirmava que o ambiente fora validado com Python 3.12; o interpretador em uso é 3.13.9.

**Por que importa pro TCC**: permite que a banca, ou qualquer pessoa, reconstrua o ambiente exato. Sem isso, "os resultados não reproduzem" fica sem explicação possível.

### LangChain

O laboratório de junho usa `Document`, um retriever lexical e composição por `Runnable` com `PromptTemplate`. Ele não chama LLM e não é o RAG final. Seu objetivo é validar as interfaces e as restrições antes da inclusão de embeddings e FAISS.

### Frase-chave para o Capítulo 3

O protocolo separa três evidências: o classificador estima risco; SHAP registra influências locais; o RAG recupera contexto documental. A geração em linguagem natural deve permanecer subordinada às duas evidências e se abster quando o suporte for insuficiente.
