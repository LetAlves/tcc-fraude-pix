# Anotações da Metodologia — TCC Fraude Pix

Registro compartilhado de entendimentos importantes discutidos ao longo do cronograma.
Objetivo: virar base direta de parágrafos do TCC (principalmente Capítulo 3 — Metodologia) sem precisar reconstruir o raciocínio depois. Preencher juntos (Letícia e Lucas) conforme as tarefas do [cronograma](https://letalves.github.io/tcc-fraude-pix/) forem concluídas.

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

- usar corte temporal como avaliação principal e registrar qualquer análise estratificada aleatória apenas como complemento;
- ajustar toda transformação e reamostragem somente no treino;
- comparar ponderação de classe e SMOTE em pipelines separados;
- escolher limiar na validação, nunca no teste;
- usar somente features do registro versionado como ponte semântica SHAP→RAG;
- manter somente as quatro features aprovadas pela dupla em 16/08/2026 como ponte semântica: valor atípico, frequência recente, dispositivo raro e ciclo diário relativo, todas explicitamente qualificadas como proxies do IEEE-CIS;
- usar a implementação causal de `src/features/pix_features.py`, sem consultar alvo ou eventos futuros.

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

O cronograma descreve esta tarefa como "split estratificado (70/15/15)", enquanto o implementado é corte temporal, conforme a decisão metodológica registrada acima. A verificação mostra que **não houve troca**: as proporções 70/15/15 são exatas e as três taxas de fraude ficam dentro de 0,08 ponto percentual entre si. O corte cronológico produziu conjuntos balanceados sem estratificar, então o realismo temporal foi obtido sem custo de equilíbrio de classe. O texto da tarefa no organizador está desatualizado em relação à metodologia, não o código.

**Integridade da fronteira**: 2,9% das linhas do dataset compartilham `TransactionDT` com outra transação. O `dividir_temporal` passou a avançar qualquer fronteira que caia dentro de um grupo empatado até a próxima mudança de timestamp. Assim, eventos simultâneos não são separados. Como consequência, as proporções são aproximadamente 70/15/15 quando existe empate na fronteira. Nos dois cortes efetivos do dataset completo (`TransactionDT` 10.437.996 e 13.151.840), não foi necessário deslocar as posições e as contagens originais foram preservadas.

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

O contraste só é observável porque o corte temporal preserva períodos distintos. Um split aleatório misturaria os períodos e poderia mascarar parte da degradação, mas esse controle não foi executado no dataset completo; por isso, não se atribui a ele uma AUC-PR específica. É evidência interna para a escolha metodológica do Capítulo 3 e uma limitação a declarar no Capítulo 5: um protótipo operacional exigiria monitoramento temporal e critérios de retreino.

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

### LangChain

O laboratório de junho usa `Document`, um retriever lexical e composição por `Runnable` com `PromptTemplate`. Ele não chama LLM e não é o RAG final. Seu objetivo é validar as interfaces e as restrições antes da inclusão de embeddings e FAISS.

### Frase-chave para o Capítulo 3

O protocolo separa três evidências: o classificador estima risco; SHAP registra influências locais; o RAG recupera contexto documental. A geração em linguagem natural deve permanecer subordinada às duas evidências e se abster quando o suporte for insuficiente.
