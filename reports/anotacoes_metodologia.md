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

### LangChain

O laboratório de junho usa `Document`, um retriever lexical e composição por `Runnable` com `PromptTemplate`. Ele não chama LLM e não é o RAG final. Seu objetivo é validar as interfaces e as restrições antes da inclusão de embeddings e FAISS.

### Frase-chave para o Capítulo 3

O protocolo separa três evidências: o classificador estima risco; SHAP registra influências locais; o RAG recupera contexto documental. A geração em linguagem natural deve permanecer subordinada às duas evidências e se abster quando o suporte for insuficiente.
