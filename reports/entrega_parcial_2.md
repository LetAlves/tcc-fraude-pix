# Entrega parcial 2 Modelos de classificação e explicabilidade

**TCC:** Protótipo Híbrido para Detecção e Explicação de Fraudes em Transações Pix Simuladas Utilizando Machine Learning, SHAP e RAG

**Autores:** Letícia Alves (ML e dados) · Lucas Nogueira (RAG e escrita)

**Data da consolidação:** 20 de setembro de 2026

> **Sobre os dados:** este trabalho **não utiliza dados reais do Pix** nem de instituições financeiras. Os experimentos usam o dataset público IEEE-CIS Fraud Detection (Kaggle), de transações com cartão. As variáveis inspiradas no Pix são proxies analíticos documentados, conforme a seção 4.1 da proposta aprovada.

## 1. O que esta entrega cobre

Os dois modelos previstos na proposta — **XGBoost** como principal e **Random Forest** como comparativo — treinados sobre as mesmas 590.540 transações, com o mesmo pré-processamento e o mesmo corte temporal do baseline de junho, e a aplicação do **SHAP** sobre o modelo escolhido.

Todos os modelos usam o mesmo tratamento de desbalanceamento (ponderação de classe, adotada em junho) e são medidos pelo mesmo código, `src/models/evaluator.py`. Isso é o que permite atribuir as diferenças ao algoritmo, e não à forma de medir.

O artefato XGBoost entregue, que também produziu o relatório SHAP, alcançou AUC-PR 0,5719 na validação e 0,4833 no teste. A avaliação final reutilizou o modelo e o pré-processador persistidos. O limiar 0,6834220290 foi escolhido por maximização de F1 exclusivamente na validação, antes de medir o teste.

## 2. Comparação entre modelos — validação

Conjunto de validação: 88.581 transações, 3.042 fraudes, taxa base 3,434%. A AUC-PR de um classificador aleatório equivale à taxa base, e é a régua de leitura.

| Modelo | AUC-PR | AUC-ROC | Tempo de treino |
|---|---|---|---|
| Regressão logística (baseline de junho) | 0,3930 | 0,8410 | 6 min |
| Random Forest, profundidade ≤ 20 | 0,4573 | 0,8751 | 140 s |
| XGBoost, hiperparâmetros padrão | 0,5293 | 0,9044 | 56 s |
| Random Forest, árvores completas | 0,5298 | 0,9078 | 176 s |
| **XGBoost entregue e usado no SHAP** | **0,5719** | **0,9067** | Não medido separadamente¹ |

¹ Os 30 s registrados anteriormente pertencem ao retreino histórico do tuning, não à geração deste artefato. A busca histórica com 50 tentativas levou 829 s; o fluxo completo levou 902 s e atingiu pico de 5.266 MB. Não se atribui esse tempo ao artefato atual.

O resultado 0,5703 permanece como evidência histórica em `reports/tuning_xgboost.json`. O modelo reconstruído com os parâmetros selecionados para compatibilidade com SHAP obteve 0,5719 e passou a ser o artefato entregue. A tabela identifica essa execução; os números históricos não foram apagados nem reinterpretados como novas medições.

**Dois pontos que a tabela já sustenta:**

O **Random Forest empata com o XGBoost sem ajuste** (0,5298 contra 0,5293). Não cabe afirmar superioridade genérica do modelo principal sobre o comparativo; o que a evidência mostra é que a diferença, se existir, vem da busca de hiperparâmetros.

**Ambos superam o baseline com folga**, o que justifica a complexidade adicional em relação a um modelo linear — o critério que havíamos declarado para esta etapa.

## 3. Metodologia da busca de hiperparâmetros

Seis decisões, registradas em `reports/anotacoes_metodologia.md`:

1. **A métrica otimizada é a AUC-PR na validação.** Otimizar AUC-ROC premiaria ordenar bem os 96,5% de casos fáceis; acurácia premiaria não detectar nada.
2. **O parâmetro de desbalanceamento não entra na busca.** Fica fixo na razão real entre classes (27,43), que é a forma do XGBoost expressar a ponderação de classe adotada em junho. Deixá-lo variar transformaria uma decisão metodológica já comparada em mais um hiperparâmetro.
3. **O pré-processamento é ajustado uma vez, fora da busca.** Não depende dos hiperparâmetros, e é ajustado somente no treino — validação e teste apenas recebem a transformação.
4. **O conjunto de teste não é tocado durante a busca.** A seleção acontece na validação; o teste é avaliado uma única vez, ao final, com o modelo já escolhido.
5. **A busca usa uma amostra temporal do treino.** Cada tentativa usa as 150 mil transações mais recentes do treino, com a validação completa. O candidato alcançou AUC-PR 0,5912 nessa condição. O artefato entregue, treinado anteriormente no treino completo com os parâmetros selecionados e XGBoost 3.0.5, apresenta AUC-PR 0,5719. O retreino histórico do tuning registrou 0,5703 com XGBoost 3.4.1. São execuções distintas; a avaliação final usa apenas o artefato entregue.
6. **O espaço e a poda foram reduzidos por viabilidade.** O teto passou de 600 para 300 árvores e a profundidade máxima de 10 para 8. Um `MedianPruner` interrompeu 25 das 50 tentativas após aquecimento. A melhor configuração atingiu os dois tetos, portanto não se afirma que seja um ótimo global.

Os melhores parâmetros foram `learning_rate=0,2308`, `max_depth=8`, `min_child_weight=14,5371`, `subsample=0,8712`, `colsample_bytree=0,9159`, `gamma=1,6854`, `reg_alpha=0,1130` e `reg_lambda=1,2327`, com 300 árvores e `scale_pos_weight=27,4343` fixo.

## 4. Explicabilidade — SHAP

O XGBoost final foi explicado com `TreeExplainer`, usando
`feature_perturbation="interventional"`, saída em escala de probabilidade e um
fundo de 500 linhas amostradas exclusivamente do treino (seed 42). A análise
global usou 1.000 transações aleatórias da validação.

As cinco variáveis de maior média absoluta de contribuição foram
`TransactionAmt`, `C13`, `C1`, `C14` e `card1`. Duas proxies com semântica
documentada também aparecem cedo no ranking: `valor_atipico_proxy` em 7º lugar
e `frequencia_recente_proxy` em 13º. Esse resultado indica influência no
comportamento do modelo; **não prova causalidade** e não revela o significado
oculto das colunas anônimas.

Foram produzidos force plots para um verdadeiro positivo, um falso positivo,
um falso negativo e um verdadeiro negativo, escolhidos pela probabilidade
mediana de cada quadrante. A fidelidade numérica foi verificada em todas as
linhas explicadas: o erro máximo em `valor base + soma das contribuições =
probabilidade prevista` foi 3,04 × 10^-7, abaixo da tolerância 10^-5. O relatório,
o summary plot e os quatro casos estão em `reports/pessoa_1/julho/shap/`.

Os casos SHAP históricos foram selecionados com limiar 0,5. A escolha posterior do limiar F1 não altera as contribuições, mas pode alterar a classificação de casos locais; por isso, seus rótulos VP, FP, FN e VN não são reapresentados como se tivessem sido calculados com o limiar final.

## 5. Avaliação final no teste

A avaliação final foi executada em 18/09/2026, sobre as 88.581 transações do período de teste, com 3.083 fraudes (3,4804%). O modelo e o pré-processador foram carregados por `src.models.persistencia.carregar`, com verificação de hashes. Não houve retreino ou novo ajuste de pré-processamento.

| Métrica | Validação no limiar F1 | Teste com limiar congelado |
|---|---|---|
| Limiar | 0,6834220290 | 0,6834220290 |
| AUC-PR (average precision) | 0,5719 | 0,4833 |
| AUC-ROC | 0,9067 | 0,8700 |
| Precisão | 0,6680 | 0,5852 |
| Recall | 0,4829 | 0,4324 |
| F1 | 0,5606 | 0,4973 |

Matriz de confusão no teste, com linhas representando a classe real e colunas representando a classificação do modelo:

| Classe real | Prevista legítima | Prevista fraude |
|---|---|---|
| Legítima | 84.553 | 945 |
| Fraude | 1.750 | 1.333 |

O modelo sinalizou 2.278 transações e recuperou 1.333 das 3.083 fraudes rotuladas. A AUC-PR caiu aproximadamente 0,0887 entre validação e teste, evidenciando degradação no período posterior. Esses resultados não sustentam desempenho equivalente em Pix real.

Os valores completos, versões, hashes do dataset e hashes do artefato estão em `reports/avaliacao_xgboost_entregue.json`. O limiar foi salvo em `models/xgboost/politica_decisao.json` antes da inferência no teste. O script `scripts/avaliar_xgboost_entregue.py` recusa sobrescrever uma avaliação já registrada.

## 6. Ressalvas metodológicas

**As métricas de decisão não são comparáveis entre modelos no limiar padrão.** No corte de 0,5, os modelos operam em pontos muito distintos da curva:

| Modelo | Recall | Precisão |
|---|---|---|
| Regressão logística | 0,68 | 0,14 |
| XGBoost padrão | 0,67 | 0,27 |
| Random Forest, árvores completas | 0,32 | 0,80 |

Cada modelo distribui suas probabilidades de forma própria, e o mesmo 0,5 cai em lugares diferentes de cada curva. Comparar precisão, recall ou F1 nesse ponto compara **calibração**, não qualidade. A comparação legítima entre modelos é a AUC-PR, que independe do limiar.

**Critério adotado: maximização de F1 na validação.** A dupla decidiu usar `criterio="f1"`, pois restrições de precisão mínima ou recall mínimo exigiriam alvos numéricos sem justificativa operacional disponível. A função `escolher_limiar` selecionou 0,6834220290184021; em empate de F1, a implementação escolhe o menor limiar. Essa escolha não usa os rótulos do teste.

O quadro acima preserva os pontos de operação históricos no corte 0,5. Aplicar 0,683422 aos outros modelos não remove diferenças de calibração e não equivale a selecionar F1 para cada modelo. Não foi calculada uma nova comparação nesse corte: os artefatos ou vetores de probabilidades da regressão logística e do Random Forest não estavam disponíveis nesta execução. Essa comparação permanece uma limitação explícita; a comparação principal da seção 2 usa AUC-PR, independente de limiar.

**Profundidade no Random Forest.** Com árvores completas, a floresta chega a profundidade média 64 e 16.430 folhas por árvore — memorização de transações individuais. Limitar a profundidade a 20 reduz o custo, mas **custa 0,07 de AUC-PR** (0,4573 contra 0,5298). A capacidade extra é aproveitada pela média das 100 árvores, e por isso foi mantida.

Uma medição preliminar com 10 árvores havia indicado empate entre as duas configurações. A conclusão não se sustentou com 100: extrapolação linear vale para custo, não para qualidade.

## 7. Limitações assumidas

- **O dataset é de cartão, não de Pix** — nenhum resultado indica desempenho no Pix real.
- **Três variáveis da proposta não têm equivalente no IEEE-CIS**: tipo de chave Pix, perfil do destinatário e idade da conta.
- **O Random Forest não passou por busca de hiperparâmetros**, conforme seu papel de modelo comparativo na proposta. A comparação com o XGBoost ajustado precisa ser lida com essa assimetria em mente.
- **Retreino e carregamento são procedimentos diferentes**: reconstruir árvores pode variar com versões e operações numéricas. Nesta avaliação, carregar o mesmo artefato reproduziu as métricas registradas na validação. O comportamento observado não garante igualdade bit a bit em qualquer ambiente.
- **A busca amostral é uma aproximação**: o candidato na cauda do treino marcou 0,5912. Para a entrega, usa-se 0,5719 do artefato congelado; 0,5703 identifica somente a execução histórica do tuning.
- **A melhor tentativa atingiu os limites do espaço reduzido** (300 árvores e profundidade 8), de modo que a busca não demonstra que esses tetos sejam ótimos globais.
- **Há degradação temporal no XGBoost entregue**: a AUC-PR caiu de 0,5719 na validação para 0,4833 no teste preservado.

## 8. Próxima etapa

O código de integração das três camadas está integrado ao repositório: recuperador documental, construção de prompt, pipeline e notebook demonstrativo. O índice reconstruído possui proveniência própria. Permanecem pendentes a ativação do cliente LLM, a resolução das entradas do pipeline na interface e a avaliação experimental das explicações; testes com clientes simulados não equivalem a geração real pelo LLM.

## 9. Ponto para alinhamento

O envio ao orientador deve ser feito pela dupla após a revisão deste documento. A implementação do cliente LLM, o responsável pela chave e o teto mensal de custo exigem decisão operacional conjunta. A presente entrega conclui a medição do modelo principal e a escolha do limiar; não declara a geração RAG ou a comparação de todos os classificadores no limiar final como concluídas.
