<!--
RASCUNHO INCOMPLETO — NÃO ENVIAR.

Três lacunas marcadas com "A PREENCHER" dependem de resultados que ainda não
existem. Enquanto elas estiverem no texto, este documento não está pronto para
o orientador. Apagar este comentário e o aviso abaixo quando estiver completo.
-->

> ⚠️ **Rascunho.** Este documento tem lacunas identificadas e não deve ser enviado nesta forma.

# Entrega parcial 2 — Modelos de classificação e explicabilidade

**TCC:** Protótipo Híbrido para Detecção e Explicação de Fraudes em Transações Pix Simuladas Utilizando Machine Learning, SHAP e RAG
**Autores:** Letícia Alves (ML e dados) · Lucas Nogueira (RAG e escrita)
**Data:** _a definir no envio_

> **Sobre os dados:** este trabalho **não utiliza dados reais do Pix** nem de instituições financeiras. Os experimentos usam o dataset público IEEE-CIS Fraud Detection (Kaggle), de transações com cartão. As variáveis inspiradas no Pix são proxies analíticos documentados, conforme a seção 4.1 da proposta aprovada.

## 1. O que esta entrega cobre

Os dois modelos previstos na proposta — **XGBoost** como principal e **Random Forest** como comparativo — treinados sobre as mesmas 590.540 transações, com o mesmo pré-processamento e o mesmo corte temporal do baseline de junho, e a aplicação do **SHAP** sobre o modelo escolhido.

Todos os modelos usam o mesmo tratamento de desbalanceamento (ponderação de classe, adotada em junho) e são medidos pelo mesmo código, `src/models/evaluator.py`. Isso é o que permite atribuir as diferenças ao algoritmo, e não à forma de medir.

## 2. Comparação entre modelos — validação

Conjunto de validação: 88.581 transações, 3.042 fraudes, taxa base 3,434%. A AUC-PR de um classificador aleatório equivale à taxa base, e é a régua de leitura.

| Modelo | AUC-PR | AUC-ROC | Tempo de treino |
|---|---|---|---|
| Regressão logística (baseline de junho) | 0,3930 | 0,8410 | 6 min |
| Random Forest, profundidade ≤ 20 | 0,4573 | 0,8751 | 140 s |
| XGBoost, hiperparâmetros padrão | 0,5293 | 0,9044 | 56 s |
| Random Forest, árvores completas | 0,5298 | 0,9078 | 176 s |
| **XGBoost ajustado (Optuna, retreino completo)** | **0,5703** | **0,9070** | 30 s¹ |

¹ Tempo aproximado somente do retreino final. A busca com 50 tentativas levou 829 s; o fluxo completo levou 902 s e atingiu pico de 5.266 MB.

**Dois pontos que a tabela já sustenta:**

O **Random Forest empata com o XGBoost sem ajuste** (0,5298 contra 0,5293). Não cabe afirmar superioridade genérica do modelo principal sobre o comparativo; o que a evidência mostra é que a diferença, se existir, vem da busca de hiperparâmetros.

**Ambos superam o baseline com folga**, o que justifica a complexidade adicional em relação a um modelo linear — o critério que havíamos declarado para esta etapa.

## 3. Metodologia da busca de hiperparâmetros

Seis decisões, registradas em `reports/anotacoes_metodologia.md`:

1. **A métrica otimizada é a AUC-PR na validação.** Otimizar AUC-ROC premiaria ordenar bem os 96,5% de casos fáceis; acurácia premiaria não detectar nada.
2. **O parâmetro de desbalanceamento não entra na busca.** Fica fixo na razão real entre classes (27,43), que é a forma do XGBoost expressar a ponderação de classe adotada em junho. Deixá-lo variar transformaria uma decisão metodológica já comparada em mais um hiperparâmetro.
3. **O pré-processamento é ajustado uma vez, fora da busca.** Não depende dos hiperparâmetros, e é ajustado somente no treino — validação e teste apenas recebem a transformação.
4. **O conjunto de teste não é tocado durante a busca.** A seleção acontece na validação; o teste é avaliado uma única vez, ao final, com o modelo já escolhido.
5. **A busca usa uma amostra temporal do treino.** Cada tentativa usa as 150 mil transações mais recentes do treino, com a validação completa. O candidato alcançou AUC-PR 0,5912 nessa condição; depois de retreinado no treino completo, obteve 0,5703, que é o número oficial. A diferença é registrada como custo da aproximação.
6. **O espaço e a poda foram reduzidos por viabilidade.** O teto passou de 600 para 300 árvores e a profundidade máxima de 10 para 8. Um `MedianPruner` interrompeu 25 das 50 tentativas após aquecimento. A melhor configuração atingiu os dois tetos, portanto não se afirma que seja um ótimo global.

Os melhores parâmetros foram `learning_rate=0,2308`, `max_depth=8`, `min_child_weight=14,5371`, `subsample=0,8712`, `colsample_bytree=0,9159`, `gamma=1,6854`, `reg_alpha=0,1130` e `reg_lambda=1,2327`, com 300 árvores e `scale_pos_weight=27,4343` fixo.

## 4. Explicabilidade — SHAP

> ⚠️ **A PREENCHER — Camada 2.** Depende da `m3_p1_4`, ainda não iniciada, que por sua vez depende do modelo final do XGBoost. Deve trazer: a importância global das variáveis, a explicação de transações individuais, e a ressalva de que contribuição do SHAP indica influência sobre a decisão do modelo, **não causalidade** nem significado semântico para as colunas anônimas do dataset.

## 5. Avaliação final no teste

> ⚠️ **A PREENCHER.** Só deve ser executada depois de o modelo e o limiar de decisão estarem escolhidos na validação. Avaliar no teste antes disso, ou avaliar várias configurações e reportar a melhor, invalida o número.

## 6. Ressalvas metodológicas

**As métricas de decisão não são comparáveis entre modelos no limiar padrão.** No corte de 0,5, os modelos operam em pontos muito distintos da curva:

| Modelo | Recall | Precisão |
|---|---|---|
| Regressão logística | 0,68 | 0,14 |
| XGBoost padrão | 0,67 | 0,27 |
| Random Forest, árvores completas | 0,32 | 0,80 |

Cada modelo distribui suas probabilidades de forma própria, e o mesmo 0,5 cai em lugares diferentes de cada curva. Comparar precisão, recall ou F1 nesse ponto compara **calibração**, não qualidade. A comparação legítima entre modelos é a AUC-PR, que independe do limiar.

O limiar será escolhido na validação por critério declarado, e só então as métricas de decisão passam a ser comparáveis. _Critério a definir._

**Profundidade no Random Forest.** Com árvores completas, a floresta chega a profundidade média 64 e 16.430 folhas por árvore — memorização de transações individuais. Limitar a profundidade a 20 reduz o custo, mas **custa 0,07 de AUC-PR** (0,4573 contra 0,5298). A capacidade extra é aproveitada pela média das 100 árvores, e por isso foi mantida.

Uma medição preliminar com 10 árvores havia indicado empate entre as duas configurações. A conclusão não se sustentou com 100: extrapolação linear vale para custo, não para qualidade.

## 7. Limitações assumidas

- **O dataset é de cartão, não de Pix** — nenhum resultado indica desempenho no Pix real.
- **Três variáveis da proposta não têm equivalente no IEEE-CIS**: tipo de chave Pix, perfil do destinatário e idade da conta.
- **O Random Forest não passou por busca de hiperparâmetros**, conforme seu papel de modelo comparativo na proposta. A comparação com o XGBoost ajustado precisa ser lida com essa assimetria em mente.
- **A execução não é reprodutível bit a bit**: o uso de `float32`, necessário para caber na memória disponível, faz os resultados variarem a partir da terceira casa decimal.
- **A busca amostral é uma aproximação**: o melhor candidato na cauda do treino marcou 0,5912, mas o mesmo conjunto de parâmetros retreinado no treino completo marcou 0,5703. Só o segundo valor entra na comparação final.
- **A melhor tentativa atingiu os limites do espaço reduzido** (300 árvores e profundidade 8), de modo que a busca não demonstra que esses tetos sejam ótimos globais.
- **A degradação temporal identificada em junho continua valendo** e deve ser verificada também nos modelos desta etapa.

## 8. Próxima etapa

Agosto prevê a integração das três camadas: recuperação documental sobre a base regulatória (RAG), geração da explicação em linguagem natural combinando os fatores do SHAP com os documentos recuperados, e a interface de demonstração.

## 9. Ponto para alinhamento

<!-- Letícia: ajuste conforme a conversa sobre calendário. -->

_A definir._
