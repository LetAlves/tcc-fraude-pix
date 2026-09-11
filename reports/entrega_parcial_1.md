# Entrega parcial 1 — EDA, features Pix simuladas e baseline

**TCC:** Protótipo Híbrido para Detecção e Explicação de Fraudes em Transações Pix Simuladas Utilizando Machine Learning, SHAP e RAG
**Autores:** Letícia Alves (ML e dados) · Lucas Nogueira (RAG e escrita)
**Data:** 11/09/2026

> **Sobre os dados:** este trabalho **não utiliza dados reais do Pix** nem de instituições financeiras. Os experimentos usam o dataset público IEEE-CIS Fraud Detection (Kaggle), de transações com cartão. As variáveis inspiradas no Pix são proxies analíticos documentados, conforme a seção 4.1 da proposta aprovada.

## 1. O que foi concluído

Análise exploratória, as quatro features Pix simuladas, o pré-processamento completo, a divisão treino/validação/teste e o baseline treinado e avaliado sobre as 590.540 transações do dataset.

O escopo previsto era EDA e features; o baseline foi incluído porque já está concluído e produziu o resultado da seção 3.

**Decisões metodológicas principais:** divisão **temporal** (transações antigas no treino, recentes no teste), em vez de aleatória; exclusão de `card4` e `card6` por serem conceitos exclusivos de cartão, sem equivalente no Pix; e acurácia não reportada em momento algum, por ser inútil num problema com 3,5% de fraude. O registro completo das decisões está em `reports/anotacoes_metodologia.md`.

## 2. Resultados do baseline

Regressão logística, limiar 0,5. Duas estratégias de desbalanceamento comparadas em pipelines separados. A taxa de fraude é de 3,43% na validação e 3,48% no teste — como a AUC-PR de um classificador aleatório equivale à taxa base, ela é a régua de leitura.

| | Validação | | Teste | |
|---|---|---|---|---|
| **Métrica** | Ponderação | SMOTE | Ponderação | SMOTE |
| AUC-PR | 0,3934 | 0,3967 | 0,1858 | 0,1840 |
| AUC-ROC | 0,8414 | 0,8392 | 0,8289 | 0,8234 |
| Recall | 0,6785 | 0,6831 | 0,7032 | 0,7071 |
| Precisão | 0,1347 | 0,1316 | 0,1236 | 0,1213 |

**As duas estratégias empataram**, na validação e no teste (diferenças de 0,003 e 0,002). O sinal da diferença inclusive se inverte conforme o recorte dos dados. A escolha entre elas será feita por critérios secundários — não fabricar dados sintéticos, não interpolar variáveis categóricas e custo computacional (o SMOTE levou 42 minutos contra 6) — e não por desempenho.

Em termos operacionais no teste: das 3.083 fraudes o modelo recupera cerca de 2.180, marcando ~18.000 das 88.581 transações como suspeitas. São 20% do total, com 8 de cada 10 acusações sendo alarme falso — comportamento esperado de um baseline linear, e o piso que os modelos de julho precisam superar.

## 3. Achado principal: degradação temporal

**A AUC-PR cai pela metade entre validação e teste — de 0,3934 para 0,1858.** A queda não se explica por desbalanceamento: as taxas base são equivalentes (3,43% e 3,48%), a AUC-ROC quase não se move e o recall até sobe. O que se degrada é a pureza das previsões de maior confiança no período mais recente: os padrões aprendidos envelhecem.

Verificamos experimentalmente que isso só é visível por causa da divisão temporal. Um controle com o mesmo modelo e os mesmos dados, mudando apenas a forma de dividir:

| | Corte temporal | Divisão aleatória |
|---|---|---|
| AUC-PR validação | 0,3936 | 0,4220 |
| AUC-PR teste | 0,1858 | 0,4250 |
| Variação | **−52,8%** | +0,7% |

A divisão aleatória reporta **2,3 vezes** a AUC-PR no teste e não mostra degradação alguma: descreveria um modelo estável e duas vezes melhor do que ele é. *Ressalva:* ela também coloca transações do mesmo identificador de cartão nos dois lados, então a diferença combina período e identificador compartilhados, e não se atribui inteiramente ao tempo.

O achado indica que um sistema assim exigiria retreino periódico em operação, e é evidência própria para a justificativa metodológica do Capítulo 3.

## 4. Limitações assumidas

- **O dataset é de cartão, não de Pix** — nenhum resultado aqui indica desempenho no Pix real.
- **Três variáveis da proposta não têm equivalente no IEEE-CIS**: tipo de chave Pix, perfil do destinatário e idade da conta.
- **O limiar de decisão permanece em 0,5**, valor padrão, não escolhido. Ajustá-lo na validação é a próxima melhoria de menor custo.
- **O SMOTE interpola variáveis categóricas já codificadas**, gerando combinações inexistentes no domínio.
- **A execução não é reprodutível bit a bit**: o uso de `float32`, necessário para caber na memória disponível, faz os resultados variarem a partir da terceira casa decimal entre execuções.

## 5. Observação sobre a base regulatória

A proposta aprovada cita a **Resolução BCB nº 403/2023** como base de prevenção a fraudes no Pix. Ao levantar as fontes para a base documental do RAG, identificamos que o Mecanismo Especial de Devolução foi criado pela **Resolução BCB nº 103/2021**; a nº 403 é de 22/07/2024 e trata de outro ajuste. A referência será corrigida na monografia.

## 6. Próxima etapa e alinhamento

Julho prevê XGBoost (principal) e Random Forest (comparativo), o módulo de avaliação com matriz de confusão e curva de precisão-recall, e a aplicação do SHAP. Os modelos serão avaliados no mesmo corte temporal, para que a queda entre validação e teste seja comparável. O critério de sucesso é **melhorar a precisão sem perder recall**.

<!-- Letícia: ajuste o parágrafo abaixo antes de enviar. -->

Esta entrega está sendo enviada em setembro, fora do prazo originalmente previsto. Gostaríamos de alinhar o calendário das etapas seguintes antes de avançar, para que as próximas entregas tenham prazos realistas.

---

**Evidências:** `notebooks/02_preprocessing.ipynb` (pré-processamento e baseline executados) · `reports/anotacoes_metodologia.md` (decisões e experimentos) · `reports/pessoa_2/junho/04_entrega_parcial_eda_features.md` e `reports/eda_summary.txt` (detalhamento da EDA e registro das features) · `src/features/` (código).
