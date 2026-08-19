# Notas de estudo para o Capítulo 2 — Revisão Bibliográfica

O rascunho acadêmico está em `monografia/capitulos/02_revisao_bibliografica.tex`. Este guia resume o que precisa ser entendido em cada uma das três tarefas de escrita de junho.

## 1. Fraudes no Pix e pagamentos instantâneos (`m2_p2_1`)

### Ideia central

O Pix é a infraestrutura de pagamento; a fraude pode ocorrer ao redor dela por manipulação da vítima, comprometimento de credenciais, abuso de conta ou outros mecanismos. Crescimento de uso não prova crescimento de fraude. O texto usa números do Banco Central apenas para demonstrar escala e relevância.

### Papel do MED

O Mecanismo Especial de Devolução integra o conjunto de regras do Pix e foi introduzido pela Resolução BCB nº 103/2021. Ele não deve ser apresentado como classificador de fraude, garantia de devolução ou equivalente automático de chargeback. Regras operacionais precisam ser citadas na versão vigente do Regulamento e do Guia do MED.

### Consequência para o protótipo

O modelo produz **risco estatístico**, não decisão jurídica. O RAG recupera contexto documental, mas não determina elegibilidade ao MED. A interface deve evitar frases acusatórias e indicar revisão humana.

## 2. Machine learning para detecção de fraude (`m2_p2_2`)

### Por que o problema é difícil

- a classe positiva é rara;
- padrões mudam ao longo do tempo;
- histórico e ordem carregam sinal;
- o rótulo pode chegar com atraso;
- falso positivo e falso negativo têm custos diferentes.

No treino IEEE-CIS local, apenas 3,499% das linhas têm `isFraud=1`. Por isso, a acurácia deve ser acompanhada de AUC-PR, precisão, recall, F1 e matriz de confusão.

### Modelos do TCC

- **Regressão Logística:** baseline simples, rápido e interpretável; demonstra se a engenharia e o protocolo já produzem sinal antes de modelos complexos.
- **Random Forest:** comparador não linear baseado em árvores e agregação.
- **XGBoost:** modelo principal previsto para dados tabulares, com regularização e boosting.

### SMOTE sem vazamento

SMOTE cria exemplos sintéticos da classe minoritária. Ele deve ser aplicado somente no treino, preferencialmente dentro de um pipeline que execute a reamostragem em cada fold. Aplicar antes da divisão faz amostras sintéticas derivadas de dados de validação/teste contaminarem o treino.

### Consequência para a escrita

Capítulo 2 explica métodos e riscos. Capítulo 3 descreve o protocolo escolhido. Capítulo 4 receberá métricas somente depois da execução. Não se deve escrever resultado futuro como se já tivesse sido observado.

## 3. RAG e LLMs (`m2_p2_3`)

### Componentes

1. documentos com fonte e versão;
2. chunks preservando metadados;
3. embeddings;
4. índice/vector store;
5. retriever;
6. prompt com contexto e evidência SHAP;
7. LLM para organizar a explicação;
8. validação de fontes e fidelidade.

### Por que usar RAG

O classificador e o SHAP não contêm o texto do Regulamento do Pix. O RAG permite recuperar evidência documental no momento da explicação. Isso reduz dependência da memória paramétrica do LLM e torna possível conferir a fonte.

### O que RAG não resolve sozinho

- documento irrelevante recuperado;
- norma desatualizada;
- afirmação sem suporte no trecho;
- atribuição falsa de significado a `V*`, `C*` ou `id_*`;
- divergência entre narrativa e sinal SHAP;
- vazamento de dado sensível no prompt.

## 4. Síntese da lacuna

A literatura oferece modelos para classe rara, SHAP para explicar a previsão e RAG para incorporar documentos. A lacuna investigada pelo TCC é a integração controlada dessas evidências em português, com metadados, limitações e separação explícita entre risco estatístico, influência do modelo e contexto regulatório.

## 5. Fontes centrais

- Banco Central: Regulamento do Pix, Resolução BCB nº 103/2021 e Guia do MED vigente;
- Abdallah, Maarof e Zainal (2016): survey de detecção de fraude;
- Jurgovsky et al. (2018): relevância de sequência e histórico;
- Dal Pozzolo et al. (2018): avaliação realista e mudança de conceito;
- Chawla et al. (2002): SMOTE;
- Saito e Rehmsmeier (2015): valor da curva precisão-recall em classe rara;
- Lundberg e Lee (2017): SHAP;
- Lewis et al. (2020): RAG.

As entradas completas estão em `monografia/referencias.bib`.
