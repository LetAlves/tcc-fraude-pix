# Metodologia — visão geral das três camadas

O texto acadêmico correspondente está em `monografia/capitulos/03_metodologia.tex`. Este documento registra contratos, decisões e pendências de implementação.

## Desenho do estudo

O trabalho é uma prova de conceito experimental sobre o IEEE-CIS. O dataset permite testar a integração técnica, mas não estimar desempenho operacional em Pix. O fluxo será avaliado em três dimensões separadas: classificação, explicação local e recuperação/geração documental.

## Fluxo planejado

```text
transação IEEE-CIS
  -> pré-processamento e features proxy controladas
  -> modelo de classificação
  -> escore + limiar + versão
  -> SHAP por transação
  -> pacote de evidências semânticas
  -> consulta ao retriever
  -> chunks com fonte, versão e vigência
  -> prompt restritivo + LLM
  -> explicação em português + citações + limitações
```

## Contrato por camada

| Camada | Entrada | Saída | Avaliação | Limite |
|---|---|---|---|---|
| ML | atributos preparados | probabilidade, classe, limiar, versão | AUC-PR, AUC-ROC, precisão, recall, F1 e matriz | correlação, classe rara e mudança de distribuição |
| SHAP | modelo, referência e linha preparada | valor-base e contribuições locais | consistência numérica e estabilidade | não é causalidade nem revela semântica anônima |
| RAG + LLM | pacote SHAP + consulta controlada | explicação, chunks e fontes | recuperação, suporte documental e clareza | pode recuperar ou gerar conteúdo incorreto |

## Ponte SHAP→RAG

Colunas anônimas podem permanecer no classificador, mas não viram consulta regulatória. As quatro features aprovadas em 16/08/2026 e descritas no `config/pix_feature_registry.json` podem fornecer tags semânticas ao RAG. A implementação está em `src/features/pix_features.py`. O pacote deve preservar:

- ID e versão da feature;
- valor observado e unidade;
- contribuição SHAP e direção;
- tag conceitual aprovada;
- advertência de domínio.

Se uma coluna anônima estiver entre as maiores contribuições, a explicação deve chamá-la de “atributo anonimizado de alta influência”, sem inventar seu significado.

## Protocolo experimental planejado

1. ordenar as transações por `TransactionDT`;
2. separar temporalmente treino, validação e teste em 70/15/15, mantendo timestamps empatados na mesma partição e sem consultar o alvo das partições futuras;
3. ajustar imputação, encoding, escalonamento, seleção e estatísticas históricas apenas no treino;
4. comparar baseline com peso de classe e, separadamente, SMOTE dentro do pipeline de treino;
5. escolher limiar na validação segundo critério registrado;
6. avaliar uma única vez no teste preservado;
7. registrar seed, versões, features e parâmetros.

O cronograma original falava em divisão estratificada 70/15/15. Para fraude com dependência temporal, a estratificação aleatória pode misturar padrões futuros no treino e superestimar a generalização. A decisão aprovada é usar corte temporal 70/15/15 como avaliação principal e reservar a estratificação aleatória para análises complementares ou folds internos devidamente justificados.

No dataset completo, o protocolo produziu 413.378 linhas de treino, 88.581 de validação e 88.581 de teste. As taxas de fraude foram, respectivamente, 3,517%, 3,434% e 3,480%, com diferença máxima de 0,083 ponto percentual. As classes continuam desbalanceadas, porém a proporção da classe positiva permaneceu semelhante entre as partições sem necessidade de estratificação. Ocorrências repetidas além da primeira representam 2,9% das linhas; todas as linhas pertencentes a grupos de `TransactionDT` repetidos representam 5,7%. Nenhuma delas atravessou uma fronteira do corte.

## Avaliação da explicação

- **fidelidade ao modelo:** features e sinais narrados coincidem com SHAP;
- **recuperação:** perguntas de referência possuem chunks relevantes esperados;
- **fidelidade documental:** cada afirmação verificável é suportada pelo trecho citado;
- **qualidade humana:** texto claro, não acusatório e útil para revisão;
- **abstenção:** ausência de evidência gera ressalva, não preenchimento criativo.

## Reprodutibilidade e segurança

- nenhum CSV bruto ou chave de API entra no Git;
- dados enviados ao LLM serão sintéticos ou minimizados;
- documentos guardam emissor, versão, vigência, URL e hash;
- prompts usam templates controlados, nunca templates Jinja fornecidos pelo usuário;
- saídas do LLM não executam comandos, SQL ou ações administrativas;
- resultados devem registrar versão do código e ambiente.

## Pendências antes de congelar a metodologia

- [x] aprovação da dupla sobre o registro de quatro features e implementação causal;
- [ ] execução e revisão do baseline;
- [x] escolha final do split temporal 70/15/15, preservando empates de `TransactionDT`;
- [ ] escolha final do limiar;
- [ ] modelo de embeddings validado em português;
- [ ] LLM e política de custo/privacidade;
- [ ] conjunto de perguntas e chunks de referência para avaliação;
- [ ] aprovação do protocolo pelo orientador.
