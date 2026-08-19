# Estudo prático de LangChain: chains, retrievers e prompt templates

Data da verificação: **18/08/2026**.

## Objetivo

Entender as abstrações que serão usadas no RAG sem antecipar a implementação de embeddings, FAISS ou LLM prevista para os próximos meses. O laboratório é local, determinístico e não usa chave de API.

Versões verificadas no ambiente:

| Pacote | Versão |
|---|---:|
| `langchain` | `1.3.15` |
| `langchain-core` | `1.5.5` |
| `langchain-community` | `0.4.2` |

As páginas oficiais atuais descrevem a recuperação como um fluxo modular composto por documentos, divisores de texto, embeddings, vector stores e retrievers. Também distinguem retriever de vector store: um retriever é uma interface que recebe uma consulta e devolve uma lista de `Document`; a origem pode ser vetorial, lexical, API ou outra fonte ([LangChain — Retrieval](https://docs.langchain.com/oss/python/langchain/retrieval), [LangChain — semantic search](https://docs.langchain.com/oss/python/langchain/knowledge-base)).

## 1. `Document`

`Document` reúne:

- `page_content`: o texto recuperável;
- `metadata`: identificador, emissor, URL, versão, página/seção e demais campos de rastreabilidade.

No TCC, o texto sem metadados é insuficiente. Uma explicação precisa indicar de qual documento e versão veio a evidência. O laboratório cria três documentos: Resolução BCB nº 103/2021, artigo de SHAP e artigo original de RAG.

## 2. Retriever

Um retriever implementa o contrato conceitual:

```text
consulta em linguagem natural -> lista ordenada de Document
```

Ele **não** redige a resposta e **não** é necessariamente um banco vetorial. O laboratório usa `KeywordRetriever`, que ranqueia por sobreposição lexical. Isso permite testar o contrato sem baixar um modelo de embeddings. Em julho, essa implementação didática deve ser substituída por embeddings multilíngues e FAISS, preservando a mesma interface de invocação.

Exemplo:

```python
retriever = KeywordRetriever(documents=criar_documentos_didaticos(), k=2)
docs = retriever.invoke("MED Resolução Banco Central Pix")
```

## 3. Prompt template

Um prompt template define campos variáveis e regras constantes. O laboratório usa o formato `f-string`, recomendado para evitar o risco de executar templates Jinja fornecidos por origem não confiável ([referência oficial de `PromptTemplate`](https://reference.langchain.com/python/langchain-core/prompts/prompt/PromptTemplate)).

Campos utilizados:

- `question`: pergunta validada, não vazia e limitada a 500 caracteres;
- `context`: documentos recuperados com identificador e URL;
- regras fixas: não acusar fraude, não inventar significado de atributo anônimo e declarar insuficiência de evidência.

O template melhora consistência, mas não prova que a resposta de um LLM será correta. O pipeline futuro deve validar formato, citações e coerência com SHAP depois da geração.

## 4. Chain e `Runnable`

No LangChain Core atual, etapas que implementam `Runnable` podem ser compostas com `|`. A saída de uma etapa se torna a entrada da seguinte; `RunnableSequence` também oferece `invoke`, `batch`, variantes assíncronas e streaming quando os componentes suportam ([referência oficial de `RunnableSequence`](https://reference.langchain.com/python/langchain-core/runnables/base/RunnableSequence)).

O laboratório compõe:

```text
pergunta
  -> validação + retriever + formatação das fontes
  -> PromptTemplate
  -> prompt final (sem chamada a LLM)
```

Código correspondente:

```python
chain = RunnableLambda(retrieve_context) | PROMPT_EXPLICACAO
prompt_value = chain.invoke("Qual é o papel do MED no Pix?")
```

## 5. Execução

```powershell
.venv\Scripts\python.exe -m src.rag.langchain_basics
.venv\Scripts\python.exe -m unittest tests.test_langchain_basics -v
```

Os testes verificam:

1. recuperação de uma fonte identificada do MED;
2. presença da fonte e das restrições no prompt;
3. rejeição de pergunta vazia;
4. mensagem explícita quando nenhum documento é relevante.

## 6. O que ficou para os próximos meses

- extração e chunking dos documentos completos;
- embeddings avaliados em português;
- índice FAISS persistente;
- filtros por emissor, tipo, versão e vigência;
- integração com evidência SHAP;
- escolha e configuração do LLM;
- avaliação de recuperação e fidelidade documental.

Não seria adequado chamar o laboratório atual de “RAG completo”: ainda não há recuperação semântica nem geração por LLM. A tarefa de junho é entender e testar as interfaces que sustentarão essas etapas.

## Checklist de entendimento

- [x] Sei distinguir documento, vector store e retriever.
- [x] Sei que o retriever devolve documentos, não a resposta final.
- [x] Sei por que metadados fazem parte da evidência.
- [x] Sei compor etapas por `Runnable` sem usar uma classe antiga de chain como caixa-preta.
- [x] Sei que prompt template não elimina alucinação nem prompt injection.
- [x] Sei explicar o que ainda falta para o pipeline ser um RAG.
