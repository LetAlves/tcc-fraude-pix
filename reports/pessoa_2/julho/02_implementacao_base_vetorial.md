# Implementação da base vetorial RAG

Data da execução: **30/08/2026**. Escopo: `m3_p2_2`, `m3_p2_3` e `m3_p2_4`.

> Registro histórico. Após a atualização para o Guia MED 4.4, a [execução de 10/09/2026](../setembro/01_revalidacao_corpus_rag.md) manteve 106 unidades e 290 chunks pais, passando de 1.192 para 1.195 janelas de embedding.

## Objetivo

Transformar o corpus oficial em uma base local pesquisável, com chunking reproduzível, embeddings multilíngues e recuperação por similaridade. Esta etapa implementa somente ingestão e busca; não chama um LLM e não gera uma conclusão sobre fraude.

## Arquitetura implementada

```text
catálogo versionado de fontes
  → download HTTPS autorizado
  → manifesto local com SHA-256
  → extração PDF / JSON normativo do BCB
  → chunks pais 500/50
  → janelas de embedding 126/24 ligadas ao pai
  → MiniLM multilíngue (384 dimensões)
  → normalização L2
  → FAISS IndexFlatIP
  → índice + metadados JSONL + manifesto com hashes
```

## Componentes

| Arquivo | Responsabilidade |
|---|---|
| [`../../../scripts/download_rag_corpus.py`](../../../scripts/download_rag_corpus.py) | baixa somente fontes catalogadas, valida HTTPS/MIME/tamanho, grava atomicamente e calcula SHA-256 |
| [`../../../src/rag/knowledge_base.py`](../../../src/rag/knowledge_base.py) | verifica integridade, extrai PDF/JSON/HTML e cria chunks pais rastreáveis |
| [`../../../src/rag/embeddings.py`](../../../src/rag/embeddings.py) | respeita o limite real do modelo, cria janelas filhas e codifica documento/consulta |
| [`../../../src/rag/vector_store.py`](../../../src/rag/vector_store.py) | cria, salva, carrega e consulta o `IndexFlatIP`; não usa pickle |
| [`../../../scripts/build_rag_index.py`](../../../scripts/build_rag_index.py) | orquestra a construção completa |
| [`../../../scripts/query_rag_index.py`](../../../scripts/query_rag_index.py) | permite testar recuperação local sem LLM |

## Resultado real da construção

| Medida | Resultado |
|---|---:|
| Documentos catalogados | 4 |
| Unidades extraídas (páginas ou documentos normativos) | 106 |
| Chunks pais | 290 |
| Janelas indexadas | 1.192 |
| Limite do modelo | 128 tokens |
| Conteúdo máximo por janela | 126 tokens |
| Dimensão do vetor | 384 |
| Índice | FAISS `IndexFlatIP` exato |

O índice local gerado possui 1.192 vetores. Seus artefatos permanecem em `data/rag/index/`, ignorados pelo Git. O manifesto local registra os hashes do índice e dos metadados para impedir carregamento silencioso de arquivos alterados.

## Teste semântico de fumaça

Consulta: “Como funciona o Mecanismo Especial de Devolução em caso de suspeita de fraude?”

Os dois primeiros resultados vieram da Resolução BCB nº 103/2021, com escores aproximados de **0,7644** e **0,7626**. Os três seguintes vieram do Guia do MED, páginas 35, 39 e 4. O primeiro trecho recuperado contém a definição do MED e sua aplicação em suspeita de fraude. Esse resultado confirma o funcionamento ponta a ponta, mas ainda não é uma avaliação estatística da recuperação.

## Como reproduzir

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_rag_corpus.py --refresh
.\.venv\Scripts\python.exe scripts\build_rag_index.py
.\.venv\Scripts\python.exe scripts\query_rag_index.py "Como funciona o MED?" --k 5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Depois do primeiro download do modelo, `scripts\build_rag_index.py --offline` reconstrói o índice sem consultar o Hugging Face. A consulta já exige o modelo presente no cache local e opera em modo offline.

## Segurança e manutenção

- URLs não são recebidas do usuário; apenas o catálogo versionado pode autorizar downloads.
- O coletor aceita HTTPS, limita cada documento a 100 MB, valida tipo de conteúdo e usa gravação atômica.
- O carregador verifica o hash do documento antes da extração e os hashes do índice antes da leitura.
- Metadados são JSON/JSONL; nenhum arquivo `pickle` é desserializado.
- Documentos, chunks e vetores não entram no Git; código, configuração, testes e evidências agregadas podem ser versionados.
- O índice deve ser reconstruído sempre que o hash ou a versão de uma fonte mudar.

## Pendência de avaliação

O próximo critério de pronto é criar perguntas de referência em português, marcar manualmente os chunks relevantes e medir `precision@k`, `recall@k` e posição do primeiro resultado relevante. O teste de fumaça não substitui esse conjunto de avaliação.
