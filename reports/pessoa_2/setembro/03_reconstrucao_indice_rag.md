# Reconstrução versionada do índice RAG

## Objetivo

Reconstruir o índice vetorial necessário à camada RAG sem apresentar a nova execução como se fosse o artefato histórico validado em setembro. A cópia original não estava disponível no workspace; por isso, esta execução possui identificação e proveniência próprias.

## Procedimento

O script `scripts/reconstruir_rag_versionado.py` utilizou o catálogo versionado em `config/rag_corpus_sources.json`, os quatro documentos baixados das fontes oficiais e a revisão local `e8f8c211226b894fcb81acc59f3b34ba3efd5f42` do modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. A execução ocorreu sem novo acesso ao Hugging Face.

Foram preservados os parâmetros definidos no projeto:

- chunks pais de 500 tokens e sobreposição de 50 tokens;
- janelas de embedding de até 126 tokens de conteúdo e sobreposição de 24 tokens;
- embeddings normalizados com 384 dimensões;
- índice exato `FAISS IndexFlatIP`, equivalente à similaridade de cosseno após normalização;
- lote de 32 textos por chamada de embedding.

## Resultado técnico

| Item | Resultado |
|---|---:|
| Fontes documentais | 4 |
| Unidades extraídas | 106 |
| Chunks pais | 290 |
| Janelas de embedding | 1.195 |
| Vetores no índice | 1.195 |
| Dimensões por vetor | 384 |
| Tamanho de `index.faiss` | 1.835.565 bytes |
| Tamanho de `metadata.jsonl` | 2.271.323 bytes |

Os hashes de três fontes coincidem com o snapshot de 10/09/2026. O JSON do Regulamento do Pix possui hash diferente e, portanto, não se presume identidade com a cópia histórica. O conteúdo completo da comparação, as versões de bibliotecas, os hashes e duas consultas de fumaça estão em `reconstrucao_rag.json`.

As consultas de fumaça recuperaram trechos relacionados ao Mecanismo Especial de Devolução e ao Guia MED, demonstrando que o índice pode ser carregado e consultado. Elas não medem relevância, fidelidade ou qualidade da explicação. Esses resultados experimentais permanecem como `{{RESULTADO}}` no protocolo de avaliação até que haja anotação humana.

## Decisão de versionamento

Os três arquivos operacionais do índice são versionados: `index.faiss`, `metadata.jsonl` e `manifest.json`. A decisão evita que cada integrante produza vetores diferentes e mantém a demonstração executável. Os documentos baixados e os chunks intermediários permanecem ignorados pelo Git.

O manifesto registra o nome e a revisão do modelo de embedding, a dimensão, a quantidade de documentos e hashes SHA-256. O carregamento valida a integridade antes de abrir o índice e o arquivo de metadados.

## Limitações

- esta reconstrução não substitui silenciosamente o índice histórico ausente;
- a alteração de hash de uma das quatro fontes impede afirmar identidade integral do corpus;
- a avaliação objetiva das explicações RAG ainda não foi executada;
- o cliente LLM permanece `{{PREENCHER}}`, portanto não há resultado de geração real a relatar.
