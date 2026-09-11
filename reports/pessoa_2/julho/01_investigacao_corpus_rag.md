# Investigação e versionamento do corpus RAG

Data da investigação: **30/08/2026**. Escopo: `m3_p2_1`.

> Registro histórico. O Guia MED 4.3 era a versão coletada nesta data. A [revalidação de 10/09/2026](../setembro/01_revalidacao_corpus_rag.md) atualizou o catálogo para o Guia 4.4 e gerou um novo snapshot.

## Objetivo

Confirmar quais documentos devem formar a primeira base de conhecimento do TCC, localizar suas versões oficiais atuais e definir o papel permitido de cada fonte. A investigação separa fatos confirmados, decisões técnicas e pendências; um link conhecido não foi considerado evidência suficiente sem inspeção do conteúdo efetivamente baixado.

## Veredito

**ACHADO — confiança alta:** as quatro fontes previstas no cronograma foram localizadas em canais oficiais e coletadas com hash SHA-256. O corpus bruto totaliza **13.471.408 bytes** e permanece fora do Git. O catálogo versionado está em [`../../../config/rag_corpus_sources.json`](../../../config/rag_corpus_sources.json), e o retrato dos hashes em [`corpus_snapshot_2026-08-30.json`](corpus_snapshot_2026-08-30.json).

| Documento | Evidência oficial verificada | Papel no RAG | Limite de uso |
|---|---|---|---|
| Regulamento do Pix, Resolução BCB nº 1/2020 | A página do BCB informa versão vigente atualizada em **27/04/2026** | Fonte normativa principal | Revalidar versão antes de uma nova entrega |
| Resolução BCB nº 103/2021 | O texto oficial introduz o MED no Regulamento do Pix | Origem normativa e histórica do MED | Ler em conjunto com o regulamento e o guia atuais |
| Guia do MED | O histórico do próprio PDF identifica a versão **4.3**, de **07/07/2026** | Procedimentos operacionais do MED | Não substitui o regulamento |
| Pesquisa Febraban 2024, volume 1 | Relatório oficial da entidade setorial | Contexto de tecnologia bancária | Não é norma e não comprova taxa de fraude Pix |

Fontes primárias: [Regulamento do Pix](https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?numero=1&tipo=Resolu%C3%A7%C3%A3o+BCB), [Resolução BCB nº 103/2021](https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?lang=pt&numero=103&tipo=Resolu%C3%A7%C3%A3o+BCB), [Guia do MED](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Guia_MED.pdf) e [Pesquisa Febraban 2024](https://portal.febraban.org.br/pagina/3106/1117/pt-br/pesquisa).

## Descobertas que alteraram a implementação

### 1. A página visível do BCB não é o documento baixável

**ACHADO — confiança alta:** uma requisição simples às duas páginas normativas retornou a mesma casca HTML de 2.871 bytes, sem o texto das resoluções. Aceitar esses arquivos teria produzido um corpus aparentemente válido, porém vazio.

A investigação do aplicativo público `buscanormas` mostrou que a própria página usa o endpoint oficial `/api/conteudo/app/normativos/exibenormativo`. O coletor passou a guardar a página pública como `source_url` e a resposta JSON oficial como `download_url`. A ingestão seleciona somente o texto normativo e metadados necessários, removendo conteúdo riscado/obsoleto antes do chunking.

### 2. O chunk do cronograma excede o limite do modelo

**ACHADO — confiança alta:** o modelo [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) produz vetores de 384 dimensões, mas seu comprimento máximo é 128 tokens. Enviar diretamente os chunks de 500 tokens previstos no cronograma descartaria silenciosamente a maior parte do texto.

Decisão técnica aplicada:

1. manter chunks pais de **500 tokens**, com **50 tokens** de sobreposição, para citação e contexto acadêmico;
2. criar janelas filhas de no máximo **126 tokens de conteúdo**, reservando dois tokens especiais do modelo;
3. usar **24 tokens** de sobreposição entre as janelas filhas;
4. ligar cada janela ao `parent_chunk_id`, preservando fonte, página, versão e hash.

Essa estrutura satisfaz a granularidade do cronograma sem introduzir truncamento. A documentação do Sentence Transformers também distingue codificação de documentos e consultas na busca semântica: [Semantic Search](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html).

### 3. Índice adequado ao tamanho atual

**SINAL FORTE — confiança alta para a prova de conceito:** o corpus gerou somente 1.192 janelas. Para esse porte, um índice exato é simples e evita a perda de recall introduzida por aproximações. O FAISS documenta que similaridade cosseno pode ser implementada normalizando os vetores e usando produto interno. Foram adotados `IndexFlatIP` e normalização L2: [métricas no FAISS](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) e [tipos de índice](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes).

## Matriz de riscos

| Risco | Consequência | Controle adotado |
|---|---|---|
| Norma muda após a coleta | resposta baseada em texto desatualizado | versão, data, URL e SHA-256; coleta reproduzível |
| Página SPA sem conteúdo | embeddings de menu/casca do site | endpoint oficial validado e teste de texto não vazio |
| Texto revogado misturado ao vigente | recuperação de regra incorreta | remoção de marcação riscada e preferência pelo texto consolidado |
| Truncamento do MiniLM | perda de evidência sem erro visível | janelas filhas limitadas pelo `max_seq_length` real |
| Relatório setorial tratado como norma | afirmação com autoridade indevida | `document_type`, `norm_status` e `corpus_role` explícitos |
| Alteração local de documento/índice | origem não reproduzível | validação SHA-256 antes da leitura |

## Limites da conclusão

- Os hashes comprovam os bytes coletados nesta data, não a permanência futura dos documentos na internet.
- A investigação validou origem e versão; não realizou parecer jurídico sobre o conteúdo.
- A qualidade da busca ainda precisa de um conjunto de perguntas e relevâncias anotadas para calcular recall em `k` e precisão em `k`.
