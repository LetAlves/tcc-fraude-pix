# Revalidação do corpus RAG e fechamento técnico

Data: **10/09/2026**. Escopo: tarefas `m3_p2_1` a `m3_p2_5`.

## Objetivo

Verificar a implementação já integrada, atualizar as fontes regulatórias que mudaram e revisar a metodologia SHAP antes de um novo commit. A coleta foi limitada às quatro fontes previstas no cronograma.

## Fontes revalidadas

| Fonte | Resultado em 10/09/2026 | Papel no corpus |
|---|---|---|
| [Regulamento do Pix — Resolução BCB nº 1/2020](https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?numero=1&tipo=Resolu%C3%A7%C3%A3o+BCB) | Página oficial informa versão vigente atualizada em 27/04/2026 | norma consolidada principal |
| [Resolução BCB nº 103/2021](https://www.bcb.gov.br/estabilidadefinanceira/exibenormativo?lang=pt&numero=103&tipo=Resolu%C3%A7%C3%A3o+BCB) | ato oficial que introduziu o MED; conteúdo coletado pelo endpoint JSON do BCB | origem normativa e histórica |
| [Guia MED 4.4](https://www.bcb.gov.br/content/estabilidadefinanceira/pix/Regulamento_Pix/versoes_futuras/Guia_MED-versao4-4.pdf) | substitui a versão 4.3 no catálogo; vigências em 01/09/2026 e 26/10/2026 | orientação operacional com controle temporal |
| [Pesquisa Febraban de Tecnologia Bancária 2024 — volume 1](https://portal.febraban.org.br/paginas/1117/pt-br/pesquisa) | volume permanece publicado no catálogo oficial da entidade | contexto setorial, nunca fonte normativa |

O [catálogo versionado](../../../config/rag_corpus_sources.json) e o [snapshot de 10/09/2026](corpus_snapshot_2026-09-10.json) preservam identificador, versão e SHA-256.

O JSON consolidado do Regulamento do Pix mudou de hash e tamanho em relação ao snapshot de 30/08/2026, embora o cabeçalho oficial ainda informe atualização normativa em 27/04/2026. O snapshot foi renovado, mas a diferença de bytes não foi interpretada isoladamente como nova alteração normativa, pois o envelope JSON do endpoint também pode mudar.

## Achado principal — Guia MED 4.4

O PDF 4.3 do Banco Central passou a apontar para a versão 4.4. O histórico oficial da nova versão registra:

- **01/09/2026:** ampliação para 80 dias do prazo de contestação de transação de devolução;
- **26/10/2026:** inclusão da camada no grafo de rastreamento em informações da notificação de infração, além de ajustes associados.

O catálogo, a ingestão e os testes foram atualizados para transportar `effective_from` e `future_effective_from` até os chunks. Isso permite que a futura camada de geração identifique que o documento inclui alterações com vigência futura e aplique uma verificação adicional.

### Limitação temporal

Os metadados são documentais, não identificam automaticamente cada frase afetada pela vigência de 26/10/2026. Até essa data, uma resposta sobre camada no grafo deve conferir o histórico da versão ou se abster. Uma evolução recomendada é anotar vigência por seção/chunk.

## Resultado da execução

| Verificação | Resultado |
|---|---:|
| Testes automatizados | 30 aprovados |
| Fontes coletadas | 4 |
| Tamanho bruto | 13.474.017 bytes |
| Unidades extraídas | 106 |
| Chunks pais | 290 |
| Janelas indexadas | 1.195 |
| Dimensão | 384 |
| Índice | FAISS `IndexFlatIP` |
| SHAP disponível no ambiente | 0.52.0 — protocolo revisado, sem valores calculados |

A consulta “Qual é o prazo para contestar uma transação de devolução do MED?” recuperou o Guia 4.4 e um trecho que informa o limite de 80 dias. Esse é um teste de fumaça; não substitui avaliação com julgamentos de relevância.

## Evidência técnica das escolhas

- O [cartão do MiniLM multilíngue](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) informa vetores de 384 dimensões e `max_seq_length` igual a 128; por isso os chunks pais de 500 tokens são divididos antes da codificação.
- A [documentação do Sentence Transformers](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html) descreve a busca semântica e a distinção entre codificação de consultas e documentos.
- A [documentação do FAISS](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) orienta normalizar vetores antes de usar produto interno como similaridade de cosseno.
- A [documentação do TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) diferencia saída bruta, probabilidade e perda e exige explicitar como dependências entre features serão tratadas.

## Riscos e pendências

- A qualidade do retriever ainda não possui `precision@k` e `recall@k` em um conjunto anotado.
- O Guia MED deve ser revalidado novamente após 26/10/2026.
- A execução SHAP depende dos modelos e do pipeline congelados pela Pessoa 1.
- A revisão da dupla e do orientador continua necessária antes da versão acadêmica final.
