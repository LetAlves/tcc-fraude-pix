# Revisão acadêmica final — P2 (Lucas)

## 1. Capítulo 1 — Introdução

O texto final, pronto para inserção e já incorporado ao projeto LaTeX, está em [`monografia/capitulos/01_introducao.tex`](../../../monografia/capitulos/01_introducao.tex). A versão revisada contém contextualização do Pix e da fraude, problema e questão de pesquisa, motivação, justificativa, objetivo geral, objetivos específicos, delimitação, contribuições e organização do trabalho.

A redação final evita afirmar que o IEEE-CIS representa transações Pix e distingue a função de cada camada. O objetivo geral passou a usar “apoiar a produção de explicações”, pois a recuperação e a montagem do prompt estão implementadas, mas o cliente LLM ainda não está disponível. A base de conhecimento foi descrita como normativa, operacional e setorial; a literatura científica sustenta a metodologia, mas não integra o índice vetorial versionado.

## 2. Capítulo 5 — Conclusão

O texto final, pronto para inserção e já incorporado ao projeto LaTeX, está em [`monografia/capitulos/05_conclusao.tex`](../../../monografia/capitulos/05_conclusao.tex). A conclusão apresenta:

- o pipeline e o protocolo temporal efetivamente desenvolvidos;
- as métricas do artefato XGBoost entregue, sem confundi-las com o retreino histórico do tuning;
- a configuração e a verificação de fidelidade do SHAP;
- a reconstrução versionada do índice RAG;
- a situação real da interface e do orquestrador;
- a ausência do cliente LLM e da avaliação experimental das explicações;
- o atendimento integral ou parcial de cada grupo de objetivos;
- contribuições, limitações e trabalhos futuros.

`{{REVISÃO_NECESSÁRIA}}`: existe divergência entre o limiar 0,6834 avaliado e persistido para o artefato final e o valor padrão 0,5 usado atualmente pela interface. O Capítulo 5 explicita essa limitação. Nenhuma explicação de LLM foi apresentada como resultado.

## 3. Revisão ABNT

| Item | Situação encontrada | Ajuste realizado | Pendência |
|---|---|---|---|
| Papel | Classe `abntex2` configurada para A4 | Mantido | Confirmar aderência ao manual institucional |
| Margens | Não estavam explícitas | Definidas em 3 cm (superior e esquerda) e 2 cm (inferior e direita) | Validar contra o template institucional |
| Fonte | Latin Modern, 12 pt | Substituída por `newtxtext`/`newtxmath`, família compatível com Times, em 12 pt | Confirmar se o template institucional exige exatamente Times New Roman ou aceita equivalente tipográfico no LaTeX |
| Espaçamento | Não havia configuração explícita global | Definido espaçamento de 1,5 e parágrafo com recuo de 1,25 cm, sem espaço adicional | Conferir citações longas, notas e referências após compilação |
| Hierarquia de títulos | Estruturada por `\chapter`, `\section` e `\subsection` | Mantida e completada com capítulos 4 e 5 | Revisar numeração e quebras de página no PDF |
| Paginação | Controlada pela classe, mas sem PDF validado | Mantido o fluxo `pretextual`/`textual`/`postextual` | Validar posição e início da numeração no PDF final |
| Legendas de figuras | Não há figuras consolidadas no texto atual | Nenhum conteúdo técnico foi inventado | Inserir título, fonte e nota em cada figura do Capítulo 4 |
| Tabelas e quadros | Não há tabelas ou quadros consolidados no projeto LaTeX | Nenhum dado foi criado para preencher a ausência | Padronizar título acima, fonte abaixo e continuidade de tabelas extensas |
| Notas de rodapé | Não identificadas | Nenhum ajuste necessário | Revisar quando o texto final estiver completo |
| Citações no texto | Sistema autor-data por `abntex2cite` | Conferidas 19 chaves citadas contra 19 entradas bibliográficas | Fazer inspeção visual após BibTeX |
| Lista de figuras e tabelas | Ausentes | Não adicionadas porque ainda não existem objetos consolidados | Incluir automaticamente quando o Capítulo 4 estiver completo |
| Sumário | Automático | Mantido `\tableofcontents*` | Compilar mais de uma vez para atualizar referências e páginas |
| Elementos pré-textuais | Apenas capa, folha de rosto e sumário | Dados desconhecidos foram trocados por `{{PREENCHER}}` | Faltam, conforme template institucional, folha de aprovação, resumo, palavras-chave, abstract, keywords e eventuais listas |
| Identificação institucional | Cidade, instituição e orientador estavam genéricos ou ausentes | Marcadores explícitos inseridos | Preencher cidade, instituição e orientador |
| Capítulo 4 | Arquivo consolidado não existia | Criado marcador honesto para preservar a numeração da conclusão | Substituir pelo capítulo de resultados aprovado, sem inventar resultados de RAG/LLM |
| Compilação | Não há distribuição TeX instalada no ambiente local | Sintaxe e referências foram verificadas estaticamente | Compilar no Overleaf e revisar órfãs, viúvas, estouros, legendas e paginação |
| Norma bibliográfica | Versão institucional não informada | Mantido `abntex2cite` no sistema autor-data | Confirmar `{{VERSÃO_NBR_6023}}` antes da entrega |

## 4. Revisão das referências

A auditoria estática encontrou 19 chaves citadas, 19 entradas bibliográficas, nenhuma citação sem referência e nenhuma entrada não utilizada. Três entradas sem uso no texto foram removidas: `bcb_resolucao403_2024`, `bcb_pix_numeros` e `carcillo2021combining`.

| Referência | Problema encontrado | Correção sugerida/realizada | Status |
|---|---|---|---|
| Conjunto da bibliografia | A versão da NBR 6023 exigida pela instituição não foi informada | Confirmar `{{VERSÃO_NBR_6023}}` e validar a saída do `abntex2cite` | Pendente de informação institucional |
| Banco Central do Brasil (2020), Resolução BCB nº 1 | O registro usa título descritivo abreviado | Conferir o título oficial completo na versão da norma adotada | `{{DADO_BIBLIOGRÁFICO_PENDENTE}}` |
| Banco Central do Brasil (2021), Resolução BCB nº 103 | O registro não contém a ementa/título oficial completo | Conferir o título oficial completo sem alterar autor, ano ou URL | `{{DADO_BIBLIOGRÁFICO_PENDENTE}}` |
| Banco Central do Brasil (2026), Guia MED v4.4 | A vigência é escalonada e a fonte pertence ao diretório de versões futuras | Manter versão e datas já registradas; confirmar qual situação normativa será descrita no texto final | Revisão humana necessária |
| FEBRABAN e Deloitte (2024) | O texto metodológico identifica “volume 1”, mas a entrada bibliográfica não registra volume | Confirmar o volume no exemplar efetivamente utilizado | `{{DADO_BIBLIOGRÁFICO_PENDENTE}}` |
| Lundberg e Lee (2017) | A referência eletrônica não tinha data de acesso | Incluída data de acesso de 20 set. 2026 | Corrigida |
| Lewis et al. (2020) | A referência eletrônica não tinha data de acesso | Incluída data de acesso de 20 set. 2026 | Corrigida |
| Reimers e Gurevych (2019) | Metadados confrontados com ACL Anthology | Autores, título, ano, páginas e DOI conferem | Verificada |
| Es et al. (2024) | Metadados confrontados com ACL Anthology | Autores, título, ano, páginas e DOI conferem | Verificada |

## 5. Pendências para revisão humana

1. Confirmar o template institucional e a versão `{{VERSÃO_NBR_6023}}`.
2. Preencher instituição, cidade, orientador e demais dados da folha de rosto.
3. Inserir folha de aprovação, resumo, palavras-chave, abstract e keywords conforme o manual do curso.
4. Substituir o marcador do Capítulo 4 pelos resultados consolidados e aprovados.
5. Corrigir a interface para carregar o limiar 0,6834 do artefato final ou explicar formalmente por que utilizará outro ponto de operação.
6. Implementar e versionar o cliente LLM antes de apresentar saídas textuais como resultado do sistema.
7. Executar a avaliação anotada do RAG e preencher somente então os campos `{{RESULTADO}}`.
8. Revisar a vigência das fontes do Pix e o hash divergente do Regulamento consolidado.
9. Preencher os dados bibliográficos marcados como `{{DADO_BIBLIOGRÁFICO_PENDENTE}}`.
10. Compilar no Overleaf, executar BibTeX e inspecionar visualmente paginação, sumário, quebras, legendas e referências.

## 6. Resumo das alterações realizadas

| Seção | O que foi revisado | O que foi alterado |
|---|---|---|
| Capítulo 1 | Coerência entre problema, objetivos, escopo e implementação | Redação final; distinção entre SHAP, recuperação e geração; corpus descrito conforme o índice real; cliente LLM tratado como pendente |
| Capítulo 2 | Coerência temporal e distinção entre desenho conceitual e implementação | Removido tempo futuro de decisões já executadas e registrada a limitação da consulta por nomes anônimos |
| Capítulo 3 — protocolo e SHAP | Coerência com os experimentos concluídos | Atualizado para o passado; registrados limiar, escopo real do SHAP, fidelidade numérica e ausência de SHAP equivalente para o Random Forest |
| Capítulo 3 — Camada 3 | Confronto do texto com `src.pipeline`, `src.rag.retriever`, `src.rag.explainer` e artefatos do índice | Atualizados números, data da reconstrução, `top_k=5`, ausência de filtros, conteúdo real do prompt, estado do cliente LLM e avaliação ainda não executada |
| Capítulo 4 | Estrutura inexistente | Criado marcador explícito de revisão necessária para manter a numeração correta sem inventar resultados |
| Capítulo 5 | Comparação entre objetivos e resultados reais | Criada conclusão completa com métricas, SHAP, RAG, interface, contribuições, limitações e trabalhos futuros |
| Formatação | Estrutura `abntex2`, papel A4, fonte, margens, espaçamento, recuo e metadados | Aplicada família Times compatível em 12 pt; configurações explícitas e marcadores para dados institucionais ausentes |
| Referências | Correspondência entre citações e bibliografia | Removidas três entradas sem uso; adicionadas datas de acesso; auditoria final 19/19 |
| README da monografia | Estado dos arquivos e instruções | Atualizado para refletir capítulos finais, pendências e quantidade correta de referências |
