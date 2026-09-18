# Maio de 2026 - Pessoa 2 (RAG + escrita)

Este diretório reúne as evidências das tarefas de maio atribuídas à Pessoa 2 no [cronograma de tarefas](https://lunogueira-67.github.io/tcc-organizador-tarefas/). O objetivo não é apenas marcar itens: cada entrega registra o que foi estudado, as fontes usadas, os limites da evidência e como o conteúdo entra no protótipo ou na monografia.

Data da revisão: **15/08/2026**.

## Situação das tarefas

| ID | Tarefa | Situação | Evidência |
|---|---|---|---|
| `m1_p2_0a` | Entender ML -> SHAP -> RAG | Concluída | [`01_guia_arquitetura_e_rag.md`](01_guia_arquitetura_e_rag.md) |
| `m1_p2_0b` | Pesquisar Pix, norma e FEBRABAN 2024 | Concluída com correção do enunciado | [`02_pesquisa_pix_e_corpus.md`](02_pesquisa_pix_e_corpus.md) |
| `m1_p2_0c` | Ler três artigos sobre fraude financeira com ML | Concluída; quatro artigos revisados | [`03_revisao_artigos_e_referencias.md`](03_revisao_artigos_e_referencias.md) |
| `m1_p2_0d` | Entender RAG e sua utilidade | Concluída | [`01_guia_arquitetura_e_rag.md`](01_guia_arquitetura_e_rag.md) |
| `m1_p2_1` | Levantar 15+ referências | Concluída; 22 entradas importáveis | [`03_revisao_artigos_e_referencias.md`](03_revisao_artigos_e_referencias.md) e [`../../../monografia/referencias.bib`](../../../monografia/referencias.bib) |
| `m1_p2_2` | Criar documento compartilhado | Preparada; publicação externa pendente | [`../../../monografia/README.md`](../../../monografia/README.md) e projeto LaTeX compatível com Overleaf |
| `m1_p2_3` | Rascunhar o Capítulo 1 | Concluída; requer revisão da dupla/orientador | [`04_capitulo_1_rascunho.md`](04_capitulo_1_rascunho.md) e [`../../../monografia/capitulos/01_introducao.tex`](../../../monografia/capitulos/01_introducao.tex) |
| `m1_p2_4` | Repositório e convenção de branches | Concluída no repositório já compartilhado | [`../../../CONTRIBUTING.md`](../../../CONTRIBUTING.md) |

## Tarefas conjuntas de maio

| Tarefa | Situação verificável | Próxima ação humana |
|---|---|---|
| Ler o dicionário e discutir o mapeamento IEEE-CIS -> Pix | Concluída: leitura técnica, auditoria e validação conjunta registradas em [`../../reunioes/2026-08-16_mapeamento_ieee_cis_pix.md`](../../reunioes/2026-08-16_mapeamento_ieee_cis_pix.md) | Nenhuma |
| Reunião de kickoff: responsabilidades, calendário e ferramentas | Pauta e decisões recomendadas preparadas | Realizar a reunião e preencher data, participantes e decisões finais |
| Entregar proposta ao orientador | O guia informa que a proposta já foi aprovada | Anexar protocolo ou retorno do orientador se a instituição exigir |

As pendências humanas estão em [`05_mapeamento_e_kickoff.md`](05_mapeamento_e_kickoff.md). Elas não foram marcadas como concluídas porque uma reunião, uma aprovação ou a criação de um documento numa conta externa não pode ser presumida pelo repositório.

## Correções científicas relevantes

1. O Mecanismo Especial de Devolução (MED) foi introduzido pela **Resolução BCB nº 103, de 8 de junho de 2021**, com efeitos em novembro de 2021. A referência do guia à "Resolução BCB nº 403/2023" estava incorreta: a Resolução nº 403 é de **22 de julho de 2024** e fez um ajuste específico no regulamento.
2. O dataset IEEE-CIS representa transações de comércio eletrônico/cartão fornecidas pela Vesta; ele não é um dataset Pix. O trabalho deve declarar que se trata de uma **prova de conceito por analogia de sinais de risco**, não de validação de desempenho em Pix real.
3. `C*`, `D*`, `M*`, `V*` e `id_*` são parcial ou totalmente anonimizadas. Não é válido transformar nomes como `V258` em um significado Pix inventado. A ponte SHAP -> RAG deve usar somente atributos com semântica documentada ou atributos derivados descritos num registro controlado.

## Critério de pronto

Uma tarefa documental só é considerada pronta quando contém fonte primária ou científica rastreável, separa fato de inferência, registra limitações e aponta a aplicação no projeto. Uma tarefa conjunta ou externa só é concluída depois de uma ação verificável da dupla.
