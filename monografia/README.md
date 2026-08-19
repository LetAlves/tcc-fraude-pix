# Projeto de escrita compatível com Overleaf

Esta pasta atende à parte local e versionável da tarefa `m1_p2_2`. Ela pode ser enviada ao Overleaf como um novo projeto (`New Project` -> `Upload Project`) depois de compactar o conteúdo de `monografia/`.

## Arquivos

- `main.tex`: entrada do projeto;
- `capitulos/01_introducao.tex`: rascunho do Capítulo 1;
- `capitulos/02_revisao_bibliografica.tex`: rascunho do Capítulo 2;
- `capitulos/03_metodologia.tex`: visão geral e protocolo planejado do Capítulo 3;
- `referencias.bib`: 22 referências importáveis também pelo Zotero.

## Antes de publicar

1. confirmar o template obrigatório da instituição;
2. substituir instituição, curso, cidade, orientador e título;
3. criar o projeto na conta escolhida pela dupla;
4. convidar a outra pessoa e, se apropriado, o orientador;
5. registrar neste README o link compartilhado, sem token de edição na URL pública;
6. manter o Git como histórico técnico e evitar edições concorrentes longas no Git e no Overleaf.

**Link do projeto compartilhado:** pendente de criação pela dupla.

## Compilação

O esqueleto usa `abntex2` e `abntex2cite`, disponíveis no Overleaf. Se a instituição fornecer classe própria, ela tem precedência. O arquivo ainda não foi compilado localmente porque a distribuição TeX não faz parte do ambiente deste repositório.

## Fluxo recomendado

- editar capítulos numa branch `docs/...`;
- revisar citações e compilar no Overleaf;
- exportar ou sincronizar as alterações aprovadas;
- abrir pull request e pedir revisão cruzada;
- nunca inserir chaves de API ou dados pessoais no projeto.
