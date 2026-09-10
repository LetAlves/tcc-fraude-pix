# Resumo para estudo — tarefas de julho de 2026

> Atualizado em 10/09/2026: corpus migrado do Guia MED 4.3 para 4.4, índice reconstruído com 1.195 janelas e suíte validada com 30 testes.

## Visão geral

Em julho, o objetivo da Pessoa 2 foi construir a primeira base da camada RAG do TCC e documentar como as explicações SHAP serão integradas ao sistema. O trabalho transformou documentos oficiais sobre Pix em uma base local pesquisável, preparada para futuramente fornecer contexto confiável a um modelo de linguagem.

O fluxo desenvolvido foi:

```text
fontes oficiais
  → download e verificação de integridade
  → extração dos textos
  → divisão em chunks
  → geração de embeddings
  → indexação no FAISS
  → consulta semântica
  → futura geração de resposta com LLM
```

As entregas originais foram integradas à branch `main` do GitHub pela Pull Request nº 3 e revalidadas com 30 testes automatizados em 10/09/2026.

## 1. Investigação e seleção do corpus RAG

### Objetivo

Escolher documentos confiáveis para formar a base de conhecimento e garantir que cada informação usada pelo sistema possa ser rastreada até uma fonte oficial.

### O que foi realizado

Foram localizadas, analisadas e catalogadas quatro fontes:

1. Regulamento do Pix — Resolução BCB nº 1/2020;
2. Resolução BCB nº 103/2021, que introduziu o MED;
3. Guia do Mecanismo Especial de Devolução — MED, atualizado para a versão 4.4;
4. Pesquisa Febraban de Tecnologia Bancária 2024.

Cada documento recebeu informações de origem, versão, data, tipo, papel dentro do RAG e hash SHA-256. O hash funciona como uma impressão digital: se o arquivo mudar, a verificação detecta que ele não é mais igual ao documento originalmente catalogado.

### Descoberta importante

As páginas visíveis das resoluções do Banco Central carregavam apenas a estrutura do site, sem o conteúdo normativo completo. Foi localizado o endpoint oficial utilizado pelo próprio sistema do BCB e o coletor passou a baixar o texto normativo correto em JSON.

### Por que essa tarefa é importante

Um sistema RAG não é confiável apenas porque possui documentos. Ele precisa usar fontes adequadas, atuais e verificáveis. Sem esse controle, o modelo de linguagem poderia responder usando páginas vazias, normas antigas ou documentos sem autoridade normativa.

## 2. Download seguro e preparação dos documentos

### Objetivo

Automatizar a coleta dos documentos e impedir que arquivos incorretos ou alterados entrem silenciosamente na base.

### O que foi implementado

O script de download:

- aceita somente endereços previamente registrados no catálogo;
- exige conexão HTTPS;
- valida tipo e tamanho do conteúdo;
- limita cada arquivo a 100 MB;
- grava os arquivos de maneira atômica;
- calcula e registra o SHA-256;
- permite atualizar o corpus de forma reproduzível.

Os documentos brutos ficam em `data/rag/` e não são enviados ao GitHub. O Git guarda apenas código, configurações, testes e evidências leves. Essa separação evita colocar arquivos grandes no repositório e respeita a regra do projeto de não versionar datasets.

## 3. Extração e chunking

### Objetivo

Converter PDF e JSON em partes menores, rastreáveis e adequadas à busca semântica.

### Conceito principal

Um **chunk** é um trecho de texto. Documentos inteiros são grandes demais para uma busca precisa, por isso são divididos em partes menores.

### Estratégia utilizada

- chunks pais com até 500 tokens;
- sobreposição de 50 tokens entre chunks pais;
- manutenção de fonte, página, versão e hash nos metadados;
- remoção de conteúdo normativo riscado ou obsoleto quando identificado.

A sobreposição evita perder o sentido de uma frase ou regra localizada exatamente na divisão entre dois chunks.

### Resultado

- 4 documentos catalogados;
- 106 unidades extraídas, considerando páginas e documentos normativos;
- 290 chunks pais produzidos.

## 4. Embeddings multilíngues

### Objetivo

Transformar os textos e as perguntas em vetores numéricos, permitindo comparar seus significados.

### Modelo escolhido

Foi utilizado o `paraphrase-multilingual-MiniLM-L12-v2`, adequado a consultas em português e capaz de gerar vetores com 384 dimensões.

### Problema identificado e solução

O modelo aceita no máximo 128 tokens. Enviar diretamente um chunk de 500 tokens faria o modelo truncar parte do conteúdo sem deixar isso evidente.

Para evitar a perda de informação, cada chunk pai foi dividido em janelas filhas:

- até 126 tokens de conteúdo;
- 2 tokens reservados pelo modelo;
- sobreposição de 24 tokens;
- vínculo com o identificador do chunk pai.

Essa solução preserva o contexto acadêmico nos chunks maiores e respeita o limite técnico do modelo na criação dos embeddings.

## 5. Banco vetorial FAISS

### Objetivo

Armazenar os embeddings e recuperar os trechos semanticamente mais próximos de uma pergunta.

### Implementação

Foi utilizado o FAISS com `IndexFlatIP`. Antes da indexação, os vetores são normalizados com L2. Assim, o produto interno funciona como comparação por similaridade de cosseno.

O índice é exato, isto é, compara a consulta com todos os vetores. Essa alternativa foi escolhida porque a base atual é pequena e possui somente 1.195 janelas, não justificando a perda de precisão de um índice aproximado.

### Resultado

- 1.195 janelas vetoriais;
- 384 dimensões por vetor;
- índice e metadados protegidos por hashes;
- nenhum uso de arquivos `pickle`, reduzindo riscos de desserialização insegura.

### Teste realizado

Na pergunta “Como funciona o Mecanismo Especial de Devolução em caso de suspeita de fraude?”, os primeiros resultados vieram da Resolução BCB nº 103/2021 e do Guia do MED. Isso comprovou que o fluxo completo de recuperação estava funcionando.

Esse teste é uma verificação inicial, não uma avaliação estatística. Ainda será necessário criar perguntas de referência e calcular métricas como `precision@k` e `recall@k`.

## 6. Recomendação do modelo de linguagem

### Objetivo

Comparar Claude via API e Llama local via Ollama para a futura camada que transformará os documentos recuperados em uma resposta textual.

### Recomendação preparada

- Claude Haiku 4.5 como opção principal para o protótipo;
- Ollama como alternativa local para privacidade e contingência.

Essa não é uma decisão final. Letícia e Lucas ainda precisam aprovar:

- o modelo escolhido;
- a política de privacidade;
- o limite mensal de custo;
- a administração da chave de API;
- os testes comparativos.

Nenhuma chave foi configurada e nenhum dado foi enviado para um LLM durante esta etapa.

## 7. Metodologia da Camada 2 — SHAP

### Objetivo

Definir como o sistema explicará a previsão do classificador antes de conectar essa explicação ao RAG.

### Conceito principal

SHAP atribui uma contribuição a cada atributo usado pelo modelo. A consistência da explicação deve obedecer aproximadamente à relação:

```text
saída do modelo ≈ valor-base + soma das contribuições SHAP
```

### Protocolo documentado

1. congelar divisão dos dados, pré-processamento, features, modelo e limiar;
2. construir o explicador apenas com artefatos do treino;
3. analisar exemplos de verdadeiro positivo, falso positivo, falso negativo e verdadeiro negativo;
4. verificar aditividade, ordem das colunas, valores ausentes e estabilidade numérica;
5. apresentar importância global pela média do valor absoluto de SHAP;
6. apresentar explicações locais com valor, sinal e magnitude;
7. comparar modelos somente sob o mesmo protocolo;
8. registrar limitações e casos sem interpretação segura.

### Cuidados essenciais

- SHAP explica o comportamento do modelo, não a causa real da fraude;
- uma feature importante não é necessariamente causal;
- atributos anônimos do IEEE-CIS não podem receber significados inventados;
- o dataset IEEE-CIS representa comércio eletrônico/cartão e não comprova desempenho real no Pix;
- somente conceitos aprovados no registro de features podem virar termos de busca do RAG.

## 8. Testes e controles de qualidade

Foram aprovados 30 testes automatizados, cobrindo principalmente:

- catálogo e download das fontes;
- validação de integridade;
- extração dos documentos;
- chunking e rastreabilidade;
- limites do modelo de embeddings;
- construção, salvamento e carregamento do FAISS;
- consulta à base vetorial;
- reconstrução em modo offline;
- evidências exigidas nas entregas de julho.

## 9. Arquivos principais para revisão

| Arquivo | Finalidade |
|---|---|
| `config/rag_corpus_sources.json` | catálogo das fontes autorizadas |
| `scripts/download_rag_corpus.py` | download seguro do corpus |
| `src/rag/knowledge_base.py` | extração e chunking |
| `src/rag/embeddings.py` | janelas e embeddings |
| `src/rag/vector_store.py` | criação e consulta do índice FAISS |
| `scripts/build_rag_index.py` | construção completa da base |
| `scripts/query_rag_index.py` | consulta local sem LLM |
| `reports/pessoa_2/julho/` | investigação, resultados e metodologia |

## 10. O que está concluído e o que permanece pendente

### Concluído

- investigação e catalogação das quatro fontes;
- download seguro e reproduzível;
- extração e chunking 500/50;
- janelas compatíveis com o limite do MiniLM;
- embeddings multilíngues;
- base vetorial FAISS;
- consulta semântica de fumaça;
- metodologia SHAP;
- testes automatizados;
- integração das entregas no GitHub.

### Pendente

- decisão conjunta sobre o LLM;
- avaliação do RAG com perguntas anotadas e métricas formais;
- execução dos modelos e valores SHAP pela Pessoa 1;
- revisão acadêmica do texto pela dupla e pelo orientador;
- integração final entre classificador, SHAP, RAG e LLM.

## 11. Resposta curta para explicar o trabalho

> Em julho, construímos a base da camada RAG do TCC. Selecionamos quatro fontes oficiais sobre Pix, criamos um processo seguro para coletar e verificar os documentos, dividimos o conteúdo em chunks rastreáveis e geramos embeddings multilíngues. Esses vetores foram indexados no FAISS para permitir busca por significado. Também documentamos como as explicações SHAP deverão ser produzidas e enviadas ao RAG sem inventar significados para atributos anônimos. Após a revalidação com o Guia MED 4.4, a solução gerou 1.195 vetores e foi validada por 30 testes automatizados.

## 12. Perguntas para revisar antes da apresentação

1. O que é RAG e por que ele reduz respostas sem fundamento?
2. Por que o corpus utiliza fontes oficiais e hashes SHA-256?
3. Qual é a diferença entre um chunk pai e uma janela de embedding?
4. Por que chunks de 500 tokens não foram enviados diretamente ao MiniLM?
5. O que representa um embedding de 384 dimensões?
6. Por que foi escolhido o `IndexFlatIP` em vez de um índice aproximado?
7. O que o teste de fumaça comprovou e o que ele ainda não comprova?
8. Qual é a diferença entre a função do classificador, do SHAP, do RAG e do LLM?
9. Por que SHAP não deve ser interpretado como causalidade?
10. Quais decisões ainda dependem da aprovação de Letícia e Lucas?

## Glossário

- **RAG:** técnica que recupera documentos relevantes antes de o LLM gerar uma resposta.
- **Corpus:** conjunto de documentos usados como base de conhecimento.
- **Chunk:** trecho menor extraído de um documento.
- **Token:** unidade de texto processada por um modelo.
- **Embedding:** representação numérica do significado de um texto.
- **Busca semântica:** busca por proximidade de significado, não apenas por palavras iguais.
- **FAISS:** biblioteca para armazenar e pesquisar vetores com eficiência.
- **Similaridade de cosseno:** medida de proximidade entre vetores normalizados.
- **SHA-256:** hash usado para verificar a integridade de um arquivo.
- **SHAP:** método que estima a contribuição de cada atributo para uma previsão.
- **LLM:** modelo de linguagem usado para produzir respostas em texto.
- **MED:** Mecanismo Especial de Devolução do Pix.
