# TCC — Fraude Pix com Machine Learning, SHAP e RAG

Prova de conceito acadêmica para **detecção e explicação de risco de fraude em contexto Pix**, combinando classificação por aprendizado de máquina, explicações locais com SHAP e recuperação de fontes documentais com RAG.

**Autores:** Letícia Alves — Pessoa 1 (ML e dados); Lucas Nogueira — Pessoa 2 (RAG e escrita).

> O IEEE-CIS contém transações do domínio de comércio eletrônico/cartão, não transações Pix reais. As features do projeto representam analogias analíticas documentadas. Os experimentos não comprovam desempenho operacional no Pix e o protótipo não deve ser usado para bloquear transações ou acusar pessoas.

[Guia do TCC](https://letalves.github.io/tcc-fraude-pix/) · [Entregas de maio](reports/pessoa_2/maio/README.md) · [Entregas de junho](reports/pessoa_2/junho/README.md) · [Entregas de julho](reports/pessoa_2/julho/README.md) · [Revalidação de setembro](reports/pessoa_2/setembro/README.md) · [Monografia](monografia/README.md) · [Como contribuir](CONTRIBUTING.md)

## Arquitetura proposta

```text
Dados IEEE-CIS + features proxy documentadas
  → pré-processamento e classificação de risco (ML)
  → contribuições locais dos atributos (SHAP)
  → recuperação de fontes e explicação com citações (RAG + LLM)
```

O classificador estima risco; SHAP descreve a influência dos atributos; o RAG fornece contexto documental. Uma contribuição SHAP não prova causalidade e uma coluna anônima não ganha significado Pix por ser importante para o modelo.

O [registro de features proposto no PR #2](https://github.com/LetAlves/tcc-fraude-pix/blob/ff73aa5ce60a5daf43fdf8195d1ac9386ad5718e/config/pix_feature_registry.json) descreve os conceitos controlados que podem alimentar a futura ponte SHAP → RAG.

## Estado do projeto

Situação verificada localmente em **12/09/2026**. As entregas de junho e julho da Pessoa 2 estão integradas à `main`; o corpus RAG foi revalidado com o Guia MED 4.4. A camada de ML de junho está concluída e executada no dataset completo; julho está em andamento. Dependências instaladas não significam que todas as camadas já estejam implementadas.

| Componente | Situação | Evidência |
|---|---|---|
| Download, leitura e junção dos dados | Implementados | [Data loader](src/data_loader.py) |
| Análise exploratória | Notebook executado | [EDA](notebooks/01_eda.ipynb) |
| Features Pix simuladas | Quatro conceitos implementados, gerando seis colunas | [Módulo de features](src/features/pix_features.py) e [ata de aprovação](reports/reunioes/2026-08-16_mapeamento_ieee_cis_pix.md) |
| Estudo de LangChain | Laboratório local integrado; sem chamada a LLM | [Guia prático](reports/pessoa_2/junho/01_estudo_langchain.md) |
| Monografia | Capítulos 1, 2 e 3 versionados; revisão acadêmica ainda necessária | [Projeto de escrita](monografia/README.md) |
| Pré-processamento e split temporal | Implementados, com garantia de fronteira por instante e 13 testes | [Pré-processador](src/features/preprocessor.py) |
| SMOTE e ponderação de classe | Comparados em quatro medições; empate, adotada a ponderação | [Anotações de metodologia](reports/anotacoes_metodologia.md) |
| Baseline (regressão logística) | Executado no dataset completo; AUC-PR 0,3930 na validação e 0,1850 no teste | [Notebook 02 executado](notebooks/02_preprocessing.ipynb) |
| Random Forest (modelo comparativo) | Treinado em duas configurações; AUC-PR 0,5298 | [Script](scripts/treinar_random_forest.py) |
| XGBoost (modelo principal) | Busca de hiperparâmetros implementada; execução completa pendente | [Módulo de busca](src/models/xgboost_tuning.py) |
| Avaliação e escolha de limiar | Implementadas em módulo único, usado por todos os modelos | [Evaluator](src/models/evaluator.py) |
| Persistência de modelos | Implementada, com manifesto, hashes e compressão | [Persistência](src/models/persistencia.py) |
| RAG vetorial | Implementado, integrado e revalidado com 1.195 vetores | [Revalidação de setembro](reports/pessoa_2/setembro/README.md) |
| SHAP executado e interface de demonstração | Planejados | [Metodologia SHAP](reports/pessoa_2/julho/04_metodologia_shap.md) |

O laboratório de LangChain não é o RAG final. Os textos da monografia ainda exigem revisão da dupla e do orientador; um relatório preparado não comprova seu envio ao orientador.

**Achado principal até aqui:** o desempenho cai pela metade entre validação e teste — AUC-PR de 0,3930 para 0,1850 — com taxas de fraude equivalentes nos dois conjuntos. É degradação temporal, e só é observável por causa do corte cronológico: um controle com divisão aleatória obteve 0,4243 no teste, 2,3× mais, sem mostrar queda alguma. O registro completo está nas [anotações de metodologia](reports/anotacoes_metodologia.md).

## Instalação — Windows / PowerShell

Pré-requisitos: Git e Python. O ambiente de desenvolvimento foi validado com **Python 3.13.9** em Windows 10.

```powershell
git clone https://github.com/LetAlves/tcc-fraude-pix.git
cd tcc-fraude-pix
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Execute os comandos seguintes na raiz do repositório. Eles usam diretamente o Python da venv, sem exigir sua ativação ou alteração da política de execução do PowerShell. Em Linux/macOS, o executável equivalente é `.venv/bin/python`.

As dependências diretas estão em [requirements.txt](requirements.txt), com limites mínimos de versão. Para **reproduzir os números registrados no trabalho**, instale a partir de [requirements-lock.txt](requirements-lock.txt), que fixa as versões exatas do ambiente que os produziu:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

Fixar as versões não torna a execução reprodutível bit a bit — o caminho de modelagem usa `float32` e as operações matriciais são paralelas, então a ordem das somas varia e os resultados mudam a partir da terceira casa decimal. O lockfile elimina a outra fonte de variação: não saber com que versões os números foram produzidos.

## Dataset e autenticação Kaggle

1. Acesse a competição [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) e aceite suas [regras](https://www.kaggle.com/competitions/ieee-fraud-detection/rules) na própria conta. O acesso pode exigir verificações adicionais do Kaggle.
2. Gere seu token na seção API das configurações do Kaggle.
3. Salve somente o valor do token em `%USERPROFILE%\.kaggle\access_token`, fora deste repositório. Como alternativa, configure `KAGGLE_API_TOKEN` no arquivo local `.env`.
4. Baixe e extraia os dados:

```powershell
.\.venv\Scripts\python.exe -c "from src.data_loader import _baixar_kaggle; _baixar_kaggle()"
```

Para os módulos atuais, os arquivos necessários são:

```text
data/raw/
├── train_transaction.csv
└── train_identity.csv
```

O download da competição também pode trazer outros CSVs. Se os dois arquivos de treino já existirem, `carregar_dados()` os reutiliza sem baixar novamente.

### Dados e segredos não vão para o Git

- O [.gitignore](.gitignore) exclui `data/raw/`, `data/processed/`, os artefatos de `data/rag/`, `.venv/` e `.env`.
- Não faça commit dos CSVs, de tokens ou de credenciais; não use `git add -f` para contornar essas exclusões.
- Cada integrante pode obter os dados com sua própria conta Kaggle. Google Drive e DVC **não estão configurados** neste projeto; qualquer compartilhamento externo deve respeitar as regras da fonte.
- Código, testes, documentação e estatísticas agregadas podem ser versionados sem incluir o dataset bruto.

Para verificar localmente:

```powershell
git check-ignore data/raw/train_transaction.csv data/processed/exemplo.csv .env
git ls-files data
```

O segundo comando deve retornar vazio. O `.gitignore` não remove arquivos que já tenham sido rastreados; revise sempre `git diff --cached` antes de fazer commit.

## Como executar

### Verificar a leitura dos dados

```powershell
.\.venv\Scripts\python.exe -m src.data_loader
```

Esse teste lê até 1.000 linhas de cada tabela e faz a junção por `TransactionID`. Se os arquivos estiverem ausentes, tentará baixá-los usando a configuração Kaggle.

### Abrir a análise exploratória

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks/01_eda.ipynb
```

A EDA completa exige mais memória que os testes com amostras. Execute as células em ordem e preserve as limitações de interpretação dos atributos anonimizados.

### Executar as features Pix simuladas

```powershell
.\.venv\Scripts\python.exe -m src.features.pix_features
```

O teste usa até 50.000 linhas de cada tabela, calcula as features e imprime um resumo. Ele não treina modelos nem grava automaticamente um novo dataset.

Os quatro conceitos são `valor_atipico_proxy`, `frequencia_recente_proxy`, `dispositivo_raro_proxy` e `posicao_ciclo_diario_relativa`. O último inclui também codificações seno e cosseno. `card1` não é uma conta Pix, e o ciclo temporal não representa um horário local conhecido.

### Acessar as entregas de junho em revisão

O laboratório, o gerador de relatório e seus testes estão na branch do PR #2, não na `main`. Para executá-los com o clone criado na instalação acima, e sem alterações locais pendentes:

```powershell
git fetch origin
git switch codex/junho-pessoa-2
```

Depois de consultar a entrega, `git switch main` retorna à branch principal. Essa troca local não aprova nem mescla o PR.

### Executar o laboratório de LangChain — branch de junho

```powershell
.\.venv\Scripts\python.exe -m src.rag.langchain_basics
```

Não exige dataset, download de embeddings ou chave de LLM. Recupera documentos didáticos e imprime o prompt montado; não gera uma resposta com modelo de linguagem.

### Reproduzir o relatório de junho — branch de junho

```powershell
.\.venv\Scripts\python.exe scripts/gerar_entrega_junho.py
```

Lê colunas selecionadas dos CSVs completos e **atualiza** [a entrega parcial de EDA e features](https://github.com/LetAlves/tcc-fraude-pix/blob/ff73aa5ce60a5daf43fdf8195d1ac9386ad5718e/reports/pessoa_2/junho/04_entrega_parcial_eda_features.md). Para preservar o relatório versionado e gerar uma cópia local:

```powershell
.\.venv\Scripts\python.exe scripts/gerar_entrega_junho.py --output data/processed/entrega_junho.md
```

### Construir e consultar a base RAG

```powershell
.\.venv\Scripts\python.exe scripts\download_rag_corpus.py --refresh
.\.venv\Scripts\python.exe scripts\build_rag_index.py
.\.venv\Scripts\python.exe scripts\query_rag_index.py "Como funciona o MED?" --k 5
```

Os documentos oficiais, chunks e vetores são artefatos locais ignorados pelo Git. O catálogo de fontes, o código, os testes e o snapshot de hashes são versionáveis. Depois do primeiro download do modelo, use `scripts\build_rag_index.py --offline` para reconstruir sem acesso ao Hugging Face.

O catálogo atual usa o Guia MED 4.4 e registra separadamente as vigências de 01/09/2026 e 26/10/2026. Até a segunda data, respostas sobre alterações futuras devem conferir os metadados de vigência.

## Testes e protocolo experimental

Execute a suíte na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
git diff --check
```

Na revalidação de 10/09/2026, **30 testes** foram aprovados. Eles cobrem o laboratório LangChain, o registro de features, estatísticas sintéticas, corpus RAG, chunking, embeddings, FAISS e referências dos capítulos. Eles não medem desempenho preditivo nem substituem uma avaliação anotada da recuperação ou uma validação completa dos modelos.

Para os próximos experimentos:

- separar treino, validação e teste antes de ajustar transformações;
- aplicar SMOTE somente no treino de cada partição ou fold;
- calcular atributos históricos sem usar eventos futuros;
- selecionar o limiar na validação e preservar o teste para avaliação final;
- registrar seed, versões, parâmetros, features e métricas da classe rara, incluindo AUC-PR, precisão, recall e F1;
- não apresentar métricas ainda não obtidas em experimentos executados.

## Organização do repositório

```text
config/                  Registro de features e catálogo das fontes RAG
data/raw/                CSVs originais locais — ignorados pelo Git
data/processed/          Dados derivados locais — ignorados pelo Git
docs/                    Guia e cronograma estático do TCC
monografia/              LaTeX, capítulos e bibliografia BibTeX
notebooks/               Análise exploratória
reports/                 Entregas, dicionário, anotações e atas
scripts/                 Geração de documentos e relatórios
src/data_loader.py       Download, leitura e junção dos dados
src/features/            Engenharia de features Pix simuladas
src/models/              Estrutura reservada para os modelos
src/rag/                 Laboratório LangChain, ingestão, embeddings e FAISS
tests/                   Testes das entregas de junho e julho
```

O organizador colaborativo de tarefas é um projeto separado; este repositório reúne os artefatos acadêmicos e o código do TCC.

## Monografia e contribuição

A entrada LaTeX é [monografia/main.tex](monografia/main.tex), com fontes em [referencias.bib](monografia/referencias.bib). Siga o [guia do Overleaf](monografia/README.md), confirme o template da instituição e compile os capítulos antes da entrega final.

Trabalhe em branches, faça commits focados e peça revisão da outra pessoa antes de integrar à `main`. As convenções e os critérios de revisão estão em [CONTRIBUTING.md](CONTRIBUTING.md).
