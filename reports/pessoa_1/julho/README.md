# Julho — modelos e explicabilidade (Pessoa 1)

As cinco tarefas de **XGBoost, Random Forest e SHAP** estão implementadas. Os
resultados abaixo usam o corte temporal 70/15/15 e o mesmo pré-processamento
ajustado exclusivamente no treino.

## Comparação oficial na validação

| Modelo | AUC-PR | AUC-ROC | Recall | Precisão | F1 |
|---|---:|---:|---:|---:|---:|
| Regressão logística (baseline) | 0,3930 | 0,8410 | 0,6772 | 0,1350 | 0,2251 |
| Random Forest (árvores completas) | 0,5298 | 0,9078 | 0,3192 | 0,8018 | 0,4566 |
| **XGBoost ajustado (Optuna)** | **0,5703** | 0,9070 | **0,5621** | 0,5198 | **0,5401** |

A AUC-PR é a comparação principal porque independe do limiar. Precisão, recall
e F1 usam o corte 0,5 e descrevem pontos de operação/calibração diferentes.
O conjunto de teste não foi usado nesta etapa.

O notebook executado com a tabela, o gráfico e a leitura dos resultados está em
[`notebooks/03_models.ipynb`](../../../notebooks/03_models.ipynb).

## SHAP no XGBoost

Protocolo executado:

- `shap.TreeExplainer` com `feature_perturbation="interventional"`;
- contribuições em escala de probabilidade (`model_output="probability"`);
- fundo de **500 linhas amostradas somente do treino**, sem redução silenciosa;
- importância global em 1.000 linhas aleatórias da validação;
- casos locais VP, FP, FN e VN escolhidos pela mediana de probabilidade de cada
  quadrante;
- seed 42 e limiar 0,5.

A fidelidade aditiva foi aprovada: o maior erro em
`valor_base + soma(SHAP) = probabilidade` foi **3,04 × 10⁻⁷**, abaixo da
tolerância de 10⁻⁵.

### Top 10 global

| Posição | Feature | média de \|SHAP\| |
|---:|---|---:|
| 1 | `TransactionAmt` | 0,016792 |
| 2 | `C13` | 0,014698 |
| 3 | `C1` | 0,010494 |
| 4 | `C14` | 0,008528 |
| 5 | `card1` | 0,008383 |
| 6 | `C5` | 0,007558 |
| 7 | `valor_atipico_proxy` | 0,006856 |
| 8 | `card2` | 0,006801 |
| 9 | `D1` | 0,006671 |
| 10 | `P_emaildomain` | 0,006519 |

`valor_atipico_proxy` aparece em 7º e `frequencia_recente_proxy` em 13º. Isso
mostra influência no modelo; não transforma colunas anônimas em conceitos Pix
nem estabelece causalidade.

### Artefatos

- [Relatório estruturado](shap/shap_xgboost.json)
- [Summary plot](shap/shap_summary_xgboost.png)
- [Force plot — verdadeiro positivo](shap/shap_force_verdadeiro_positivo.png)
- [Force plot — falso positivo](shap/shap_force_falso_positivo.png)
- [Force plot — falso negativo](shap/shap_force_falso_negativo.png)
- [Force plot — verdadeiro negativo](shap/shap_force_verdadeiro_negativo.png)

As versões HTML interativas dos quatro force plots estão no mesmo diretório.

## Reprodutibilidade e compatibilidade

```powershell
$env:TCC_DATA_RAW = "C:\caminho\para\ieee-fraud-detection"
.\.venv\Scripts\python.exe -m scripts.gerar_shap_xgboost
```

`TCC_DATA_RAW` é opcional; sem ele, o loader usa `data/raw/`.

O ambiente histórico do tuning usou XGBoost 3.4.1. Essa versão, combinada ao
SHAP 0.52, marca incorretamente matrizes já codificadas como categóricas e
recusa `TreeExplainer` interventional. A execução SHAP foi validada com XGBoost
3.0.5, e `requirements.txt` impede instalar a combinação incompatível. O modelo
final foi reconstruído com os parâmetros já selecionados, sem repetir ou
consultar o Optuna, e obteve AUC-PR 0,5719. Esse é o artefato congelado usado no
SHAP e na avaliação final. O valor **0,5703** permanece identificado como
resultado histórico do retreino do tuning, sem ser atribuído ao modelo entregue.

O XGBoost aprovado, seu pré-processador, manifesto e política de decisão são
versionados em `models/xgboost/`. O Random Forest e os CSVs permanecem fora do
Git. O manifesto registra os hashes do modelo e do pré-processador usados nas
explicações.

## Limitações

- O IEEE-CIS representa cartão/comércio eletrônico, não Pix real.
- SHAP explica o comportamento do classificador, não a correção da decisão.
- Features `V*`, `C*`, `D*` e identificadores mascarados não recebem significado
  semântico por aparecerem no ranking.
- O limiar 0,6834220290 foi escolhido por F1 na validação e congelado antes da
  avaliação única no teste.
