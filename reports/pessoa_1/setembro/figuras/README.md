# Figuras finais em 300 dpi (m5_p1_3)

Geradas por `python -m scripts.gerar_figuras_finais` a partir do XGBoost entregue em `models/xgboost`, sem retreino e com o limiar congelado em `politica_decisao.json` (0,6834). Antes de gravar, o script confere os hashes do modelo e recalcula as métricas do teste; se algo divergir de [avaliacao_xgboost_entregue.json](../../../avaliacao_xgboost_entregue.json), nenhuma figura é gravada. O resultado da conferência está em [figuras_finais.json](figuras_finais.json).

As imagens não têm título interno. Pela ABNT, o título vai acima da figura, no texto, e a fonte abaixo.

| Arquivo | Legenda sugerida | Onde usar |
|---|---|---|
| `fig_arquitetura.png` | Figura – Arquitetura do protótipo híbrido | Metodologia (4.8) |
| `fig_distribuicao_alvo.png` | Figura – Distribuição da variável alvo no conjunto IEEE-CIS (590.540 transações) | Base de dados (4.1) |
| `fig_top_correlacoes.png` | Figura – Dez variáveis com maior correlação de Pearson com isFraud | Base de dados (4.1), opcional |
| `fig_curva_pr.png` | Figura – Curvas de precisão-recall do XGBoost na validação e no teste | Resultados |
| `fig_matriz_confusao_teste.png` | Figura – Matriz de confusão do XGBoost no teste (limiar 0,6834) | Resultados |
| `shap/shap_summary_xgboost.png` | Figura – Importância global SHAP do XGBoost na validação | Resultados (SHAP) |
| `fig_shap_casos.png` | Figura – Contribuições SHAP em quatro casos da validação (VP, FP, FN e VN) | Resultados (SHAP) |

Fonte para todas: elaborada pelos autores (2026).

Os force plots em `shap/` (PNG e HTML) ficam como material de apoio. Na monografia, prefira `fig_shap_casos.png`, que mostra as mesmas contribuições com valores arredondados e rótulos em português.

## Conferência

- Métricas do teste idênticas às publicadas: AUC-PR 0,4833, AUC-ROC 0,8700, precisão 0,5852, recall 0,4324 e F1 0,4973.
- Matriz de confusão idêntica: 1.333 fraudes detectadas, 1.750 não detectadas e 945 falsos alarmes.
- Top-10 da importância global SHAP igual ao relatório de julho.
