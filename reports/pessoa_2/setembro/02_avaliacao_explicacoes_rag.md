# Avaliação das explicações RAG

A avaliação deve ser executada sobre um conjunto fixo de `{{PREENCHER}}` transações, preservando a predição, os valores SHAP, os trechos recuperados e a resposta integral do LLM. O valor de `k`, o limiar mínimo de similaridade, a versão do índice, o modelo de linguagem, o prompt e os parâmetros de geração devem ser registrados antes do experimento. Os avaliadores e o procedimento de resolução de divergências devem ser definidos como `{{PREENCHER}}`.

| Critério | Forma de avaliação | Métrica | Resultado |
|---|---|---|---|
| Relevância dos documentos recuperados | Para cada consulta, avaliadores marcam previamente quais chunks do corpus são relevantes e comparam essas marcações com os `k` primeiros resultados retornados. | `Precision@k`, `Recall@k` e `MRR` (inverso da posição do primeiro resultado relevante) | `{{RESULTADO}}` |
| Coerência da explicação com os valores SHAP | Conferência item a item entre as features citadas, seus valores, o sinal da contribuição e a direção descrita no texto. Atributos anonimizados não podem receber significado de negócio. | Percentual de explicações sem contradição com o pacote SHAP; concordância por campo (feature, valor e sinal) | `{{RESULTADO}}` |
| Fidelidade aos dados da transação | Decomposição da resposta em afirmações verificáveis e comparação com a entrada, a classe, a probabilidade e o limiar efetivamente produzidos pelo pipeline. | Taxa de afirmações factualmente corretas sobre a transação e a predição | `{{RESULTADO}}` |
| Ausência de alucinações | Identificação de afirmações factuais sem suporte na transação, no pacote SHAP ou nos chunks recuperados. Limitações e abstenções corretas não contam como erro. | Taxa de alucinação = afirmações sem suporte / total de afirmações factuais | `{{RESULTADO}}` |
| Clareza da explicação em português | Avaliação cega por `{{PREENCHER}}` avaliadores, usando rubrica de 1 a 5 para objetividade, legibilidade, organização e adequação terminológica. | Média da rubrica de clareza e concordância entre avaliadores por `{{PREENCHER}}` | `{{RESULTADO}}` |

Os resultados devem ser apresentados separadamente por tipo de caso analisado e acompanhados do número de observações. O teste semântico de fumaça já registrado no projeto demonstra funcionamento técnico da recuperação, mas não substitui esta avaliação experimental.
