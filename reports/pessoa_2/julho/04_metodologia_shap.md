# Metodologia da Camada 2 — SHAP

Data da redação: **30/08/2026**. Escopo: `m3_p2_5`.

## Objetivo

Definir como as explicações SHAP serão calculadas, verificadas e transformadas em evidência para o RAG. Este documento descreve protocolo; não apresenta valores SHAP nem métricas de modelos que ainda não foram entregues pela Pessoa 1.

## Contrato da camada

**Entrada:** pipeline de pré-processamento congelado, modelo treinado, linha transformada, conjunto de referência, versão das features e escala da saída explicada.

**Saída por transação:** identificador pseudonimizado, valor-base, saída do modelo, contribuição por atributo, valor observado, direção, unidade, versão do modelo, versão do explicador e advertências de domínio.

A verificação de consistência deve testar a propriedade aditiva na escala escolhida:

```text
saída_do_modelo ≈ valor_base + soma(contribuições_SHAP)
```

Para modelos de árvore, será usado `shap.TreeExplainer`. A configuração inicial será explícita: `feature_perturbation="interventional"`, fundo com 500 linhas amostradas somente do treino por seed registrada e `model_output="probability"`. A escala de probabilidade foi escolhida por ser mais compreensível na apresentação e por permitir conferir a soma diretamente contra a probabilidade prevista. O tamanho do fundo fica dentro da faixa de 100 a 1.000 exemplos recomendada na documentação do explicador; qualquer alteração deverá ocorrer antes da avaliação final e ser registrada.

Não serão usados os padrões implícitos da biblioteca, pois o comportamento de `feature_perturbation="auto"` depende da presença do conjunto de fundo e mudou entre versões do SHAP. A versão instalada será registrada. No Random Forest, a saída possui uma dimensão por classe; a classe positiva será localizada por `model.classes_ == 1`, sem presumir uma posição fixa. Contribuições de classes ou escalas diferentes não serão comparadas como se fossem equivalentes.

## Por que SHAP foi escolhido

- produz uma explicação local aditiva que pode ser verificada numericamente;
- permite obter uma visão global agregando a magnitude das contribuições locais;
- possui o `TreeExplainer`, algoritmo eficiente e exato para XGBoost e Random Forest sob a configuração declarada;
- separa a previsão do classificador da posterior contextualização documental do RAG;
- oferece uma estrutura comum para auditar acertos e erros, sem transformar importância em causalidade.

## Protocolo

1. congelar split, pré-processamento, features, modelo e limiar;
2. amostrar e congelar 500 linhas de fundo somente do treino, com seed registrada;
3. construir o explicador com `feature_perturbation="interventional"` e `model_output="probability"`;
4. calcular SHAP para uma amostra fixa do teste e para casos representativos: verdadeiro positivo, falso positivo, falso negativo e verdadeiro negativo;
5. verificar aditividade, valores ausentes, ordem das colunas, classe positiva e estabilidade numérica;
6. reportar visão global por média de `|SHAP|`, sem confundi-la com causalidade;
7. reportar explicações locais com valor observado, sinal e magnitude;
8. comparar XGBoost e Random Forest apenas após ambos serem executados no mesmo protocolo;
9. registrar limitações e exemplos em que a explicação não é semanticamente interpretável.

## Ponte SHAP → RAG

Somente features com conceito aprovado no [`../../../config/pix_feature_registry.json`](../../../config/pix_feature_registry.json) podem virar tags de busca documental. O pacote enviado ao retriever deve conter:

- `feature_id` e versão do registro;
- valor observado e unidade;
- contribuição e escala explicada;
- tag conceitual aprovada;
- advertência de que IEEE-CIS não é Pix real.

Features anônimas (`V*`, `C*`, `D*`, `M*`, `id_*`) podem aparecer como importantes para o classificador, mas devem ser narradas como “atributo anonimizado de alta influência”. SHAP não revela seu significado e não autoriza associá-las a conta, chave, dispositivo ou intenção de fraude.

## Avaliação

| Dimensão | Verificação |
|---|---|
| Fidelidade numérica | valor-base + contribuições reconstrói a saída dentro de tolerância |
| Alinhamento | nome, valor e posição da feature coincidem com o pipeline treinado |
| Estabilidade | pequenas perturbações plausíveis não geram narrativa contraditória sem alerta |
| Cobertura | análise inclui acertos e erros das duas classes |
| Segurança semântica | nenhuma causalidade, acusação ou significado anônimo é inventado |
| Integração | tags enviadas ao RAG pertencem ao registro aprovado |

## Limitações obrigatórias no texto final

- SHAP explica o comportamento do modelo, não a causa real da transação.
- Importância global não substitui explicação local.
- Atributos correlacionados podem repartir contribuição de maneira sensível à configuração do explicador.
- Uma explicação fiel pode expor um modelo errado, enviesado ou mal calibrado.
- O domínio IEEE-CIS é comércio eletrônico/cartão e não valida desempenho operacional no Pix.

Referências metodológicas: Lundberg e Lee (2017), incluída em `monografia/referencias.bib` como `lundberg2017unified`, e a [documentação oficial do `TreeExplainer`](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html), consultada em 10/09/2026.
