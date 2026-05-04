# Resultados Obtidos - Tech Challenge A - Fase 1

## Base de Dados

- Base utilizada: Wisconsin Breast Cancer Dataset.
- Total de registros: 569.
- Classes: 357 benignos e 212 malignos.

## Modelos avaliados

### Regressão Logística

- Accuracy: 0.9649
- Precision: 0.9750
- Recall: 0.9286
- F1-score: 0.9512

### Random Forest

- Accuracy: 0.9737
- Precision: 1.0000
- Recall: 0.9286
- F1-score: 0.9630

## Comparação

- Os dois modelos tiveram o mesmo recall para a classe maligna no conjunto de teste.
- O Random Forest teve melhor accuracy, precision e F1-score.
- Como o critério principal do trabalho é reduzir falsos negativos em contexto médico, o recall da classe maligna deve ser destacado no relatório.
- Como houve empate no recall, o Random Forest pode ser apresentado como o melhor equilíbrio geral de desempenho.

## Variáveis mais relevantes

Pelos resultados de correlação e importância das variáveis, os atributos mais associados à malignidade incluem:

- concave points_worst (pontos de concavidade - pior valor)
- perimeter_worst (perímetro - pior valor)
- concave points_mean (média de pontos de concavidade)
- radius_worst (raio - pior valor)
- perimeter_mean (média do perímetro)
- area_worst (área - pior valor)
- radius_mean (média do raio)
- area_mean (média da área)
- concavity_mean (média da concavidade)

## Discussão

- O modelo pode apoiar triagem e priorização de casos, mas não substituir o diagnóstico médico.
- O custo de um falso negativo é alto, por isso o recall da classe maligna é central.
- A base é bem conhecida e útil para prototipagem, mas não representa toda a diversidade clínica do mundo real.
- Para uso prático, seriam necessárias validação externa, governança de dados e avaliação ética.