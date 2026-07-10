# Resultados Obtidos para o Módulo 1

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

---

# Resultados Obtidos para o Módulo 2

## Otimização via Algoritmo Genético (entregável solicitado)

- Implementação: Algoritmo Genético para otimização de hiperparâmetros com codificação por dicionário de parâmetros; seleção por torneio (k=3); cruzamento uniforme; mutação por reamostragem do gene; elitismo; função de fitness ponderada com ênfase em recall.
- Arquivos de resultado: [outputs_quick/ga_resultados_quick_sumario.json](outputs_quick/ga_resultados_quick_sumario.json) (sumário) e [outputs_quick/ga_experimento_quick_1.json](outputs_quick/ga_experimento_quick_1.json), [outputs_quick/ga_experimento_quick_2.json](outputs_quick/ga_experimento_quick_2.json), [outputs_quick/ga_experimento_quick_3.json](outputs_quick/ga_experimento_quick_3.json).

### Resumo dos achados

- Baseline (modelos originais treinados no treino completo): `LogisticRegression` (recall 0.9286), `RandomForest` (recall 0.9286).
- Experimentos GA (3 configurações): o GA melhorou métricas na validação (ex.: RF val accuracy 0.9890). No conjunto de teste, o `LogisticRegression` otimizado apresentou ganho real (teste: accuracy 0.9737, recall 0.9524, f1 0.9639). O `RandomForest` otimizado mostrou melhor validação, porém no teste manteve recall inferior ao baseline (teste recall 0.9048), sugerindo possível sobreajuste.

### Conclusão (relativa ao comando)

- Todos os requisitos do enunciado foram atendidos:
	- Implementado Algoritmo Genético para otimização de hiperparâmetros (`ga_optimizer.py`).
	- Foram executados 3 experimentos com diferentes configurações do GA.
	- Foi realizada a comparação entre modelos originais e otimizados, com resultados salvos em `outputs_quick/`.

> Observação: mantive o relatório focado apenas no que foi pedido. Se desejar, posso gerar explicações SHAP e salvar modelos otimizados (fora do escopo pedido).