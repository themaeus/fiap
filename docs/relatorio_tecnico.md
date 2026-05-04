# Relatório Técnico - Tech Challenge A - Fase 1

## 1. Discussão da Análise Exploratória

Foi utilizada a base Wisconsin Breast Cancer Dataset (569 registros), com classe alvo binária: benigno (357; 62.74%) e maligno (212; 37.26%).

Na exploração, foram avaliadas distribuição de classes, estatísticas descritivas, boxplots de variáveis principais e matriz de correlação.

As evidências indicaram que atributos morfológicos ligados a tamanho e irregularidade da massa apresentaram maior associação com malignidade. Entre os mais relevantes na correlação com o alvo: `concave points_worst` (pontos de concavidade - pior valor), `perimeter_worst` (perímetro - pior valor), `concave points_mean` (média de pontos de concavidade), `radius_worst` (raio - pior valor) e `perimeter_mean` (média do perímetro).

Interpretação da Análise Exploratória de Dados: o problema possui padrões claros que permitem separação entre as classes, sustentando o uso de modelos supervisionados de classificação.

## 2. Estratégias de Pré-processamento

As principais estratégias aplicadas foram:

- limpeza da base com remoção de colunas sem valor para modelagem (`id` e `Unnamed: 32`);
- transformação do alvo para formato binário (`B -> 0`, `M -> 1`);
- separação treino/teste com estratificação (80/20) para preservar proporção entre classes;
- pipeline com imputação por mediana e padronização (`StandardScaler`) para garantir consistência e evitar vazamento de dados.

Essa estratégia tornou o fluxo reproduzível e adequado para comparar modelos em condições equivalentes.

## 3. Modelos Usados e Porquê's

Foram usados dois modelos:

1. Regressão Logística
2. Random Forest

Motivação da escolha:

- Regressão Logística: baseline forte, simples e com boa interpretabilidade para classificação binária.
- Random Forest: modelo não linear, robusto para capturar relações complexas e com interpretação por importância de variáveis.

A combinação permitiu comparar um modelo linear e um modelo baseado em árvores, equilibrando simplicidade e desempenho.

## 4. Resultados e Interpretação dos Dados

Resultados no conjunto de teste:

| Modelo | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Regressão Logística | 0.9649 | 0.9750 | 0.9286 | 0.9512 |
| Random Forest | 0.9737 | 1.0000 | 0.9286 | 0.9630 |

Interpretação:

- Os dois modelos atingiram o mesmo recall para classe malígna (0.9286), métrica prioritária neste problema por reduzir risco de falso negativo.
- O Random Forest apresentou melhor desempenho global (accuracy, precision e F1-score), sendo a melhor opção técnica para este recorte.
- As técnicas de explicabilidade (feature importance e SHAP) confirmaram que variáveis de tamanho e concavidade são as mais influentes na predição.

Conclusão: o pipeline atendeu ao objetivo do desafio e demonstrou viabilidade de apoio à triagem, mantendo o médico como decisor final.