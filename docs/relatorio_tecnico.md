# Resultados Obtidos para o Módulo 1

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

---

# Resultados Obtidos para o Módulo 2

## 1. Implementação do algoritmo genético para otimização de hiperparâmetros

Para o Projeto 1, foi implementado um algoritmo genético simples, porém funcional, com o objetivo de otimizar hiperparâmetros de modelos de classificação baseados em scikit-learn. A implementação foi organizada no módulo `ga_optimizer.py` e contemplou os elementos tradicionais de um algoritmo evolutivo:

- codificação dos genes como dicionário de parâmetros;
- geração inicial de indivíduos com valores válidos dentro do espaço de busca;
- seleção por torneio;
- cruzamento uniforme;
- mutação por reamostragem de genes;
- elitismo para preservar a melhor solução;
- função de fitness ponderada com ênfase em recall, em razão da importância de reduzir falsos negativos em contextos clínicos.

Os hiperparâmetros otimizados incluíram:

- para Random Forest: `n_estimators`, `max_depth`, `min_samples_split` e `max_features`;
- para Regressão Logística: `C`.

A função de fitness avaliou cada indivíduo com base em métricas de validação, priorizando recall, accuracy e F1-score.

## 2. Resultados da otimização

Foram executados três experimentos com diferentes configurações do algoritmo genético, variando tamanho da população e taxa de mutação:

- Experimento 1: população 8, gerações 6, mutação 0.10;
- Experimento 2: população 12, gerações 8, mutação 0.05;
- Experimento 3: população 16, gerações 6, mutação 0.15.

Os resultados indicaram que o algoritmo foi capaz de encontrar combinações de hiperparâmetros com melhor desempenho na validação. Em particular, o Random Forest alcançou desempenho de validação bastante elevado em algumas execuções, enquanto a Regressão Logística apresentou ganhos consistentes de recall e F1-score.

## 3. Comparativo entre modelos originais e otimizados

A comparação foi realizada entre os modelos baseline e os modelos produzidos após a otimização.

### Modelos originais

- Regressão Logística: accuracy 0.9649, precision 0.9750, recall 0.9286, F1 0.9512.
- Random Forest: accuracy 0.9737, precision 1.0000, recall 0.9286, F1 0.9630.

### Modelos otimizados

- Regressão Logística otimizada: accuracy 0.9737, precision 0.9756, recall 0.9524, F1 0.9639.
- Random Forest otimizado: accuracy 0.9649, precision 1.0000, recall 0.9048, F1 0.9500.

Esses resultados demonstram que a otimização trouxe ganho para a Regressão Logística no conjunto de teste, enquanto o Random Forest apresentou melhor desempenho na validação, mas não manteve o mesmo ganho em teste, sugerindo possível sobreajuste em algumas configurações.

## 4. Integração com LLMs para interpretação de resultados

A integração com LLMs foi implementada para transformar métricas técnicas em explicações acionáveis em linguagem natural. O módulo `llm_interface.py` foi responsável por:

- construir prompts em português para contextualizar os resultados para profissionais de saúde;
- incluir métricas como accuracy, recall e F1-score;
- destacar características mais relevantes do modelo, como variáveis associadas à malignidade;
- gerar resumos claros, com recomendações e observações de incerteza.

### Abordagem adotada

O prompt foi estruturado para orientar a LLM a responder com:

1. um resumo conciso dos resultados;
2. interpretação clínica das métricas;
3. fatores de risco e variáveis mais relevantes;
4. recomendações acionáveis;
5. limitações e incertezas do modelo.

Essa abordagem foi pensada para aproximar a saída da solução de um contexto médico real, sem substituir a avaliação profissional.

### Avaliação da qualidade das interpretações

Como não havia uma referência humana disponível para comparação direta, a qualidade foi avaliada de forma heurística. Foram considerados critérios como:

- sobreposição entre as features citadas e as variáveis mais relevantes do problema;
- presença de recomendações explícitas;
- comprimento, clareza e utilidade do texto gerado.

Na execução realizada, a interpretação gerada apresentou boa sobreposição com as features mais relevantes e incluiu recomendações acionáveis, indicando potencial de uso para suporte à análise de resultados.

## 5. Desafios enfrentados e soluções implementadas

Durante o desenvolvimento, alguns desafios foram identificados:

- variabilidade do algoritmo genético, que depende da inicialização aleatória;
- risco de overfitting em algumas configurações otimizadas;
- dependência de chave de API para uso completo da LLM;
- necessidade de manter os resultados interpretáveis para o usuário final.

As soluções adotadas foram:

- uso de `random_state` fixo para maior reprodutibilidade;
- avaliação com dados de validação e posterior comparação com teste;
- implementação de fallback local quando não há chave de API disponível;
- registro de métricas e logs para rastrear os experimentos.

## 6. Arquitetura de escalabilidade e monitoramento

A implementação atual contemplou monitoramento local e logs estruturados por meio do módulo `monitoramento.py`. Esse módulo cria diretórios para logs e métricas, registra eventos de execução e salva métricas em formato JSONL.

A implementação em nuvem não foi priorizada nesta entrega, mas a documentação disponível em `docs/arquitetura_scalabilidade.md` apresenta uma proposta de arquitetura para produção, com:

- containerização da aplicação;
- orquestração com Kubernetes e autoscaling;
- uso de Prometheus e Grafana para métricas;
- logging centralizado;
- armazenamento persistente de artefatos e modelos.

Essa abordagem preserva a viabilidade da solução para futuras expansões sem comprometer a execução local do protótipo.

## 7. Considerações finais do módulo

O módulo desenvolvido atendeu ao escopo proposto ao combinar otimização automática, avaliação comparativa e interpretação baseada em LLM. A solução demonstrou que é possível melhorar modelos de diagnóstico de forma automatizada, registrar o processo de forma reproduzível e transformar resultados técnicos em insights mais acessíveis para o usuário final.