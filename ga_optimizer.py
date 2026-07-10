from copy import deepcopy
import random
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score


def _avaliar_metricas(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }


class OtimizadorGA:
    """Algoritmo Genético simples para otimização de hiperparâmetros de estimadores scikit-learn.

    Recebe um callable `construir_modelo(params)` que retorna um estimador não ajustado
    a partir de um dicionário de parâmetros. A função de fitness é soma ponderada de
    métricas (por padrão, enfatiza recall).
    """

    def __init__(self, construir_modelo, espaco_params, tam_populacao=30, geracoes=20,
                 taxa_cruzamento=0.8, taxa_mutacao=0.1, pesos_fitness=None,
                 torneio_k=3, random_state=None):
        self.construir_modelo = construir_modelo
        self.espaco_params = espaco_params
        self.tam_populacao = tam_populacao
        self.geracoes = geracoes
        self.taxa_cruzamento = taxa_cruzamento
        self.taxa_mutacao = taxa_mutacao
        self.torneio_k = torneio_k
        self.random_state = random_state or 42
        self.rng = random.Random(self.random_state)
        if pesos_fitness is None:
            # enfatiza recall
            self.pesos_fitness = {'accuracy': 0.2, 'recall': 0.6, 'f1_score': 0.2}
        else:
            self.pesos_fitness = pesos_fitness

    def _gene_aleatorio(self, nome, spec):
        if spec['type'] == 'int':
            return self.rng.randint(spec['min'], spec['max'])
        if spec['type'] == 'float':
            return self.rng.uniform(spec['min'], spec['max'])
        if spec['type'] == 'cat':
            return self.rng.choice(spec['choices'])
        raise ValueError('Tipo de gene desconhecido')

    def _iniciar_populacao(self):
        pop = []
        for _ in range(self.tam_populacao):
            indiv = {}
            for k, spec in self.espaco_params.items():
                indiv[k] = self._gene_aleatorio(k, spec)
            pop.append(indiv)
        return pop

    def _mutacao(self, indiv):
        filho = deepcopy(indiv)
        for k, spec in self.espaco_params.items():
            if self.rng.random() < self.taxa_mutacao:
                filho[k] = self._gene_aleatorio(k, spec)
        return filho

    def _cruzamento(self, a, b):
        # cruzamento uniforme
        filho = {}
        for k in self.espaco_params.keys():
            filho[k] = a[k] if self.rng.random() < 0.5 else b[k]
        return filho

    def _torneio(self, populacao, pontuacoes):
        escolhidos = self.rng.sample(list(range(len(populacao))), self.torneio_k)
        melhor = max(escolhidos, key=lambda i: pontuacoes[i])
        return deepcopy(populacao[melhor])

    def _fitness_de_metricas(self, metricas):
        return sum(metricas[k] * self.pesos_fitness.get(k, 0) for k in metricas)

    def fit(self, X_treino, y_treino, X_val, y_val, verbose=False):
        """Executa a otimização e retorna um dict com melhores parâmetros, métricas e modelo final."""
        populacao = self._iniciar_populacao()
        melhor_solucao = None
        melhor_pontuacao = -np.inf
        historico = []

        for ger in range(self.geracoes):
            pontuacoes = []
            lista_metricas = []
            for indiv in populacao:
                modelo = self.construir_modelo(indiv)
                try:
                    modelo.fit(X_treino, y_treino)
                    preds = modelo.predict(X_val)
                except Exception:
                    preds = np.zeros_like(y_val)
                metricas = _avaliar_metricas(y_val, preds)
                fitness = self._fitness_de_metricas(metricas)
                pontuacoes.append(fitness)
                lista_metricas.append({'params': indiv, 'metricas': metricas, 'fitness': fitness})

            idx_melhor = int(np.argmax(pontuacoes))
            melhor_ger = lista_metricas[idx_melhor]
            historico.append(melhor_ger)
            if melhor_ger['fitness'] > melhor_pontuacao:
                melhor_pontuacao = melhor_ger['fitness']
                melhor_solucao = deepcopy(melhor_ger)

            if verbose:
                print(f'Geração {ger+1}/{self.geracoes} melhor fitness {melhor_ger["fitness"]:.4f}')

            # próxima geração
            nova_pop = []
            # elitismo: mantém o melhor
            elite_idx = int(np.argmax(pontuacoes))
            nova_pop.append(deepcopy(populacao[elite_idx]))

            while len(nova_pop) < self.tam_populacao:
                pai_a = self._torneio(populacao, pontuacoes)
                pai_b = self._torneio(populacao, pontuacoes)
                if self.rng.random() < self.taxa_cruzamento:
                    filho = self._cruzamento(pai_a, pai_b)
                else:
                    filho = deepcopy(pai_a)
                filho = self._mutacao(filho)
                nova_pop.append(filho)

            populacao = nova_pop

        # treina modelo final com melhores parâmetros em treino+val
        melhores_params = melhor_solucao['params']
        modelo_final = self.construir_modelo(melhores_params)
        # se X_treino for DataFrame, concatena mantendo nomes de colunas
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        if _pd is not None and hasattr(X_treino, 'columns'):
            X_comb = _pd.concat([X_treino, X_val], axis=0)
            if hasattr(y_treino, 'reset_index'):
                y_comb = _pd.concat([y_treino, y_val], axis=0)
            else:
                y_comb = np.concatenate([y_treino, y_val])
        else:
            X_comb = np.vstack([X_treino, X_val])
            y_comb = np.concatenate([y_treino, y_val])

        modelo_final.fit(X_comb, y_comb)

        return {'best_params': melhores_params, 'best_metrics': melhor_solucao['metricas'], 'best_fitness': melhor_pontuacao, 'model': modelo_final, 'history': historico}
