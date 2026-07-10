# Instruções de Execução para o Módulo 1

## 1. Pré-requisitos

- Python 3.11+
- Ambiente virtual configurado (exemplo: `.venv`)

## 2. Instalar dependências

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. Executar o notebook

1. Abra `notebooks/tech_challenge_a.ipynb`.
2. Selecione o kernel `.venv (Python 3.11.x)`.
3. Execute todas as células em ordem.

## 4. Saídas geradas

- Notebook atualizado com tabelas e resultados em `notebooks/tech_challenge_a.ipynb`.
- Gráficos em `outputs/`.

---

# Instruções de Execução para o Módulo 2

## 5. Módulo 2 - Escalabilidade, monitoramento e interpretação

### 5.1 O que foi adicionado neste módulo

Este módulo amplia a solução do Projeto 1 com:

- monitoramento e logging estruturados;
- registro de métricas em arquivos JSONL;
- geração de interpretações em linguagem natural via LLM;
- documentação de arquitetura e decisões de escalabilidade.

### 5.2 Como executar o módulo 2

#### Passo 1: executar os experimentos principais

```bash
python run_experiments_quick.py
```

Esse comando executa:
- o carregamento do dataset;
- o treino de modelos baseline;
- a execução de 3 experimentos com algoritmo genético;
- a geração de arquivos de saída em `outputs_quick/`.

#### Passo 2: testar a interpretação gerada

```bash
python test_llm_call.py
```

Esse comando gera uma interpretação em linguagem natural com base nos resultados obtidos.

### 5.3 Saídas esperadas

- resultados dos experimentos em `outputs_quick/`;
- métricas e logs em `outputs/`;
- documentação técnica em `docs/`.
