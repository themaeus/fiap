# Tech Challenge - Fase 3: Assistente Virtual Médico

Este repositório contém a solução completa para o **Tech Challenge da Fase 3** da pós-graduação em **IA para Devs**. A aplicação consiste em um **Assistente Médico Virtual** treinado com dados e protocolos internos do hospital, capaz de auxiliar na tomada de decisão em pronto-socorro, consultar prontuários estruturados e orquestrar fluxos clínicos seguros via **LangChain** e **LangGraph**.

---

## 1. Estrutura do Projeto

```
.
├── fase3/
│   ├── __init__.py
│   ├── app_gui.py            # Interface Gráfica Web em Streamlit
│   ├── assistente_medico.py   # Chain LangChain com Explainability (Citação de Fontes)
│   ├── audit_logger.py        # Gravação de logs estruturados (JSONL)
│   ├── database.py            # Gerenciador do banco de dados de prontuários (SQLite)
│   ├── dataset_prep.py        # Curadoria e remoção de dados sensíveis (PII Removal)
│   ├── fine_tuning.py         # Formatação JSONL e Modelfile para Fine-Tuning
│   ├── guardrails.py          # Bloqueio de prescrições autônomas e avisos legais
│   ├── langgraph_workflow.py  # Orquestração do fluxo de estados com LangGraph
│   └── run_demo.py            # Executável de demonstração via linha de comando
├── data/
│   ├── medical_protocols.json # Dataset de protocolos anonimizado
│   └── patient_db.sqlite      # Banco SQLite de prontuários
├── fine_tuning_artifacts/
│   ├── medical_ft_openai.jsonl# Dataset formatado para Fine-Tuning via OpenAI API
│   └── Modelfile.medical      # Definição de modelo local Ollama
├── docs/
│   └── relatorio_tecnico_fase3.md # Relatório técnico completo da Fase 3
├── outputs/
│   └── audit_log.jsonl        # Registro de auditoria estruturada
├── requirements.txt           # Dependências do projeto
└── README.md                  # Documentação principal
```

---

## 2. Pré-requisitos e Instalação

### 2.1. Criar ambiente virtual e instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Como Executar

### 3.1. Opção 1: Interface Gráfica Web (Streamlit - Recomendado)

```bash
PYTHONPATH=. python3 -m streamlit run fase3/app_gui.py
```

A interface será aberta automaticamente no seu navegador (`http://localhost:8501`), permitindo:
- Seleção visual de pacientes e prontuários;
- Banners de alerta visual para exames pendentes;
- Consultas interativas de apoio à decisão clínica com **LangGraph**;
- Abas para inspeção de logs de auditoria e artefatos de fine-tuning.

### 3.2. Opção 2: Demonstração via Linha de Comando (CLI)

```bash
PYTHONPATH=. python3 -m fase3.run_demo
```

Este comando executa a sequência completa de validação:
1. Povoamento do banco SQLite com prontuários de teste.
2. Sanitização do dataset médico de protocolos.
3. Conversão para artefatos de Fine-Tuning (JSONL / Modelfile).
4. Execução do grafo **LangGraph** para casos clínicos de emergência.
5. Gravação e verificação dos logs de auditoria em `outputs/audit_log.jsonl`.

---

## 4. Documentação Técnica

- **[docs/relatorio_tecnico_fase3.md](docs/relatorio_tecnico_fase3.md)**: Relatório técnico detalhado cobrindo fine-tuning, arquitetura LangChain, diagramas LangGraph e análise dos resultados.


