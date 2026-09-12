"""
fase3/run_demo.py

Script de Demonstração Interativa - Fase 3 Tech Challenge.
Executa todo o ecossistema:
1. Inicialização e Seed do Banco de Dados SQLite.
2. Preparação do Dataset Médico Anonimizado.
3. Demonstração de Fine-Tuning e conversão JSONL / Modelfile.
4. Execução do fluxo completo do LangGraph para diferentes casos clínicos (Dor Torácica e Sepse).
5. Exibição do Audit Log JSONL.
"""

import os
import sys
import json

from fase3.database import MedicalDatabase
from fase3.dataset_prep import build_synthetic_medical_dataset, save_dataset
from fase3.fine_tuning import MedicalFineTuner
from fase3.langgraph_workflow import MedicalLangGraphWorkflow


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" 🏥 HOSPITAL VIRTUAL - FASE 3: {title}")
    print("=" * 80)


def run_full_demonstration():
    # --- ETAPA 1: BANCO DE DADOS ---
    print_banner("ETAPA 1: Inicialização do Banco de Dados SQLite de Prontuários")
    db = MedicalDatabase()
    db.seed_sample_data()

    # --- ETAPA 2: DATASET & ANONIMIZAÇÃO ---
    print_banner("ETAPA 2: Curadoria e Anonimização de Dados Médicos (PII Removal)")
    dataset = build_synthetic_medical_dataset()
    save_dataset(dataset)
    print(f"✓ Dataset sintético criado com {len(dataset)} protocolos/FAQs sem dados sensíveis.")

    # --- ETAPA 3: FINE-TUNING & CUSTOMIZAÇÃO LLM ---
    print_banner("ETAPA 3: Pipeline de Fine-Tuning & Formatação para LLM Customizada")
    ft = MedicalFineTuner()
    jsonl_path = ft.prepare_openai_jsonl()
    modelfile_path = ft.generate_ollama_modelfile()
    print(f"✓ Artefato JSONL gerado em: {jsonl_path}")
    print(f"✓ Modelfile Ollama gerado em: {modelfile_path}")

    # --- ETAPA 4: LANGGRAPH & LANGCHAIN WORKFLOW ---
    print_banner("ETAPA 4: Execução de Casos Clínicos no LangGraph com Guardrails & Explainability")
    app = MedicalLangGraphWorkflow()

    # Caso 1: Paciente PAC-101 (Dor Torácica / Cardiologia)
    print("\n--------------------------------------------------------------------------------")
    print("▶ CASO CLÍNICO 1: Paciente PAC-101 (Leito 204-A) - Suspeita de Síndrome Coronariana")
    print("--------------------------------------------------------------------------------")
    res_1 = app.run(
        paciente_id="PAC-101",
        query_medica="Qual o protocolo de conduta para o paciente com ECG alterado e dor no peito?"
    )
    if res_1.get("alertas"):
        for al in res_1["alertas"]:
            print(f"\n{al}")
        print()
    print(res_1["resposta_final"])

    # Caso 2: Paciente PAC-202 (Sepse / UTI)
    print("\n--------------------------------------------------------------------------------")
    print("▶ CASO CLÍNICO 2: Paciente PAC-202 (UTI Leito 03) - Suspeita de Sepse e Hipotensão")
    print("--------------------------------------------------------------------------------")
    res_2 = app.run(
        paciente_id="PAC-202",
        query_medica="Quais são as condutas imediatas no pacote de 1 hora para suspeita de Sepse?"
    )
    if res_2.get("alertas"):
        for al in res_2["alertas"]:
            print(f"\n{al}")
        print()
    print(res_2["resposta_final"])

    # --- ETAPA 5: AUDITORIA E LOGS STRUCTURADOS ---
    print_banner("ETAPA 5: Verificação do Audit Log Estruturado (JSONL)")
    audit_file = "outputs/audit_log.jsonl"
    if os.path.exists(audit_file):
        with open(audit_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"✓ Total de registros de auditoria gravados: {len(lines)}")
        print("\nExemplo do último registro de log de auditoria:")
        print(json.dumps(json.loads(lines[-1]), ensure_ascii=False, indent=2))

    print_banner("DEMONSTRAÇÃO CONCLUÍDA COM SUCESSO!")


if __name__ == "__main__":
    run_full_demonstration()
