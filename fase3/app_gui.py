"""
fase3/app_gui.py

Interface Gráfica Interativa (Web UI) com Streamlit para o Assistente Virtual Médico.
Permite aos médicos e avaliadores:
- Selecionar pacientes e visualizar prontuários em tempo real.
- Executar o fluxo completo do LangGraph com alertas visuais de exames pendentes.
- Visualizar transparência de fontes (Explainability) e filtros de segurança (Guardrails).
- Inspecionar logs de auditoria (JSONL), dataset anonimizado e artefatos de fine-tuning.
"""

import os
import json
import streamlit as st
import pandas as pd

from fase3.database import MedicalDatabase
from fase3.dataset_prep import build_synthetic_medical_dataset, save_dataset
from fase3.fine_tuning import MedicalFineTuner
from fase3.langgraph_workflow import MedicalLangGraphWorkflow


# Configuração da página Streamlit
st.set_page_config(
    page_title="Assistente Virtual Médico - Hospital",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def get_db():
    db = MedicalDatabase()
    db.seed_sample_data()
    return db


@st.cache_resource
def get_workflow():
    return MedicalLangGraphWorkflow()


def main():
    db = get_db()
    workflow = get_workflow()

    st.title("🏥 Assistente Médico Virtual")
    st.caption("FIAP - IA para Devs | Tech Challenge - Fase 3")

    # --- SIDEBAR: SELEÇÃO DE PACIENTE E CONTROLES ---
    st.sidebar.header("📋 Painel do Paciente")
    
    pacientes_list = ["PAC-101", "PAC-202", "PAC-303"]
    selected_paciente_id = st.sidebar.selectbox("Selecione o Paciente:", pacientes_list)

    patient_summary = db.get_patient_summary(selected_paciente_id)
    paciente_data = patient_summary.get("paciente", {})
    prontuario_data = patient_summary.get("prontuario", {})
    exames_data = patient_summary.get("exames", [])

    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 Informações Cadastrais")
    st.sidebar.write(f"**ID:** {paciente_data.get('id', 'N/A')}")
    st.sidebar.write(f"**Nome:** {paciente_data.get('nome_anonimizado', 'N/A')}")
    st.sidebar.write(f"**Idade/Sexo:** {paciente_data.get('idade')} anos | {paciente_data.get('sexo')}")
    st.sidebar.write(f"**Leito:** {paciente_data.get('leito')}")
    st.sidebar.write(f"**Admissão:** {paciente_data.get('diagnostico_admissional')}")

    pending_exams = [e for e in exames_data if e["status"] == "PENDENTE"]
    if pending_exams:
        st.sidebar.error(f"⚠️ {len(pending_exams)} Exame(s) Pendente(s)!")
        for pe in pending_exams:
            st.sidebar.caption(f"• {pe['nome_exame']}")
    else:
        st.sidebar.success("✅ Nenhum exame pendente.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Ações do Sistema")
    if st.sidebar.button("🔄 Reinicializar Banco & Datasets"):
        db.seed_sample_data()
        ds = build_synthetic_medical_dataset()
        save_dataset(ds)
        ft = MedicalFineTuner()
        ft.prepare_openai_jsonl()
        ft.generate_ollama_modelfile()
        st.sidebar.success("Sistema reinicializado!")

    # --- TABS PRINCIPAIS DA APLICAÇÃO ---
    tab_atendimento, tab_prontuario, tab_auditoria, tab_finetuning = st.tabs([
        "🩺 Atendimento & LangGraph",
        "📊 Prontuário Eletrônico",
        "🔒 Auditoria & Logs",
        "🧠 Fine-Tuning & Datasets"
    ])

    # =========================================================================
    # TAB 1: ATENDIMENTO & LANGGRAPH
    # =========================================================================
    with tab_atendimento:
        st.subheader("💬 Consulta Clínica ao Assistente Virtual")
        st.info("Digite uma dúvida clínica ou selecione um dos atalhos rápidos abaixo.")

        if "user_query_text" not in st.session_state:
            st.session_state["user_query_text"] = ""

        # Atalho rápido específico para o caso do paciente selecionado
        if selected_paciente_id == "PAC-101":
            if st.button("🫀 Sugestão de Consulta: Protocolo para Dor no Peito e ECG Alterado"):
                st.session_state["user_query_text"] = "Qual o protocolo de conduta para paciente com dor no peito?"
        elif selected_paciente_id == "PAC-202":
            if st.button("🧪 Sugestão de Consulta: Protocolo para Infecção Grave / Sepse"):
                st.session_state["user_query_text"] = "Quais são as condutas imediatas para suspeita de infecção grave ou sepse?"
        else:
            if st.button("🫁 Sugestão de Consulta: Protocolo para Crise de Asma e Falta de Ar"):
                st.session_state["user_query_text"] = "Qual a conduta inicial para crise de asma ou falta de ar aguda no PS?"

        user_query = st.text_area("Sua consulta técnica / dúvida clínica:", key="user_query_text", height=100)

        if st.button("🚀 Processar com LangGraph", type="primary"):
            if not user_query.strip():
                st.warning("Por favor, digite uma dúvida clínica.")
            else:
                with st.spinner("Executando fluxo LangGraph (Prontuário ➔ Exames ➔ LLM ➔ Guardrails ➔ Log)..."):
                    result = workflow.run(selected_paciente_id, user_query)

                st.markdown("---")
                st.subheader("🎯 Resultado do Atendimento Inteligente")

                # Exibição de Alertas de Exames Pendentes
                alertas = result.get("alertas", [])
                if alertas:
                    for al in alertas:
                        st.error(al)

                # Resposta Final Formatada
                st.markdown(result["resposta_final"])

                # Métricas e Expander do Fluxo Interno
                st.markdown("---")
                with st.expander("🔍 Detalhes da Execução do Fluxo LangGraph"):
                    st.json({
                        "paciente_id": result["paciente_id"],
                        "query_medica": result["query_medica"],
                        "exames_pendentes_encontrados": [e["nome_exame"] for e in result["pending_exams"]],
                        "fonte_protocolo_utilizado": result["llm_suggestion"].get("fonte_protocolo"),
                        "status_guardrails": result["safety_validation"]
                    })

    # =========================================================================
    # TAB 2: PRONTUÁRIO ELETRÔNICO (SQL)
    # =========================================================================
    with tab_prontuario:
        st.subheader(f"📊 Prontuário Integrado - {selected_paciente_id}")

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("### 📝 Evolução Clínica Atual")
            st.info(f"🗣️ **Queixa Principal:**\n\n{prontuario_data.get('queixa_principal', 'N/A')}")
            st.info(f"🫀 **Sinais Vitais:**\n\n{prontuario_data.get('sinais_vitais', 'N/A')}")
            st.info(f"💊 **Medicamentos em Uso:**\n\n{prontuario_data.get('historico_medicamentoso', 'N/A')}")
            st.warning(f"🚫 **Alergias Relatadas:**\n\n{prontuario_data.get('alergias', 'Nenhuma')}")

        with col_p2:
            st.markdown("### 🔬 Painel de Exames Solicitados")
            if exames_data:
                df_exames = pd.DataFrame(exames_data)[["nome_exame", "status", "resultado", "data_solicitacao"]]
                st.dataframe(df_exames, use_container_width=True)
            else:
                st.info("Nenhum exame registrado para este paciente.")

    # =========================================================================
    # TAB 3: AUDITORIA E LOGS (JSONL)
    # =========================================================================
    with tab_auditoria:
        st.subheader("🔒 Rastreabilidade e Audit Logs (JSONL)")
        st.caption("Logs estruturados para auditoria hospitalar e conformidade de IA médica.")

        audit_path = "outputs/audit_log.jsonl"
        if os.path.exists(audit_path):
            with open(audit_path, "r", encoding="utf-8") as f:
                logs_lines = [json.loads(line) for line in f.readlines() if line.strip()]

            if logs_lines:
                df_logs = pd.DataFrame(logs_lines)[["timestamp", "session_id", "paciente_id", "query_medica", "protocolo_consultado"]]
                st.dataframe(df_logs, use_container_width=True)

                st.markdown("### 🔎 Inspeção Detalhada do Último Registro")
                st.json(logs_lines[-1])
            else:
                st.info("Nenhum registro no log de auditoria ainda.")
        else:
            st.info("Arquivo de log ainda não foi criado. Execute uma consulta na Tab 1.")

    # =========================================================================
    # TAB 4: FINE-TUNING & DATASETS
    # =========================================================================
    with tab_finetuning:
        st.subheader("🧠 Gestão de Treinamento e Customização da LLM")

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("### 📄 Modelfile (Ollama / LLaMA3 Local)")
            modelfile_path = "fine_tuning_artifacts/Modelfile.medical"
            if os.path.exists(modelfile_path):
                with open(modelfile_path, "r", encoding="utf-8") as f:
                    st.code(f.read(), language="dockerfile")
            else:
                st.warning("Modelfile não encontrado. Clique na barra lateral para gerar.")

        with col_f2:
            st.markdown("### 📑 Dataset Formato OpenAI Fine-Tuning (JSONL)")
            jsonl_path = "fine_tuning_artifacts/medical_ft_openai.jsonl"
            if os.path.exists(jsonl_path):
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f.readlines()[:3]]
                st.json(lines)
            else:
                st.warning("Arquivo JSONL não encontrado. Clique na barra lateral para gerar.")


if __name__ == "__main__":
    main()
