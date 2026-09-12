"""
fase3/langgraph_workflow.py

Orquestrador do Fluxo Clinico com LangGraph.
Coordena o ciclo completo de atendimento:
Carregar Prontuário (SQL) -> Verificar Exames -> Gerar Resposta LLM -> Guardrails -> Auditoria & Alertas.
"""

from typing import Dict, List, Any, TypedDict
from langgraph.graph import StateGraph, END

from fase3.database import MedicalDatabase
from fase3.assistente_medico import MedicalAssistantChain
from fase3.guardrails import SafetyGuardrails
from fase3.audit_logger import AuditLogger


# Definindo o Estado Clinico do LangGraph
class ClinicalState(TypedDict):
    paciente_id: str
    query_medica: str
    patient_info: Dict[str, Any]
    pending_exams: List[Dict[str, Any]]
    llm_suggestion: Dict[str, Any]
    safety_validation: Dict[str, Any]
    alertas: List[str]
    resposta_final: str


class MedicalLangGraphWorkflow:
    def __init__(self):
        self.db = MedicalDatabase()
        self.assistant = MedicalAssistantChain()
        self.guardrails = SafetyGuardrails()
        self.logger = AuditLogger()
        self.graph = self._build_graph()

    # --- Nó 1: Carregar Prontuário no SQLite ---
    def node_carregar_prontuario(self, state: ClinicalState) -> ClinicalState:
        paciente_id = state["paciente_id"]
        info = self.db.get_patient_summary(paciente_id)
        state["patient_info"] = info
        return state

    # --- Nó 2: Verificar Exames Pendentes ---
    def node_verificar_exames_pendentes(self, state: ClinicalState) -> ClinicalState:
        paciente_id = state["paciente_id"]
        pendentes = self.db.get_pending_exams(paciente_id)
        state["pending_exams"] = pendentes

        alertas = list(state.get("alertas", []))
        if pendentes:
            exames_lista = ", ".join([e["nome_exame"] for e in pendentes])
            alertas.append(f"⚠️ [ALERTA DE SEGURANÇA]: Paciente possui exames críticos PENDENTES: {exames_lista}")
        
        state["alertas"] = alertas
        return state

    # --- Nó 3: Consultar LLM Médica Customizada (LangChain) ---
    def node_gerar_conduta_llm(self, state: ClinicalState) -> ClinicalState:
        res = self.assistant.generate_response(
            query=state["query_medica"],
            patient_info=state["patient_info"],
            pending_exams=state["pending_exams"]
        )
        state["llm_suggestion"] = res
        return state

    # --- Nó 4: Aplicar Guardrails de Segurança & Explainability ---
    def node_aplicar_guardrails(self, state: ClinicalState) -> ClinicalState:
        suggestion = state["llm_suggestion"]
        raw_text = suggestion.get("resposta_llm", "")
        fonte = suggestion.get("fonte_protocolo", "")

        val_res = self.guardrails.apply_safety_filters(raw_text, fuentes_consultadas=fonte)
        state["safety_validation"] = val_res
        state["resposta_final"] = val_res["resposta_sanitizada"]
        return state

    # --- Nó 5: Gravar Log de Auditoria ---
    def node_registrar_auditoria(self, state: ClinicalState) -> ClinicalState:
        self.logger.log_interaction({
            "session_id": f"SESS-{state['paciente_id']}",
            "paciente_id": state["paciente_id"],
            "query_medica": state["query_medica"],
            "exames_pendentes": [e["nome_exame"] for e in state["pending_exams"]],
            "protocolo_consultado": state["llm_suggestion"].get("fonte_protocolo", "N/A"),
            "resposta_gerada": state["resposta_final"],
            "validacao_seguranca": state["safety_validation"],
            "alertas": state["alertas"]
        })
        return state

    # --- Construção do Grafo ---
    def _build_graph(self):
        workflow = StateGraph(ClinicalState)

        # Adicionar Nós
        workflow.add_node("carregar_prontuario", self.node_carregar_prontuario)
        workflow.add_node("verificar_exames_pendentes", self.node_verificar_exames_pendentes)
        workflow.add_node("gerar_conduta_llm", self.node_gerar_conduta_llm)
        workflow.add_node("aplicar_guardrails", self.node_aplicar_guardrails)
        workflow.add_node("registrar_auditoria", self.node_registrar_auditoria)

        # Definir Arestas de Fluxo
        workflow.set_entry_point("carregar_prontuario")
        workflow.add_edge("carregar_prontuario", "verificar_exames_pendentes")
        workflow.add_edge("verificar_exames_pendentes", "gerar_conduta_llm")
        workflow.add_edge("gerar_conduta_llm", "aplicar_guardrails")
        workflow.add_edge("aplicar_guardrails", "registrar_auditoria")
        workflow.add_edge("registrar_auditoria", END)

        return workflow.compile()

    def run(self, paciente_id: str, query_medica: str) -> Dict[str, Any]:
        """Executa o fluxo completo do LangGraph para um paciente e consulta."""
        initial_state: ClinicalState = {
            "paciente_id": paciente_id,
            "query_medica": query_medica,
            "patient_info": {},
            "pending_exams": [],
            "llm_suggestion": {},
            "safety_validation": {},
            "alertas": [],
            "resposta_final": ""
        }

        final_state = self.graph.invoke(initial_state)
        return final_state


if __name__ == "__main__":
    app = MedicalLangGraphWorkflow()
    resultado = app.run("PAC-101", "Qual a conduta para dor torácica aguda?")
    print("\n=== RESPOSTA DO LANGGRAPH ===")
    print(resultado["resposta_final"])
