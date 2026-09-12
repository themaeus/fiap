"""
fase3/audit_logger.py

Módulo de Logging Detalhado para Rastreamento e Auditoria Clínica.
Registra cada interação, decisão do fluxo, fontes consultadas,
alertas emitidos e status dos guardrails em arquivo JSONL.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any


class AuditLogger:
    def __init__(self, log_path: str = "outputs/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log_interaction(self, event_data: Dict[str, Any]) -> str:
        """
        Salva um registro estruturado de auditoria no arquivo JSONL.
        """
        payload = {
            "timestamp": datetime.now().isoformat(),
            "session_id": event_data.get("session_id", "SESS-UNKNOWN"),
            "paciente_id": event_data.get("paciente_id", "N/A"),
            "usuario_medico": event_data.get("usuario_medico", "Dr. Plantonista"),
            "query_medica": event_data.get("query_medica", ""),
            "exames_pendentes_detectados": event_data.get("exames_pendentes", []),
            "protocolo_consultado": event_data.get("protocolo_consultado", "N/A"),
            "resposta_gerada": event_data.get("resposta_gerada", ""),
            "validacao_seguranca": event_data.get("validacao_seguranca", {}),
            "alertas_emitidos": event_data.get("alertas", [])
        }

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return self.log_path


if __name__ == "__main__":
    logger = AuditLogger()
    logger.log_interaction({
        "session_id": "TESTE-01",
        "paciente_id": "PAC-101",
        "query_medica": "Conduta para dor torácica",
        "exames_pendentes": ["Troponina I Quantitativa"],
        "protocolo_consultado": "Protocolo de Triagem / Dor Torácica",
        "resposta_gerada": "Recomenda-se ECG e Troponina...",
        "alertas": ["Exame crítico pendente: Troponina I Quantitativa"]
    })
    print(f"[AuditLogger] Log de teste gravado com sucesso.")
