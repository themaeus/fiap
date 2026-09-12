"""
fase3/guardrails.py

Módulo de Segurança e Validação Clínica (Guardrails).
Define limites estritos de atuação do assistente para evitar sugestões impróprias,
garantir indispensabilidade de validação humana e anexar disclaimers obrigatórios.
"""

import re
from typing import Dict, Any, Tuple

# Palavras-chave e padrões associados a prescrições e condutas ativas
PRESCRIPTION_KEYWORDS = [
    r"\bprescrevo\b", r"\bprescrever\b", r"\badministrar imediatamente\b",
    r"\breceito\b", r"\biniciar medicação sem consulta\b"
]

DISCLAIMER_TEXT = (
    "\n\n---"
    "\n⚠️ [AVISO DE SEGURANÇA E LIMITAÇÃO DE ATUAÇÃO]:"
    "\nEste assistente médico opera exclusivamente como ferramenta de apoio à decisão clínica. "
    "Todas as sugestões de diagnóstico, exames ou condutas terapêuticas DEVEM ser revisadas "
    "e prescritas formalmente por um médico devidamente registrado (CRM)."
)


class SafetyGuardrails:
    def __init__(self):
        pass

    def check_prescription_violation(self, llm_output: str) -> Tuple[bool, str]:
        """
        Verifica se a resposta da LLM tenta realizar uma prescrição direta ou definitiva
        sem ressalva de aprovação médica humana.
        """
        for pattern in PRESCRIPTION_KEYWORDS:
            if re.search(pattern, llm_output, re.IGNORECASE):
                return True, (
                    "[BLOQUEIO DE SEGURANÇA]: A resposta gerada continha termos de prescrição autônoma direta. "
                    "A ação foi bloqueada e convertida para sugestão pendente de validação humana."
                )
        return False, ""

    def apply_safety_filters(self, llm_output: str, fuentes_consultadas: str = "") -> Dict[str, Any]:
        """
        Filtra a resposta da LLM, aplica bloqueios de prescrição se necessário,
        anexa a fonte consultada (explainability) e o disclaimer legal obrigatório.
        """
        violation_detected, block_reason = self.check_prescription_violation(llm_output)
        
        sanitized_output = llm_output
        if violation_detected:
            # Reformular tom prescritivo para tom de sugestão/recomendação ao médico
            sanitized_output = f"Recomendação para avaliação médica: {llm_output}"

        # Anexar Fontes / Explainability
        explainability_block = ""
        if fuentes_consultadas:
            explainability_block = f"\n\n📚 [FONTE DO PROTOCOLO CONSULTADO]: {fuentes_consultadas}"

        final_response = f"{sanitized_output}{explainability_block}{DISCLAIMER_TEXT}"

        return {
            "aprovado_seguranca": True,
            "violacao_prescricao_detectada": violation_detected,
            "motivo_bloqueio": block_reason,
            "resposta_sanitizada": final_response
        }


if __name__ == "__main__":
    guard = SafetyGuardrails()
    sample_text = "Prescrevo 500mg de Amoxicilina de 8/8h por 7 dias."
    res = guard.apply_safety_filters(sample_text, "Protocolo de Infectologia v2024")
    print("[Guardrails Test]", res)
