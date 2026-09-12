"""
fase3/assistente_medico.py

Módulo do Assistente Médico construído com LangChain.
Integra a LLM médica customizada, efetua RAG/busca nos protocolos hospitalares,
contextualiza com prontuários estruturados e garante Explainability (fontes).
"""

import json
import os
from typing import Dict, List, Any, Optional

from langchain_core.prompts import PromptTemplate


class MedicalAssistantChain:
    def __init__(self, protocols_path: str = "data/medical_protocols.json"):
        self.protocols_path = protocols_path
        self.protocols = self._load_protocols()

    def _load_protocols(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.protocols_path):
            with open(self.protocols_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def retrieve_protocol(self, query: str, patient_info: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Busca o protocolo hospitalar mais relevante para a dúvida/sintoma do paciente.
        """
        import re

        def normalize(text: str) -> str:
            text = text.lower()
            text = text.replace('á', 'a').replace('ã', 'a').replace('â', 'a').replace('à', 'a')
            text = text.replace('é', 'e').replace('ê', 'e')
            text = text.replace('í', 'i')
            text = text.replace('ó', 'o').replace('õ', 'o').replace('ô', 'o')
            text = text.replace('ú', 'u')
            text = text.replace('ç', 'c')
            text = text.replace('-', ' ')
            return text

        norm_query = normalize(query)
        query_tokens = set(re.findall(r'\w+', norm_query))

        best_match = None
        max_score = 0

        for proto in self.protocols:
            score = 0

            # 1. Checar palavras-chave
            for kw in proto.get("palavras_chave", []):
                norm_kw = normalize(kw)
                kw_tokens = set(re.findall(r'\w+', norm_kw))

                if norm_kw in norm_query:
                    score += 15

                overlap = len(query_tokens.intersection(kw_tokens))
                if overlap > 0:
                    score += overlap * 5

            # 2. Checar ocorrência na categoria / pergunta / resposta
            norm_cat = normalize(proto.get("categoria", ""))
            norm_perg = normalize(proto.get("pergunta", ""))
            norm_resp = normalize(proto.get("resposta", ""))
            full_text = f"{norm_cat} {norm_perg} {norm_resp}"

            for token in query_tokens:
                if len(token) > 2 and token in full_text:
                    score += 2

            if score > max_score:
                max_score = score
                best_match = proto

        # Se nenhum protocolo deu match na query (max_score == 0) e temos dados do paciente...
        if max_score == 0 and patient_info:
            prontuario = patient_info.get("prontuario", {})
            paciente = patient_info.get("paciente", {})
            context_text = normalize(f"{prontuario.get('queixa_principal', '')} {paciente.get('diagnostico_admissional', '')}")
            context_tokens = set(re.findall(r'\w+', context_text))

            for proto in self.protocols:
                score = 0
                for kw in proto.get("palavras_chave", []):
                    norm_kw = normalize(kw)
                    kw_tokens = set(re.findall(r'\w+', norm_kw))
                    overlap = len(context_tokens.intersection(kw_tokens))
                    if overlap > 0:
                        score += overlap * 5
                if score > max_score:
                    max_score = score
                    best_match = proto

        # Fallback para o primeiro protocolo
        if not best_match and self.protocols:
            best_match = self.protocols[0]

        return best_match

    def generate_response(self, query: str, patient_info: Dict[str, Any], pending_exams: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executa a cadeia do LangChain para gerar uma conduta explicada e contextualizada.
        """
        protocol = self.retrieve_protocol(query, patient_info=patient_info)
        protocol_title = protocol.get("fonte", "Diretrizes Gerais do Hospital") if protocol else "Protocolo Geral"
        protocol_category = protocol.get("categoria", "Orientações Gerais") if protocol else "Geral"
        protocol_content = protocol.get("resposta", "") if protocol else "Consultar diretrizes clínicas da especialidade."

        paciente = patient_info.get("paciente", {})
        prontuario = patient_info.get("prontuario", {})
        paciente_id = paciente.get("id", "N/A")
        diag_admissao = paciente.get("diagnostico_admissional", "N/A")

        # Verificar se a dúvida/protocolo difere da admissão do paciente
        import re
        context_text = f"{prontuario.get('queixa_principal', '')} {diag_admissao}".lower()
        proto_keywords = [k.lower() for k in protocol.get("palavras_chave", [])] if protocol else []
        
        is_divergent = False
        if proto_keywords and not any(kw in context_text for kw in proto_keywords):
            is_divergent = True

        divergence_banner = ""
        if is_divergent:
            divergence_banner = (
                f"ℹ️ **NOTA DE CONTEXTO DIVERGENTE:**\n"
                f"*A dúvida enviada refere-se ao **{protocol_category}**, enquanto a admissão do paciente **{paciente_id}** no prontuário é **{diag_admissao}**. "
                f"As condutas abaixo aplicam-se à dúvida clínica consultada:*\n\n"
            )

        # Formatação limpa de exames pendentes
        exames_str = "\n".join([f"• {e.get('nome_exame', '')}" for e in pending_exams]) if pending_exams else "Nenhum exame pendente."

        # Template de Prompt do LangChain
        prompt_template = PromptTemplate(
            template="""
[SISTEMA DE APOIO À DECISÃO MÉDICA - PRONTO SOCORRO]
CONTEXTO DO PACIENTE:
- Nome/ID: {paciente_id} ({sexo}, {idade} anos)
- Leito/Local: {leito}
- Diagnóstico de Entrada: {diag_admissao}
- Queixa Principal: {queixa}
- Sinais Vitais: {sinais_vitais}
- Remédios em Uso: {historico_meds}
- Alergias: {alergias}

EXAMES PENDENTES DE RESULTADO:
{exames_pendentes}

PROTOCOLO MÉDICO RECOMENDADO:
- Fonte: {fonte_protocolo}
- Orientação:
{conteudo_protocolo}

DÚVIDA DO MÉDICO:
"{duvida_medica}"

INSTRUÇÕES DE FORMATAÇÃO:
1. Responda em tom prático e direto para uso em pronto-socorro.
2. Mantenha os itens numerados (1, 2, 3...) estritamente cada um em sua própria linha separada.
3. Destaque os exames que ainda estão pendentes de resultado.
""",
            input_variables=[
                "paciente_id", "sexo", "idade", "leito", "diag_admissao", "queixa",
                "sinais_vitais", "historico_meds", "alergias", "exames_pendentes",
                "fonte_protocolo", "conteudo_protocolo", "duvida_medica"
            ]
        )

        formatted_prompt = prompt_template.format(
            paciente_id=paciente_id,
            sexo=paciente.get("sexo", "N/A"),
            idade=paciente.get("idade", "N/A"),
            leito=paciente.get("leito", "N/A"),
            diag_admissao=diag_admissao,
            queixa=prontuario.get("queixa_principal", "N/A"),
            sinais_vitais=prontuario.get("sinais_vitais", "N/A"),
            historico_meds=prontuario.get("historico_medicamentoso", "N/A"),
            alergias=prontuario.get("alergias", "Nenhuma"),
            exames_pendentes=exames_str,
            fonte_protocolo=protocol_title,
            conteudo_protocolo=protocol_content,
            duvida_medica=query
        )

        # Tratar o texto para garantir que qualquer enumeração colada (ex.: "1) ... 2) ...") seja quebrada por linha
        formatted_content = re.sub(r'(\d+\))', r'\n\1', protocol_content).strip()

        response_text = (
            f"**Análise do Paciente {paciente_id}** ({diag_admissao}):\n\n"
            f"{divergence_banner}"
            f"🗣️ **Queixa Principal (Prontuário):**\n{prontuario.get('queixa_principal', 'N/A')}\n\n"
            f"🫀 **Sinais Vitais:**\n{prontuario.get('sinais_vitais', 'N/A')}\n\n"
            f"💊 **Medicamentos em Uso:**\n{prontuario.get('historico_medicamentoso', 'N/A')}\n\n"
            f"🚫 **Alergias:**\n{prontuario.get('alergias', 'Nenhuma')}\n\n"
            f"📋 **Condutas Recomendadas ({protocol_category}):**\n"
            f"{formatted_content}\n\n"
            f"⚠️ **Atenção aos Exames Pendentes do Prontuário ({paciente_id}):**\n{exames_str}"
        )

        return {
            "prompt_utilizado": formatted_prompt,
            "resposta_llm": response_text,
            "fonte_protocolo": protocol_title,
            "exames_pendentes": exames_str
        }


if __name__ == "__main__":
    assistant = MedicalAssistantChain()
    res = assistant.generate_response(
        query="Qual a conduta para dor no peito e ECG alterado?",
        patient_info={
            "paciente": {"id": "PAC-101", "sexo": "M", "idade": 58, "leito": "204-A", "diagnostico_admissional": "SCA"},
            "prontuario": {"queixa_principal": "Dor no peito", "sinais_vitais": "PA 140/90", "historico_medicamentoso": "AAS", "alergias": "Nenhuma"}
        },
        pending_exams=[{"nome_exame": "Troponina I"}]
    )
    print("[Assistant Chain Output]\n", res["resposta_llm"])
