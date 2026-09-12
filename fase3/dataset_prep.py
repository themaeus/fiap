"""
fase3/dataset_prep.py

Módulo para pré-processamento, anonimização e curadoria de dados médicos.
Remove PII (Informações Pessoais Identificáveis) como CPF, Nomes, Datas, Telefones
e gera/organiza o dataset sintético de protocolos médicos e FAQs clínicas.
"""

import json
import os
import re
from typing import Dict, List, Any

# Expressões regulares para detecção de PII
CPF_REGEX = r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b|\b\d{11}\b"
PHONE_REGEX = r"\b(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9?\d{4}[-\s]?\d{4})\b"
DATE_REGEX = r"\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d\b"
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
NAME_PATTERNS = [
    r"Paciente:\s*([A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?)",
    r"Dr\(a\)\.\s*([A-Z][a-z]+\s+[A-Z][a-z]+)",
    r"Dr\.\s*([A-Z][a-z]+\s+[A-Z][a-z]+)"
]


def anonymize_text(text: str) -> str:
    """
    Substitui dados sensíveis por tokens genéricos de anonimização.
    """
    anonymized = text
    
    # Substituir CPF
    anonymized = re.sub(CPF_REGEX, "[CPF_ANONIMIZADO]", anonymized)
    
    # Substituir Telefone
    anonymized = re.sub(PHONE_REGEX, "[TELEFONE_ANONIMIZADO]", anonymized)
    
    # Substituir Email
    anonymized = re.sub(EMAIL_REGEX, "[EMAIL_ANONIMIZADO]", anonymized)
    
    # Substituir Datas exatas por placeholder genérico
    anonymized = re.sub(DATE_REGEX, "[DATA_ANONIMIZADA]", anonymized)
    
    # Substituir Nomes associados a rótulos conhecidos
    for pattern in NAME_PATTERNS:
        anonymized = re.sub(pattern, lambda m: m.group(0).split(':')[0] + ": [NOME_ANONIMIZADO]" if ':' in m.group(0) else "[NOME_ANONIMIZADO]", anonymized)
        
    return anonymized


def build_synthetic_medical_dataset() -> List[Dict[str, Any]]:
    """
    Gera conjunto de dados sintéticos cobrindo protocolos do hospital, FAQs médicas,
    modelos de laudos, prescrições padronizadas e orientações clínicas.
    """
    raw_data = [
        {
            "id": "PROT-001",
            "categoria": "Protocolo de Triagem / Dor no Peito",
            "fonte": "Protocolo Interno - Cardiologia v2024",
            "pergunta": "Qual o protocolo inicial de triagem para paciente adulto com dor no peito?",
            "resposta": (
                "Paciente [NOME_ANONIMIZADO], CPF [CPF_ANONIMIZADO], atendido em [DATA_ANONIMIZADA].\n"
                "Passos recomendados:\n"
                "1) Fazer ECG em até 10 minutos.\n"
                "2) Coletar exame de sangue (Troponina).\n"
                "3) Dar AAS 200mg mastigável se não houver alergia.\n"
                "4) Colocar no monitor e medir oxigênio.\n"
                "5) Avaliar risco cardíaco e chamar cardiologista se necessário."
            ),
            "palavras_chave": ["dor no peito", "dor toracica", "ecg", "troponina", "aas", "infarto", "pressao", "coracao", "cardiologia"]
        },
        {
            "id": "PROT-002",
            "categoria": "Protocolo de Infecção Grave / Sepse",
            "fonte": "Protocolo de Pronto Socorro - Infecção v2024",
            "pergunta": "Quais são as condutas imediatas para suspeita de infecção grave ou sepse?",
            "resposta": (
                "Paciente [NOME_ANONIMIZADO] atendido por Dr(a). [NOME_ANONIMIZADO] em [DATA_ANONIMIZADA].\n"
                "Passos recomendados:\n"
                "1) Pedir exame de Lactato.\n"
                "2) Coletar sangue para Hemocultura (2 amostras) antes de dar antibiótico.\n"
                "3) Iniciar antibiótico na veia de amplo espectro na 1ª hora.\n"
                "4) Dar soro fisiológico rápido na veia se a pressão estiver baixa.\n"
                "5) Avaliar remédio para subir a pressão (Noradrenalina) se não melhorar com soro."
            ),
            "palavras_chave": ["infecção", "infeccao", "sepse", "lactato", "febre", "pressao baixa", "soro", "hemocultura", "antibiotico"]
        },
        {
            "id": "PROT-003",
            "categoria": "Protocolo de Diabetes / Açúcar Alto",
            "fonte": "Guia Prático de Pronto Socorro - Diabetes",
            "pergunta": "Como ajustar a insulina para açúcar muito alto no sangue?",
            "resposta": (
                "Paciente CPF [CPF_ANONIMIZADO], contato [TELEFONE_ANONIMIZADO].\n"
                "Para glicemia capilar acima de 180 mg/dL:\n"
                "1) Medir o açúcar no sangue (glicemia de ponta de dedo).\n"
                "2) Aplicar insulina rápida (regular) conforme a tabela:\n"
                "   - 181 a 220 mg/dL: 2 unidades;\n"
                "   - 221 a 260 mg/dL: 4 unidades;\n"
                "   - 261 a 300 mg/dL: 6 unidades;\n"
                "   - Acima de 300 mg/dL: 8 unidades e avisar o médico.\n"
                "3) Oferecer água e monitorar os sintomas."
            ),
            "palavras_chave": ["glicemia", "açúcar alto", "acucar alto", "insulina", "diabetes", "diabete", "hiperglicemia"]
        },
        {
            "id": "PROT-004",
            "categoria": "Protocolo de Crise de Asma / Falta de Ar",
            "fonte": "Protocolo de Pronto Socorro - Pneumologia",
            "pergunta": "Qual a conduta inicial para crise de asma ou falta de ar aguda no PS?",
            "resposta": (
                "Atendimento de urgência para falta de ar:\n"
                "1) Iniciar nebulização com bombinha de resgate (Salbutamol/Aerolin) 4 a 8 jatos a cada 20 minutos.\n"
                "2) Dar corticoide via oral (Prednisolona) ou na veia imediatamente.\n"
                "3) Colocar oxigênio de cateter para manter saturação acima de 93%.\n"
                "4) Pedir raio-X de tórax se houver suspeita de pneumonia associada."
            ),
            "palavras_chave": ["falta de ar", "asma", "nebulização", "nebulizacao", "aerolin", "chiado no peito", "tosse", "pneumologia"]
        },
        {
            "id": "FAQ-001",
            "categoria": "FAQ Clínica / Remédios",
            "fonte": "Guia de Segurança do Paciente - Farmácia",
            "pergunta": "Pode tomar remédio de pressão/sangue (Warfarina) junto com anti-inflamatório (Ibuprofeno)?",
            "resposta": (
                "Alerta de segurança:\n"
                "1) Evite tomar Warfarina com Ibuprofeno, Cetoprofeno ou Diclofenaco.\n"
                "2) Essa combinação aumenta o risco de sangramento no estômago.\n"
                "3) Para dor ou febre, dê preferência ao Paracetamol ou Dipirona."
            ),
            "palavras_chave": ["warfarina", "ibuprofeno", "sangramento", "anti-inflamatorio", "antiinflamatorio", "remedio", "medicamento", "interacao"]
        },
        {
            "id": "LAUDO-001",
            "categoria": "Modelo de Laudo / Raio-X de Tórax",
            "fonte": "Serviço de Imagem e Raio-X",
            "pergunta": "Como montar um laudo simples de Raio-X de Tórax para suspeita de pneumonia?",
            "resposta": (
                "Estrutura do laudo:\n"
                "1) EXAME: Raio-X de Tórax.\n"
                "2) ACHADOS: Mancha branca (opacidade) no pulmão direito, sugestiva de secreção/pneumonia.\n"
                "3) CORAÇÃO: Tamanho normal.\n"
                "4) CONCLUSÃO: Sinais compatíveis com pneumonia. Avaliar com os sintomas e exames de sangue do paciente."
            ),
            "palavras_chave": ["raio-x", "raio x", "rx", "radiografia", "pneumonia", "laudo", "torax", "pulmao", "imagem", "exame"]
        }
    ]
    
    # Processar e anonimizar todo o dataset
    anonymized_dataset = []
    for item in raw_data:
        cleaned_item = {
            "id": item["id"],
            "categoria": item["categoria"],
            "fonte": item["fonte"],
            "pergunta": anonymize_text(item["pergunta"]),
            "resposta": anonymize_text(item["resposta"]),
            "palavras_chave": item["palavras_chave"]
        }
        anonymized_dataset.append(cleaned_item)
        
    return anonymized_dataset


def save_dataset(dataset: List[Dict[str, Any]], output_path: str = "data/medical_protocols.json") -> str:
    """
    Salva o dataset processado no caminho especificado.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"[Dataset Prep] Dataset médico anonimizado salvo com sucesso em: {output_path}")
    return output_path


if __name__ == "__main__":
    dataset = build_synthetic_medical_dataset()
    save_dataset(dataset)
