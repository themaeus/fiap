import os
from typing import Dict, List


def resumir_resultados(nome_modelo: str, metricas: Dict[str, float], top_features: List[str] = None) -> str:
    """Gera um resumo textual simples dos resultados do modelo.

    Se houver a variável de ambiente `OPENAI_API_KEY` configurada e o
    pacote `openai` instalado, esta função pode ser estendida para chamar
    um LLM. Por ora, retorna um resumo baseado em regras.
    """
    linhas = []
    linhas.append(f'Resultado do modelo: {nome_modelo}.')
    linhas.append(f"Métricas principais — accuracy: {metricas.get('accuracy', 0):.3f}, recall: {metricas.get('recall', 0):.3f}, f1: {metricas.get('f1_score', 0):.3f}.")

    if metricas.get('recall', 0) >= 0.9:
        linhas.append('O recall é alto — o modelo tende a identificar a maioria dos casos positivos (bom para reduzir falsos negativos).')
        linhas.append('Recomenda-se: priorizar investigação diagnóstica para casos positivos detectados pelo modelo.')
    elif metricas.get('recall', 0) >= 0.75:
        linhas.append('Recall aceitável, mas recomenda-se verificar casos de falso negativo em amostras clínicas.')
        linhas.append('Sugerimos: acompanhar pacientes com resultados limítrofes e considerar exames adicionais quando há suspeita clínica.')
    else:
        linhas.append('Recall baixo — atenção, risco de falsos negativos significativo; não usar sem revisão clínica.')
        linhas.append('Considerar: reavaliação do modelo ou protocolo de confirmação adicional antes de decisões clínicas.')

    if top_features:
        linhas.append('Principais variáveis que contribuíram para as decisões: ' + ', '.join(top_features[:10]) + '.')
        linhas.append('Interpretação clínica rápida: quando estas features estiverem elevadas, aumente o nível de suspeita e priorize exames confirmatórios.')

    linhas.append('Observação: este resumo é automático e não substitui interpretação médica.')
    return '\n'.join(linhas)


def criar_prompt_llm(nome_modelo: str, metricas: Dict[str, float], top_features: List[str] = None) -> str:
    """Cria um prompt pronto para envio a um LLM (ex: GPT) para obter explicações mais ricas.

    O usuário pode colar este prompt em uma ferramenta que possua chave de API.
    """
    prompt = [
        f"Você é um assistente que ajuda profissionais de saúde a interpretar modelos de classificação.",
        f"O modelo é: {nome_modelo}.",
        f"Métricas obtidas: {metricas}.",
    ]
    if top_features:
        prompt.append('As variáveis mais importantes são: ' + ', '.join(top_features[:15]) + '.')
    prompt.append('Explique em linguagem clara, foque em implicações clínicas, limitações do modelo e recomendações práticas curtas.')
    return '\n'.join(prompt)
