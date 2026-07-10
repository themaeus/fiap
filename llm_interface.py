"""
Interface mínima para integração com LLMs e avaliação simples das interpretações.

Comportamento:
- `criar_prompt_interpretacao(...)` gera prompt em pt-BR com instruções médicas claras.
- `chamar_llm(...)` usa OpenAI se `OPENAI_API_KEY` estiver definida; caso contrário
  retorna um sumário fallback usando `nlp_interpret.resumir_resultados()`.
- `avaliar_interpretacao(...)` calcula uma métrica heurística de qualidade baseada
  na sobreposição com `top_features` e presença de termos relevantes.

Uso (exemplo):
    from llm_interface import criar_prompt_interpretacao, chamar_llm, avaliar_interpretacao
    prompt = criar_prompt_interpretacao(resultados, top_features)
    resposta = chamar_llm(prompt)
    score = avaliar_interpretacao(resposta, top_features)

Observação: este módulo fornece integração básica e fallback sem exigir chaves.
"""
from __future__ import annotations

import os
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

try:
    import openai
except Exception:
    openai = None

from nlp_interpret import resumir_resultados

logger = logging.getLogger(__name__)


def criar_prompt_interpretacao(resultados: Dict[str, Any], top_features: Optional[Iterable[str]] = None, paciente_info: Optional[Dict[str, Any]] = None) -> str:
    """Gera um prompt em Português (pt-BR) pedindo explicações médicas acionáveis.

    - `resultados`: dicionário com métricas e informações do modelo (ex.: saída de `run_experiments_quick`).
    - `top_features`: lista de nomes de features mais importantes (strings).
    - `paciente_info`: informações do paciente (opcional) para contextualizar recomendações.

    O prompt pede linguagem clara, prioridades clínicas e limitações/certezas.
    """
    partes = []
    partes.append("Você é um assistente clínico que resume resultados de modelos de diagnóstico.")
    partes.append("Responda em Português do Brasil de forma concisa, direcionada a médicos, destacando:")
    partes.append("1) Interpretação dos resultados do modelo (o que indicam para o paciente).")
    partes.append("2) Fatores de risco / features mais relevantes e como interpretá-las clinicamente.")
    partes.append("3) Recomendações acionáveis e sugestões de próximos passos (exames, monitoramento, cautela).")
    partes.append("4) Principais fontes de incerteza e limitações do modelo.")
    partes.append("Use linguagem apropriada para profissionais de saúde e inclua observações sobre sensibilidade/especificidade quando possível.")

    # Incluir resumo dos resultados compactado
    partes.append("\nResumo dos resultados fornecidos:\n")
    # compact summary
    summary_lines = []
    if 'baseline' in resultados:
        summary_lines.append('- Baseline:')
        for k, v in resultados['baseline'].items():
            summary_lines.append(f"  - {k}: acc={v.get('accuracy'):.4f}, recall={v.get('recall'):.4f}, f1={v.get('f1_score'):.4f}")
    if 'experimentos' in resultados:
        summary_lines.append(f"- Experimentos: {len(resultados['experimentos'])} realizadas. Forneça o destaque do melhor resultado no conjunto de validação e no teste.")
    partes.append('\n'.join(summary_lines))

    if top_features:
        partes.append('\nFeatures mais relevantes (fornecidas): ' + ', '.join(list(top_features)[:10]))
    if paciente_info:
        partes.append('\nContexto do paciente:')
        for k, val in (paciente_info or {}).items():
            partes.append(f"- {k}: {val}")

    partes.append('\nInstruções de formato:')
    partes.append('1) Inicie com um resumo em 2-3 frases. 2) Liste até 5 pontos acionáveis numerados. 3) Termine com uma linha sobre incertezas.')

    prompt = '\n'.join(partes)
    return prompt


def chamar_llm(prompt: str, provider: str = 'auto', model: Optional[str] = None, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Chama uma LLM para gerar interpretação.

    - Se `OPENAI_API_KEY` estiver definida e `openai` disponível, usa OpenAI ChatCompletion.
    - Caso contrário, retorna um fallback usando `resumir_resultados()`.

    Observação: não força instalação de pacotes externos — integra quando disponíveis.
    """
    # preferência por OpenAI quando chave presente
    api_key = os.environ.get('OPENAI_API_KEY')
    if provider in ('auto', 'openai') and api_key and openai is not None:
        try:
            # suporta a nova interface openai>=1.0 (client OpenAI)
            model_name = model or os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
            logger.info(f'Chamando OpenAI model={model_name} max_tokens={max_tokens}')
            if hasattr(openai, 'OpenAI'):
                client = openai.OpenAI()
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                # nova API: resp.choices[0].message.content
                text = resp.choices[0].message.get('content') if resp.choices and resp.choices[0].message else None
                if text is None:
                    # tentar caminho alternativo
                    text = getattr(resp.choices[0], 'message', {}).get('content', '')
                text = text.strip() if text else ''
                return text
            else:
                # fallback para versões antigas do pacote
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                return text
        except Exception:
            logger.exception('Falha ao chamar OpenAI; usando fallback')

    # fallback rule-based summary
    logger.info('Usando fallback de sumário local (resumir_resultados)')
    # se existir um sumário gerado pelos experimentos, leia e gere resumos por modelo
    try:
        from pathlib import Path
        resumo_path = Path('outputs_quick/ga_resultados_quick_sumario.json')
        if resumo_path.exists():
            import json as _json
            data = _json.loads(resumo_path.read_text(encoding='utf-8'))
            partes = []
            # tentar extrair top_features do prompt, se presente
            feats = None
            try:
                m = re.search(r'Features mais relevantes.*?:\s*(.+)', prompt, re.I)
                if m:
                    feats = [s.strip() for s in m.group(1).split(',') if s.strip()]
            except Exception:
                feats = None
            # resumir baselines
            baseline = data.get('baseline', {})
            for nome_modelo, metricas in baseline.items():
                partes.append(resumir_resultados(nome_modelo, metricas, top_features=feats))
            return '\n\n'.join(partes)
    except Exception:
        logger.exception('Erro ao gerar fallback a partir de outputs_quick')

    try:
        # último recurso: devolver prompt compactado (evita SyntaxWarning usando raw string)
        return 'Resumo (fallback): ' + re.sub(r'\s+', ' ', prompt)[:1000]
    except Exception:
        return 'Resumo (fallback): (erro ao comprimir prompt)'


def avaliar_interpretacao(resposta: str, top_features: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Avaliação heurística simples da qualidade da interpretação.

    Retorna um dict com métricas locais:
    - `overlap_score`: fração de `top_features` mencionadas no texto (0..1)
    - `comprimento`: número de caracteres
    - `tem_recomendacoes`: bool (procura por verbos de ação)
    """
    text = resposta.lower()
    metrics: Dict[str, Any] = {}
    if top_features:
        features = [f.lower() for f in top_features]
        found = sum(1 for f in features if f in text)
        overlap = found / max(1, len(features))
        metrics['overlap_score'] = round(overlap, 3)
        metrics['features_found'] = found
    else:
        metrics['overlap_score'] = None
        metrics['features_found'] = 0

    metrics['comprimento'] = len(resposta)
    metrics['tem_recomendacoes'] = bool(re.search(r'(?:(recomenda|sugere|indicar|considerar|monitorar|investigar)\b)', text, re.I))
    # heurística simples de 'clareza' baseada em número de frases
    sentences = re.split(r'[\.!?]+', resposta)
    metrics['frases'] = len([s for s in sentences if s.strip()])
    return metrics


if __name__ == '__main__':
    # exemplo rápido
    exemplo_prompt = criar_prompt_interpretacao({'baseline': {}}, top_features=['radius_mean', 'concavity_mean'])
    print('PROMPT EXEMPLO:\n', exemplo_prompt)
    print('\nRESPOSTA (falsa):\n', chamar_llm(exemplo_prompt))
    print('\nAVALIAÇÃO:', avaliar_interpretacao('O modelo indica risco aumentado; recomenda-se biópsia', ['radius_mean']))
