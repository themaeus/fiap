import json
from pathlib import Path
from llm_interface import criar_prompt_interpretacao, chamar_llm, avaliar_interpretacao

ROOT = Path('.')
resumo_path = ROOT / 'outputs_quick' / 'ga_resultados_quick_sumario.json'
with open(resumo_path, 'r', encoding='utf-8') as f:
    resultados = json.load(f)

# top features (from relatório) - exemplo fixo
top_features = ['concave points_worst', 'perimeter_worst', 'concave points_mean', 'radius_worst', 'perimeter_mean']

prompt = criar_prompt_interpretacao(resultados, top_features=top_features)
print('--- PROMPT ---\n')
print(prompt)
print('\n--- CHAMANDO LLM (fallback esperado) ---\n')
resposta = chamar_llm(prompt)
print(resposta)

print('\n--- AVALIAÇÃO HEURÍSTICA ---')
metrics = avaliar_interpretacao(resposta, top_features=top_features)
print(metrics)
