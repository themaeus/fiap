"""
Módulo simples de monitoramento e logging em pt-BR.
- Cria diretórios `outputs/logs/` e `outputs/metrics/`.
- Configura logging para arquivo e stdout.
- Oferece helper para registrar métricas em JSONL e contexto de tempo.

Uso mínimo:
    from monitoramento import setup_logging, registrar_metrica, TempoExecucao
    setup_logging()
    with TempoExecucao('experimento_1'):
        ...
    registrar_metrica('rf_test_accuracy', 0.95, tags={'exp':1})

Este módulo é intencionalmente simples para atender ao requisito de monitoramento
sem adicionar dependências externas. Para produção, recomendo integrar
Prometheus/Grafana e pipelines de logging centralizado (ELK/Cloud Logging).
"""
from __future__ import annotations

import json
import logging
from contextlib import ContextDecorator
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

OUTPUT_DIR = Path('outputs')
LOG_DIR = OUTPUT_DIR / 'logs'
METRICS_DIR = OUTPUT_DIR / 'metrics'
_METRICS_FILE = METRICS_DIR / 'metrics.jsonl'


def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / 'app.log'
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_path, encoding='utf-8')
    ]
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=handlers)


def _append_metric(record: Dict[str, Any]):
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with open(_METRICS_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def registrar_metrica(nome: str, valor: Any, tags: Optional[Dict[str, Any]] = None):
    """Registra uma métrica como JSONL em outputs/metrics/metrics.jsonl e loga uma linha."""
    record = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'nome': nome,
        'valor': valor,
        'tags': tags or {}
    }
    _append_metric(record)
    logging.getLogger().info(f"METRICA {nome}={valor} tags={tags}")


class TempoExecucao(ContextDecorator):
    """Context manager para medir tempo de execução (registra em métricas).

    Exemplo:
        with TempoExecucao('exp_1'):
            executar_algo()
    """
    def __init__(self, nome: str, tags: Optional[Dict[str, Any]] = None):
        self.nome = nome
        self.tags = tags or {}
        self.start = None

    def __enter__(self):
        self.start = datetime.utcnow()
        logging.getLogger().info(f"START {self.nome}")
        return self

    def __exit__(self, exc_type, exc, tb):
        end = datetime.utcnow()
        dur = (end - self.start).total_seconds() if self.start else None
        registrar_metrica(f"tempo_{self.nome}", dur, tags=self.tags)
        logging.getLogger().info(f"END {self.nome} duration={dur}s")
        return False
