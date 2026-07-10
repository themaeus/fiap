# Arquitetura e Estratégia de Escalabilidade (resumo)

Este documento resume opções e decisões para habilitar escalabilidade automática e monitoramento para a solução de otimização de modelos.

## Objetivo
Permitir que a solução lide com variações de demanda, monitorar desempenho e facilitar operação (observabilidade).

## Camadas principais
- Containerização: empacotar serviço(s) em imagens Docker para consistência e facilitar orquestração.
- Orquestração: Kubernetes (k8s) com Horizontal Pod Autoscaler (HPA) para autoescalonamento por CPU/latência/metricas customizadas; alternativa: ECS/Fargate em AWS.
- Escalonamento de infraestrutura: Autoscaling Groups (VMs) ou provisionamento serverless (Cloud Run, AWS Lambda) dependendo da carga e requisitos de latência.
- Monitoramento: Prometheus para métricas, Grafana para visualização; Alertmanager para alertas.
- Logging: centralizado via ELK (Elasticsearch, Logstash, Kibana) ou soluções gerenciadas (Cloud Logging, Datadog).

## Implementação mínima entregue
- Logging local e gravação de métricas em `outputs/metrics/metrics.jsonl` via o módulo `monitoramento.py`.
- Instrumentação básica no script de experimentos rápidos (`run_experiments_quick.py`) para medir tempos e gravar métricas de teste.

Racional: a implementação mínima fornece rastreabilidade local e é suficiente para validar comportamento antes de integrar uma stack de observabilidade completa.

## Passos recomendados para produção (próximos passos)
1. Containerizar a aplicação:
   - `Dockerfile` para criar imagem com dependências.
2. Deploy em Kubernetes:
   - Implantar serviços em Deployment/Service.
   - Habilitar HPA com métricas de CPU e métricas customizadas via Prometheus Adapter.
3. Observabilidade:
   - Exportar métricas Prometheus (`prometheus_client`) em endpoint `/metrics`.
   - Enviar logs em JSON para stdout e coletá-los com um agente (Fluentd / Filebeat).
   - Configurar dashboards Grafana e regras de alerta (ex.: queda de recall, latência alta).
4. Pipelines e reprodutibilidade:
   - Armazenar artefatos (modelos, SHAP) em armazenamento persistente (S3/GCS) e versioná-los.
   - Executar experimentos em CI/CD para reproducibilidade e auditoria.

## Decisões de projeto (por que)
- Usar Kubernetes/HPA: maior controle sobre políticas de escalonamento e facilidade para métricas customizadas.
- Prometheus + Grafana: padrão de fato para métricas, leve e integrável com aplicações Python.
- Logging centralizado: essencial para investigação de incidentes; logs locais não são suficientes em escala.

## Considerações de segurança
- Isolar credenciais em `Secrets` (k8s) e não armazenar chaves em repositório.
- Controlar acesso a dashboards e dados sensíveis (PHI) com políticas RBAC e criptografia.

## Observação final
A entrega atual implementa a instrumentação mínima sem alterar a arquitetura de deploy — é suficiente ao escopo do pedido. Se quiser, prossigo com containerização e exportação Prometheus (requer adicionar dependência `prometheus_client` e criar `Dockerfile`).
