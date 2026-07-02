import sys
import logging
from common_local.metrics_context import MetricsContext

# Structured metrics logger
metric_logger = logging.getLogger("agent_metrics")
metric_logger.setLevel(logging.INFO)
if not metric_logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    metric_logger.addHandler(_h)

ctx = MetricsContext(agent_id="mlorchestrator")
