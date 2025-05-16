"""Metric module for evaluating downstream model outputs."""

from __future__ import annotations

from harness.api.metric import Metric

from .metrics import metric_registry


def get_and_instantiate_metric(metric_name: str) -> Metric | None:
    """Get and instantiate a metric from a configuration dictionary."""
    metric_cls = metric_registry.get(metric_name)
    if metric_cls is None:
        raise ValueError(f'Metric {metric_name} not found in registry')
    return metric_cls()
