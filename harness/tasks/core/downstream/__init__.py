"""Core downstream modeling module."""

from __future__ import annotations

from typing import TypeVar

from sklearn.base import BaseEstimator

from harness.api.task import DownstreamModel

from .classification import sklearn_mlp_classifier
from .classification import sklearn_multilabel_mlp_classifier
from .classification import sklearn_svc
from .regression import sklearn_mlp_regressor
from .regression import sklearn_svr

classification_models = {
    'mlp': sklearn_mlp_classifier,
    'svc': sklearn_svc,
    'default': sklearn_svc,  # If no model type is specified, use the default
}
regression_models = {
    'mlp': sklearn_mlp_regressor,
    'svr': sklearn_svr,
    'default': sklearn_svr,  # If no model type is specified, use the default
}

multi_label_classification_models = {
    'mlp': sklearn_multilabel_mlp_classifier,
    'default': sklearn_multilabel_mlp_classifier,
}

task_map = {
    'classification': classification_models,
    'regression': regression_models,
    'multi-label-classification': multi_label_classification_models,
}


def get_downstream_model(
    task_type: str, model_type: str = 'default'
) -> DownstreamModel:
    """Get the downstream model function for the given task and model type."""
    model_fn = task_map.get(task_type, {}).get(model_type)
    if model_fn is None:
        raise ValueError(
            f'No model found for task type {task_type} and model type {model_type}'
        )
    return model_fn
