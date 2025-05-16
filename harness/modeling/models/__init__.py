"""Submodule for LM model instances."""

from __future__ import annotations

from typing import Any
from typing import Union

from harness.distribution import parsl_registry

from .ankh import ankh_models
from .baseline import baseline_models
from .calm import calm_models
from .dnabert import dnabert_models
from .esm import esm_models

model_registry = {
    **baseline_models,
    **ankh_models,
    **calm_models,
    **dnabert_models,
    **esm_models,
}

ModelConfigTypes = Union[*model_registry.keys(),]
ModelTypes = Union[*model_registry.values(),]


# This is a workaround to support optional registration.
# Make a function to combine the config and instance initialization
# since the registry only accepts functions with hashable arguments.
def _factory_fn(**kwargs: dict[str, Any]) -> ModelTypes:
    model_config = kwargs.get('model_config', None)
    if not model_config:
        raise ValueError(
            f'Unknown model config: {kwargs}. Available: {ModelConfigTypes}',
        )

    model_cls = model_registry.get(model_config.__class__)

    return model_cls(model_config)


def get_model(
    model_config: ModelConfigTypes,
    register: bool = False,
) -> ModelTypes:
    """Get instance of a model based on the configuration present.

    Parameters
    ----------
    model_config : ModelConfigTypes
        The model configuration instance.
    register : bool, optional
        Register the instance for warmstart. Caches the
        instance based on the kwargs, by default False.

    Returns
    -------
    ModelTypes
        The instance.

    Raises
    ------
    ValueError
        If the `config` is unknown.
    """
    # Create and register the instance
    kwargs = {'model_config': model_config}
    if register:
        parsl_registry.register(_factory_fn)
        return parsl_registry.get(_factory_fn, **kwargs)

    return _factory_fn(**kwargs)
