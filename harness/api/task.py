"""API for tasks."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import TypeVar

from sklearn.base import BaseEstimator

from harness.api.config import BaseConfig
from harness.api.metric import Metric
from harness.api.modeling import LM

# TypeVar for downstream models
DownstreamModel = TypeVar('DownstreamModel', bound=BaseEstimator)


class TaskConfig(BaseConfig):
    """General configuration for a task."""

    # The name of the dataset or the path to the dataset (for HF dataset loading)
    dataset_name_or_path: str


class Task(ABC):
    """A general task interface."""

    def __init__(self, config: TaskConfig):
        """Initialize the task."""
        self.config = config

    @abstractmethod
    def evaluate(
        self, model: LM, cache_dir: Path
    ) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
        """Evaluate the task and return its metrics."""
        ...
