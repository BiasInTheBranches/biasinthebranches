"""API for metrics."""

from __future__ import annotations

import json
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from harness.api.logging import logger


class Metric(ABC):
    """
    An abstract base class for evaluation metrics.

    Each metric stores multiple evaluation scores for both training and testing
    sets, allowing aggregation (e.g., computing mean test score).

    Subclasses should implement the `evaluate()` method and `is_higher_better` property.

    Attributes
    ----------
    _train_scores : list of float
        The list of training scores recorded for this metric.
    _test_scores : list of float
        The list of testing scores recorded for this metric.

    """

    def __init__(self):
        """Initialize a new Metric instance with empty score lists."""
        self._train_scores: list[float] = []
        self._test_scores: list[float] = []

    @property
    @abstractmethod
    def is_higher_better(self) -> bool:
        """
        Whether higher values of this metric indicate better performance.

        Returns
        -------
        bool
            True if higher metric values are better, False otherwise.
        """
        pass

    @abstractmethod
    def evaluate(self, predicted: Any, labels: Any, train: bool = False) -> None:
        """
        Evaluate the metric given predictions and labels.

        Parameters
        ----------
        predicted : Any
            The model predictions for the given samples.
        labels : Any
            The ground truth labels corresponding to the predictions.
        train : bool, optional (default=False)
            If True, the scores are recorded as training scores.
            If False, the scores are recorded as testing scores.

        Returns
        -------
        None
        """
        pass

    @property
    def train_score(self) -> float | None:
        """
        The average training score across all evaluations.

        Returns
        -------
        float or None
            The mean of all recorded training scores, or None if there are no scores.
        """
        return np.mean(self._train_scores) if self._train_scores else None

    @property
    def test_score(self) -> float | None:
        """
        The average testing score across all evaluations.

        Returns
        -------
        float or None
            The mean of all recorded testing scores, or None there are no scores.
        """
        return np.mean(self._test_scores) if self._test_scores else None

    @property
    def result(self) -> float | None:
        """
        Alias for test_score, representing the final metric result on the test set.

        Returns
        -------
        float or None
            The mean test score, or None if no test scores are available.
        """
        return self.test_score

    def add_score(self, value: float, train: bool = False) -> None:
        """
        Add a single score to the training or testing list.

        Parameters
        ----------
        value : float
            The score to record.
        train : bool, optional (default=False)
            If True, add to the training scores; otherwise, add to the testing scores.

        Returns
        -------
        None
        """
        if train:
            self._train_scores.append(float(value))
        else:
            self._test_scores.append(float(value))

    def report(self) -> str:
        """
        Generate a formatted string summarizing the metric results.

        Returns
        -------
        str
            A formatted string containing the metric name, whether higher is better,
            and mean train/test scores.
        """
        if self.is_higher_better:
            not_present_score = float('-inf')
        else:
            not_present_score = float('inf')

        return (
            f'Metric: {self.__class__.__name__}\t'
            f'Train: {(self.train_score if self.train_score is not None else not_present_score):0.3f}\t'  # noqa: E501
            f'Test: {(self.test_score if self.test_score is not None else not_present_score):0.3f}\t'  # noqa: E501
            f'(higher is better: {self.is_higher_better})'
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the metric state to a dictionary.

        Returns
        -------
        dict
            A dictionary containing class name and stored training/testing scores.
        """
        return {
            'class_name': self.__class__.__name__,
            'train_scores': self._train_scores if self._train_scores else None,
            'test_scores': self._test_scores if self._test_scores else None,
            'is_higher_better': self.is_higher_better,
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """
        Reinitialize the metric's internal state from a dictionary.

        Parameters
        ----------
        data : dict
            A dictionary containing 'train_scores' and 'test_scores'.

        Returns
        -------
        None
        """
        if 'train_scores' in data and data['train_scores'] is None:
            logger.warning(
                f'Loading metric {self.__class__.__name__} `train_scores` is None'
            )
        if 'test_scores' in data and data['test_scores'] is None:
            logger.warning(
                f'Loading metric {self.__class__.__name__} `test_scores` is None'
            )

        self._train_scores = data['train_scores'] if data['train_scores'] else []
        self._test_scores = data['test_scores'] if data['test_scores'] else []


class MetricCollection:
    """
    A collection of metrics that can be evaluated, saved, and reported together.

    This class is a container for multiple metric instances, and behaves like a list.
    """

    def __init__(self, metrics: list[Metric] | None = None):
        """
        Initialize a new MetricCollection with an optional list of metrics.

        Parameters
        ----------
        metrics : list of Metric, optional
            A list of metric instances. If None, an empty list is used.
        """
        self.metrics = metrics if metrics is not None else []

    def __getitem__(self, index: int) -> Metric:
        """Returns the metric at the given index."""
        return self.metrics[index]

    def __len__(self) -> int:
        """Provides length of the collection."""
        return len(self.metrics)

    def __iter__(self):
        """Returns an iterator over the metrics in the collection."""
        return iter(self.metrics)

    def add_metric(self, metric: Metric) -> None:
        """
        Add a metric to the collection.

        Parameters
        ----------
        metric : Metric
            The metric instance to add.
        """
        self.metrics.append(metric)

    def evaluate(self, predicted, labels, train: bool = False) -> None:
        """
        Evaluate all metrics on the given predictions and labels.

        Parameters
        ----------
        predicted : Any
            The model predictions.
        labels : Any
            The ground truth labels.
        train : bool, optional (default=False)
            If True, the evaluation results are recorded as training scores.
            Otherwise, they are recorded as testing scores.
        """
        for metric in self.metrics:
            metric.evaluate(predicted, labels, train=train)

    def save(self, path: Path) -> None:
        """
        Save all metrics to a single JSON file.

        Parameters
        ----------
        path : Path
            The file path to save the metrics to.
        """
        output_data = [metric.to_dict() for metric in self.metrics]
        with open(path, 'w') as f:
            json.dump(output_data, f, indent=4)

    def load(self, path: Path, metric_classes: dict[str, Metric]) -> None:
        """
        Load metrics from a JSON file.

        Parameters
        ----------
        path : Path
            The file path to load the metrics from.
        metric_classes : dict of str to Type[Metric]
            A dictionary mapping class names to Metric classes.
            Used to reconstruct metric instances from the saved data.

        Raises
        ------
        ValueError
            If a metric class in the file cannot be found in `metric_classes`.
        """
        with open(path) as f:
            input_data = json.load(f)

        cls_registry = {cls.__name__: cls for cls in metric_classes.values()}

        self.metrics = []
        for data in input_data:
            class_name = data['class_name']
            metric_cls = cls_registry.get(class_name)
            if metric_cls is None:
                raise ValueError(f"Metric class '{class_name}' not found.")
            metric = metric_cls()
            metric.from_dict(data)
            self.metrics.append(metric)

    def report(self) -> str:
        """
        Return a formatted report of all metrics in the collection.

        Returns
        -------
        str
            A string with each metric's report on a new line.
        """
        return '\n'.join(metric.report() for metric in self.metrics)

    @property
    def results(self) -> list[float]:
        """
        Return a list of test_scores for convenience.

        Returns
        -------
        list of float
            A list of each metric's test_score. If a metric has no test scores,
            its value will be None.
        """
        return [m.result for m in self.metrics]
