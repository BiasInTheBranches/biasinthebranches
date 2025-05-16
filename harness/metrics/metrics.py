"""Implementation of metrics for downstream model performance."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

from harness.api.metric import Metric


class MSE(Metric):
    """
    Mean Squared Error (MSE) metric.

    Lower values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return False

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the MSE between predicted and labels and record the score.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth labels.
        train : bool, optional (default=False)
            If True, record as a training score; otherwise as a testing score.
        """
        result = mean_squared_error(labels, predicted)
        self.add_score(result, train=train)


class RMSE(Metric):
    """
    Root Mean Squared Error (RMSE) metric.

    Lower values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return False

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the RMSE between predicted and labels and record the score.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth labels.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        result = root_mean_squared_error(labels, predicted)
        self.add_score(result, train=train)


class R2(Metric):
    """
    R-squared (R2) regression score.

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the R2 score between predicted and labels and record the score.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth labels.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        result = r2_score(labels, predicted)
        self.add_score(result, train=train)


class Accuracy(Metric):
    """
    Accuracy classification score.

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the accuracy between predicted and labels and record the score.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted classes.
        labels : np.ndarray
            True classes.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        result = accuracy_score(labels, predicted)
        self.add_score(result, train=train)


class F1(Metric):
    """
    F1 score (weighted-averaged).

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute and record the F1 score (micro average) between predicted and labels.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted classes.
        labels : np.ndarray
            True classes.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        result = f1_score(labels, predicted, average='weighted')
        self.add_score(result, train=train)


class PearsonCorrelation(Metric):
    """
    Pearson correlation coefficient.

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the Pearson correlation coefficient between predicted and labels.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth values.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        pearson_r, _ = pearsonr(predicted, labels)
        self.add_score(pearson_r, train=train)


class SpearmanCorrelation(Metric):
    """
    Spearman rank correlation coefficient.

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the Spearman correlation coefficient between predicted and labels.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth values.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        spearman_r, _ = spearmanr(predicted, labels)
        self.add_score(spearman_r, train=train)


class MatthewsCorrelationCoefficient(Metric):
    """Matthews correlation coefficient.

    Higher values indicate better performance.
    """

    @property
    def is_higher_better(self) -> bool:
        """Reports whether higher values of this metric indicate better performance."""
        return True

    def evaluate(
        self, predicted: np.ndarray, labels: np.ndarray, train: bool = False
    ) -> None:
        """
        Compute the Matthews correlation coefficient between predicted and labels.

        Parameters
        ----------
        predicted : np.ndarray
            Predicted values.
        labels : np.ndarray
            Ground truth values.
        train : bool, optional
            If True, record as a training score; otherwise as a testing score.
        """
        result = matthews_corrcoef(labels, predicted)
        self.add_score(result, train=train)


metric_registry = {
    'mse': MSE,
    'rmse': RMSE,
    'r2': R2,
    'accuracy': Accuracy,
    'f1': F1,
    'pearson': PearsonCorrelation,
    'spearman': SpearmanCorrelation,
    'mcc': MatthewsCorrelationCoefficient,
}
