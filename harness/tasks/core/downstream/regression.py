"""General regression models and utilities for downstream tasks."""

from __future__ import annotations

import numpy as np
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from harness import SEED
from harness import SKLEARN_RANDOM_STATE
from harness.api.logging import logger
from harness.api.metric import Metric
from harness.api.task import DownstreamModel

# -----------------------------------------------------------------------------
# Generic Training/Eval Function (same structure as classification)
# -----------------------------------------------------------------------------


def _train_and_evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[Metric],
) -> tuple[DownstreamModel, list[Metric]]:
    """Generic function for fitting an sklearn model, predicting, evaluating metrics."""
    # Drop NaNs
    train_mask = ~np.isnan(X_train).any(axis=1)
    test_mask = ~np.isnan(X_test).any(axis=1)
    if not train_mask.all() or not test_mask.all():
        logger.warning('NaN values present in the input features. Dropping them.')
    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    for metric in metrics:
        metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
        metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

    return model, metrics


# -----------------------------------------------------------------------------
# Single Regression Pipeline
# -----------------------------------------------------------------------------


def _sklearn_regression_pipeline(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int,
    build_model_fn,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """SKlearn regression pipeline.

    A single pipeline that:
      - Takes a dataset (could be single or dict with train/test),
      - Optionally performs k-fold splitting or a single train/test split,
      - Calls the build_model_fn(...) to get an sklearn model,
      - Trains/evaluates the model (or multiple in folds),
      - Returns models + metrics.
    """
    logger.info('Starting regression pipeline...')

    # set to numpy
    if isinstance(task_dataset, Dataset):
        dset_format = task_dataset.format
        task_dataset.set_format('numpy')
    else:
        formats = {}
        for key in task_dataset:
            formats[key] = task_dataset[key].format
            task_dataset[key].set_format('numpy')

    downstream_models = {}

    if k_fold > 0:
        logger.info(f'K-Fold CV with {k_fold} folds')
        X = task_dataset[input_col]
        y = task_dataset[target_col]

        kf = KFold(n_splits=k_fold, shuffle=True, random_state=SKLEARN_RANDOM_STATE)
        for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model = build_model_fn()
            model, metrics = _train_and_evaluate_model(
                model, X_train, y_train, X_test, y_test, metrics
            )
            downstream_models[f'fold_{fold_idx}'] = model
            logger.info(f'\tFold {fold_idx} completed')
    else:
        # If we are able to, split the data into train and test sets
        # If there is no `train_test_split` method, we will assume the dataset
        # is already split into train and test sets

        logger.info('Single train/test split')
        if hasattr(task_dataset, 'train_test_split') and callable(
            task_dataset.train_test_split
        ):
            split_dataset = task_dataset.train_test_split(test_size=0.2, seed=SEED)
        else:
            split_dataset = task_dataset
            assert (  # noqa: PT018
                'train' in split_dataset and 'test' in split_dataset
            ), (
                'The downstream dataset does not have a train_test_split method and '
                'does not contain a train and test split'
            )

        X_train = split_dataset['train'][input_col]
        y_train = split_dataset['train'][target_col]
        X_test = split_dataset['test'][input_col]
        y_test = split_dataset['test'][target_col]

        model = build_model_fn()
        model, metrics = _train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, metrics
        )
        downstream_models['default'] = model

    # return format
    if isinstance(task_dataset, Dataset):
        task_dataset.set_format(**dset_format)
    else:
        for key in task_dataset:
            task_dataset[key].set_format(**formats[key])

    return downstream_models, metrics


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def sklearn_svr(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Train a Support Vector Regressor (SVR) using embeddings and target values."""
    logger.info('Evaluating with Support Vector Regressor')

    def build_model_fn():
        return SVR()

    return _sklearn_regression_pipeline(
        task_dataset=task_dataset,
        input_col=input_col,
        target_col=target_col,
        metrics=metrics,
        k_fold=k_fold,
        build_model_fn=build_model_fn,
    )


def sklearn_mlp_regressor(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Train a Multi Layer Perceptron Regressor using embeddings and target values."""
    logger.info('Evaluating with Multi Layer Perceptron Regressor')

    # peek at task dataset to determine the dimension of input features
    if isinstance(task_dataset, Dataset):
        hidden_size = len(task_dataset[input_col][0])
    elif isinstance(task_dataset, DatasetDict) and 'train' in task_dataset:
        hidden_size = len(task_dataset['train'][input_col][0])
    else:
        hidden_size = 256
        logger.warning(
            f'Unable to determine input size from {task_dataset},'
            f' defaulting to {hidden_size}'
        )

    def build_model_fn():
        return MLPRegressor(
            hidden_layer_sizes=(512, 128),
            activation='relu',
            solver='adam',
            early_stopping=True,
            n_iter_no_change=5,
            random_state=SKLEARN_RANDOM_STATE,
        )

    return _sklearn_regression_pipeline(
        task_dataset=task_dataset,
        input_col=input_col,
        target_col=target_col,
        metrics=metrics,
        k_fold=k_fold,
        build_model_fn=build_model_fn,
    )
