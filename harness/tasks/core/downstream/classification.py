"""General classification utilities and models."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from datasets import Dataset
from datasets import DatasetDict
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.utils import resample

from harness import SEED
from harness import SKLEARN_RANDOM_STATE
from harness.api.logging import logger
from harness.api.metric import Metric
from harness.api.task import DownstreamModel


def balance_classes(task_dataset: Dataset, target_col: str) -> Dataset:
    """Balance classes by undersampling the majority classes.

    Parameters
    ----------
    task_dataset : datasets.Dataset
        The dataset containing the input features and target labels
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    datasets.Dataset
        The balanced dataset
    """
    # Extract the input features and target labels
    y = task_dataset[target_col]
    if isinstance(y, torch.Tensor):
        y = y.tolist()

    # Identify unique classes and their counts
    unique_classes, counts = np.unique(y, return_counts=True)
    # class_counts = dict(zip(unique_classes, counts))
    min_class_size = counts.min()

    # Undersample each class to the size of the smallest class
    sample_indices = []
    for class_value in unique_classes:
        class_indices = [i for i, label in enumerate(y) if label == class_value]
        class_sampled_indices = resample(
            class_indices,
            replace=False,
            n_samples=min_class_size,
            random_state=SKLEARN_RANDOM_STATE,
        )
        sample_indices.extend(class_sampled_indices)

    return task_dataset.select(sample_indices)


def object_to_label(semantic_labels: list[Any]) -> np.ndarray:
    """Convert a list of objects into class labels appropriate for SVC.

    NOTE: this function sorts the labels and uses the sorted order to encode them.
    This is important for reproducibility and consistency across runs. But it
    does require that labels are hashable.

    Parameters
    ----------
    objects_list : list
        List of objects to be converted into class labels

    Returns
    -------
    list
        List of class labels
    """
    le = LabelEncoder()
    le_labels = sorted(list(set(semantic_labels)))  # noqa: C414
    le = le.fit(le_labels)
    return le.transform(semantic_labels)


def object_to_multi_label(semantic_labels: list[Any]) -> np.ndarray:
    """Convert a list of objects into labels appropriate for multi label prediction.

    Parameters
    ----------
    objects_list : list
        List of objects to be converted into class labels

    Returns
    -------
    list
        List of class labels
    """
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(semantic_labels)


# -----------------------------------------------------------------------------
# Generic Training/Eval Function
# -----------------------------------------------------------------------------


def _train_and_evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: list[Metric],
) -> tuple[DownstreamModel, list[Metric]]:
    """Train and evaluate the provided sklearn model.

    Generic function to:
      1) Remove NaNs
      2) Train the provided sklearn model
      3) Predict on train and test sets
      4) Evaluate with the given metrics
    """
    # Remove NaN values and issue warning
    train_mask = ~np.isnan(X_train).any(axis=1)
    test_mask = ~np.isnan(X_test).any(axis=1)
    if not train_mask.all() or not test_mask.all():
        logger.warning('NaN values present in the input features. Dropping them.')

    X_train, y_train = X_train[train_mask], y_train[train_mask]
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate
    for metric in metrics:
        metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
        metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

    return model, metrics


# -----------------------------------------------------------------------------
# Single Classification Pipeline
# -----------------------------------------------------------------------------


def _sklearn_classification_pipeline(  # noqa: PLR0912, PLR0915
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int,
    build_model_fn,  # function that returns an unfit sklearn model
    multi_label=False,  # if True, skip label transforms & do special K-Fold strategy
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """SKlearn classification pipeline.

    A single pipeline that:
      - Takes a dataset (could be single or dict with train/test),
      - Optionally performs k-fold splitting or a single train/test split,
      - Calls the build_model_fn(...) to get an sklearn model,
      - Trains/evaluates the model (or multiple in folds),
      - Returns models + metrics.
    """
    logger.info('Starting classification pipeline...')

    # Temporarily set dataset to numpy
    if isinstance(task_dataset, Dataset):
        dset_format = task_dataset.format
        task_dataset.set_format('numpy')
    else:  # DatasetDict
        formats = {}
        for key in task_dataset:
            formats[key] = task_dataset[key].format
            task_dataset[key].set_format('numpy')

    downstream_models = {}

    if k_fold > 0:
        logger.info(f'K-Fold CV with {k_fold} folds')

        X = task_dataset[input_col]
        raw_y = task_dataset[target_col]

        skf = StratifiedKFold(
            n_splits=k_fold, shuffle=True, random_state=SKLEARN_RANDOM_STATE
        )
        if multi_label:
            # Trick: Convert each row of 0/1 labels to a string for StratifiedKFold
            # e.g. [1,0,1] -> "101", [0,0,1] -> "001"
            # This helps the stratified k fold group them by "class pattern".
            # We do NOT do any label transforms here because it is already binarized.
            y_bin_str = [''.join(arr.astype(str)) for arr in raw_y]

            splits = skf.split(X, y_bin_str)
            # We'll keep `raw_y` as our final Y (the 0/1 matrix).
            y = raw_y
        else:
            # Single-label. Convert the raw labels to integer-encoded classes.
            y = object_to_label(raw_y)
            splits = skf.split(X, y)
        # Now that we have splits, run the downstream model on each fold
        for fold_idx, (train_index, test_index) in enumerate(splits):
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

        # Either we call dataset.train_test_split() or assume dataset is already a Dict
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
        X_test = split_dataset['test'][input_col]

        raw_train_y = split_dataset['train'][target_col]
        raw_test_y = split_dataset['test'][target_col]

        if multi_label:
            # Already binarized multi-label data. Use as is.
            y_train, y_test = raw_train_y, raw_test_y
        else:
            # Single-label. Convert to integer-encoded classes.
            y_train = object_to_label(raw_train_y)
            y_test = object_to_label(raw_test_y)

        model = build_model_fn()
        model, metrics = _train_and_evaluate_model(
            model, X_train, y_train, X_test, y_test, metrics
        )
        downstream_models['default'] = model

    # Restore original dataset format
    if isinstance(task_dataset, Dataset):
        task_dataset.set_format(**dset_format)
    else:
        for key in task_dataset:
            task_dataset[key].set_format(**formats[key])

    return downstream_models, metrics


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def sklearn_svc(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Trains a Support Vector Classifier with single-label outputs."""
    logger.info('Evaluating with Support Vector Classifier')

    def build_model_fn():
        return SVC()

    return _sklearn_classification_pipeline(
        task_dataset=task_dataset,
        input_col=input_col,
        target_col=target_col,
        metrics=metrics,
        k_fold=k_fold,
        build_model_fn=build_model_fn,
        multi_label=False,
    )


def sklearn_mlp_classifier(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Trains an MLP classifier for single-label classification tasks."""
    logger.info('Evaluating with MultiLayer Perceptron (Single-label)')

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
        # Configure hyperparams here:
        return MLPClassifier(
            hidden_layer_sizes=(512, 128),
            activation='relu',
            solver='adam',
            early_stopping=True,
            n_iter_no_change=5,
            random_state=SKLEARN_RANDOM_STATE,
        )

    return _sklearn_classification_pipeline(
        task_dataset=task_dataset,
        input_col=input_col,
        target_col=target_col,
        metrics=metrics,
        k_fold=k_fold,
        build_model_fn=build_model_fn,
        multi_label=False,
    )


def sklearn_multilabel_mlp_classifier(
    task_dataset: Dataset | DatasetDict,
    input_col: str,
    target_col: str,
    metrics: list[Metric],
    k_fold: int = 0,
) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
    """Trains an MLP classifier for multi-label classification tasks."""
    logger.info('Evaluating with MultiLabel MultiLayer Perceptron')

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
        # Configure hyperparams here:
        return MLPClassifier(
            hidden_layer_sizes=(
                hidden_size // 2,
                hidden_size // 4,
            ),
            activation='relu',
            solver='adam',
            early_stopping=True,
            n_iter_no_change=5,
            random_state=SKLEARN_RANDOM_STATE,
        )

    return _sklearn_classification_pipeline(
        task_dataset=task_dataset,
        input_col=input_col,
        target_col=target_col,
        metrics=metrics,
        k_fold=k_fold,
        build_model_fn=build_model_fn,
        multi_label=True,
    )
