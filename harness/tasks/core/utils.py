"""Downstream tasks utilities."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence

import datasets
import numpy as np
from sklearn.utils import resample

from harness import SKLEARN_RANDOM_STATE
from harness.api.logging import logger
from harness.api.modeling import HDF5CachedList
from harness.api.modeling import Transform
from harness.modeling.transforms import transform_registry


def find_transformation(
    model_input: str, model_resolution: str, task_resolution: str
) -> Sequence[Transform]:
    """Map task input, model resolution, and task resolution to a transformation.

    Parameters
    ----------
    task_input : str
        task input type, must be 'dna', or 'aminoacid'
    model_resolution : str
        resolution of the tokens level representations from the model
    task_resolution : str
        required hidden representation granularity

    Returns
    -------
    Sequence[Transform]
        Sequence of transformations to be applied iteratively

    Raises
    ------
    ValueError
        If the resolution mapping is not found, or transform is not found in registry
    """
    # Map model resolution to task resolution through a transform
    task_transform_mapping = {
        # For each task resolution, map model encoding to transform name
        'dna': {
            'sequence': {
                'bpe': ('average_pool',),
                'char': ('average_pool',),
                '3mer': ('average_pool',),
                '6mer': ('average_pool',),
            },
            'nucleotide': {
                'bpe': ('super_resolution',),
                'char': ('full_sequence',),
                '3mer': ('super_resolution',),
                '6mer': ('super_resolution',),
            },
            'aminoacid': {
                'bpe': ('super_resolution', '3_window'),
                'char': ('3_window',),
                '3mer': ('full_sequence',),
                '6mer': ('super_resolution', '3_window'),
            },
        },
        'aminoacid': {
            'sequence': {
                'bpe': ('average_pool',),
                'char': ('average_pool',),
                '3mer': ('average_pool',),
                '6mer': ('average_pool',),
            },
            'nucleotide': {},
            'aminoacid': {
                'bpe': ('super_resolution',),
                'char': ('full_sequence',),
                '3mer': ('super_resolution',),
                '6mer': ('super_resolution',),
            },
        },
    }

    # Retrieve list of transforms from the registry
    transform_names = (
        task_transform_mapping.get(model_input, {})
        .get(task_resolution, {})
        .get(model_resolution, None)
    )

    # Check that we haven't missed a mapping
    if transform_names is None:
        raise ValueError(
            f'Resolution mapping not found for {model_input=}, {task_resolution=},'
            f' and {model_resolution=}'
        )

    for t_name in transform_names:
        if t_name not in transform_registry:
            raise ValueError(f'Transform {t_name} not found in registry')

    return [transform_registry.get(name) for name in transform_names]


def limit_training_samples(
    task_dataset: datasets.Dataset,
    max_samples: int,
    input_col: str,  # potentially deprecate
    target_col: str,
    continuous=False,
) -> datasets.Dataset:
    """Limit the total number of training examples respecting the class balance.

    Parameters
    ----------
    task_dataset : datasets.Dataset
        The dataset containing the input features and target labels
    max_samples : int
        Maximum number of samples to use for training
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels
    continuous : bool
        Whether the target labels are continuous, if so will bin and balance


    Returns
    -------
    datasets.Dataset
        The dataset with a limited number of training examples
    """
    # Short circuit if the dataset is already smaller than the maximum number of samples
    if max_samples >= len(task_dataset):
        return task_dataset

    if target_col is None:
        logger.debug('No target column provided, sampling randomly')
        random_indices = np.random.choice(len(task_dataset), max_samples, replace=False)
        return task_dataset.select(random_indices)

    # Extract the input features and target labels
    y = task_dataset[target_col]

    # If there are continuoys labels, bin them to balance with classes
    if continuous:
        y_bins = np.digitize(y, np.histogram_bin_edges(y, bins='auto'))
    else:
        y_bins = y

    # Calculate the proportion of each class
    unique_classes, class_counts = np.unique(y_bins, return_counts=True)
    total_samples = sum(class_counts)
    class_proportions = {
        cls: count / total_samples
        for cls, count in zip(unique_classes, class_counts, strict=False)
    }

    # Determine the number of samples for each class
    class_sample_counts = {
        cls: int(round(proportion * max_samples))
        for cls, proportion in class_proportions.items()
    }

    # Ensure the total number of samples is exactly max_samples
    while sum(class_sample_counts.values()) != max_samples:
        diff = max_samples - sum(class_sample_counts.values())
        for label_class in unique_classes:
            if diff == 0:
                break
            if diff > 0:
                class_sample_counts[label_class] += 1
                diff -= 1
            elif class_sample_counts[label_class] > 0:
                class_sample_counts[label_class] -= 1
                diff += 1

    # Sample the dataset to limit the total number of training examples
    # and respecting class balance
    sampled_indices = []
    for class_value in unique_classes:
        class_indices = [i for i, label in enumerate(y_bins) if label == class_value]

        if class_sample_counts[class_value] == 0:
            continue

        class_sampled_indices = resample(
            class_indices,
            replace=False,
            n_samples=class_sample_counts[class_value],
            random_state=SKLEARN_RANDOM_STATE,
        )
        sampled_indices.extend(class_sampled_indices)

    return task_dataset.select(sampled_indices)


def mask_nan(data: np.ndarray) -> np.ndarray:
    """Return mask of same shape as input array, with True where NaN values are present.

    Parameters
    ----------
    mat : np.ndarray
        The input numpy array

    Returns
    -------
    np.ndarray
        The numpy array with NaN values masked
    """
    return ~np.isnan(data).any(axis=1)


def _generate_token_rows_without_embeddings(
    task_dataset: datasets.Dataset,
    row_lengths: list[int],
    token_level_fields: list[str],
    sequence_level_fields: list[str],
):
    """Generate token-level rows from a dataset without embeddings.

    Parameters
    ----------
    task_dataset : datasets.Dataset
        The original dataset with sequence and other task related information. Oriented
        at single sequence per row.
    row_lengths : list[int]
        The number of tokens in each row, this is a function of the model max length
    token_level_fields : list[str]
        The fields that have token level information, usually the sequence and the
        labels
    sequence_level_fields : list[str]
        The fields that have sequence level information, usually the metadata about the
        sequence

    Yields
    ------
    Iterable[dict[str, Any]]
        An iterable of dictionaries where each dictionary corresponds to a single token
        iteratively extracted from the dataset
    """
    for i in range(len(task_dataset)):
        # Extract sequence-level fields
        seq_values = {field: task_dataset[field][i] for field in sequence_level_fields}

        # Determine how many tokens for this row based on a token-level field
        # sometimes lengths get changed as a function of max model length, so
        # we need an explicit list of lengths
        num_tokens = min(
            row_lengths[i],
            *[len(task_dataset[field][i]) for field in token_level_fields],
        )
        if row_lengths[i] != num_tokens:
            logger.debug(
                'Truncated number of tokens to match sequence length of a token level'
                f' {num_tokens=}, {row_lengths[i]=}'
            )

        # Extract token-level fields
        seq_token_values = {}
        for field in token_level_fields:
            values = task_dataset[field][i][:num_tokens]
            seq_token_values[field] = values

        # Yield one row per token
        for token_idx in range(num_tokens):
            row = {'idx': token_idx}
            # Add token-level fields
            for field in token_level_fields:
                row[field] = seq_token_values[field][token_idx]

            # Add sequence-level fields
            for field in sequence_level_fields:
                row[field] = seq_values[field]

            yield row


def _flatten_dataset_fields(
    task_dataset: datasets.Dataset,
    row_lengths: list[int],
) -> datasets.Dataset:
    """Flatten the fields of task_dataset to token-level, ignoring embeddings.

    Will flatten minimally to the length of the embeddings provided, but will be shorter
    if there are any fields shorter than the embedding length. This happens when the end
    of sequences are truncated (typically because stop codons are not labeled).
    """
    # Identify token-level fields by comparing length to 'label' column
    if 'label' not in task_dataset.column_names:
        raise ValueError(
            "The dataset must contain a 'label' column to identify token-level fields."
        )

    first_element_length = len(task_dataset['label'][0])
    token_level_fields = []
    for field in task_dataset.column_names:
        val = task_dataset[field][0]
        if isinstance(val, Iterable) and len(val) == first_element_length:
            token_level_fields.append(field)

    logger.info(f'Token level fields: {token_level_fields}')

    # Determine sequence-level fields
    all_columns = task_dataset.column_names
    sequence_level_fields = [c for c in all_columns if c not in token_level_fields]

    # Create a dataset from the generator of rows
    flattened_dataset = datasets.Dataset.from_generator(
        _generate_token_rows_without_embeddings,
        gen_kwargs={
            'task_dataset': task_dataset,
            'row_lengths': row_lengths,
            'token_level_fields': token_level_fields,
            'sequence_level_fields': sequence_level_fields,
        },
    )
    return flattened_dataset


def _flatten_embeddings(
    model_outputs: HDF5CachedList,
    truncate_end: bool = False,
) -> np.ndarray:
    """Flatten embeddings from model_outputs into a single numpy array."""
    end_pos = -1 if truncate_end else None
    flat_embeddings = []
    for output in model_outputs:
        seq_embeddings = output.embedding[:end_pos]
        flat_embeddings.extend(seq_embeddings)
    return np.array(flat_embeddings)


def flatten_to_token_level(
    task_dataset: datasets.Dataset,
    model_outputs: HDF5CachedList,
    truncate_end: bool = False,
) -> datasets.Dataset:
    """Flatten the task_dataset and embeddings into a token-level dataset.

    Steps:
      1. Flattens the dataset fields without embeddings to a token-level dataset.
      2. Flattens the embeddings. (needs separate step because can't pickle h5 objects)
      3. Combines the flattened embeddings with the token-level dataset.

    Parameters
    ----------
    task_dataset : datasets.Dataset
        The original dataset with sequence-level and token-level fields.
    model_outputs : H5CachedList
        A H5 backed list where each element corresponds to an example in task_dataset.
        Each element has an attribute `embedding` that is a numpy array
        of shape (num_tokens, embedding_dim).
    truncate_end : bool, optional
        If True, truncate the last token from each sequence.

    Returns
    -------
    datasets.Dataset
        A new dataset where each row corresponds to a single token.
    """
    # Step 1: Flatten dataset fields without embeddings
    # Determine the number of tokens in each row, this is a function of the
    # model max length and equivalent to min(len(sequence), max_length)
    row_lengths = [mo.embedding.shape[0] for mo in model_outputs]
    flattened_dataset = _flatten_dataset_fields(
        task_dataset=task_dataset, row_lengths=row_lengths
    )

    # Step 2: Flatten embeddings separately
    embeddings_dset = datasets.Dataset.from_dict(
        {
            'transformed': _flatten_embeddings(
                model_outputs=model_outputs,
                truncate_end=truncate_end,
            )
        }
    )

    # Step 3: Add embeddings as a new column to the flattened dataset
    flattened_dataset = datasets.concatenate_datasets(
        [flattened_dataset, embeddings_dset],
        axis=1,
    )

    return flattened_dataset
