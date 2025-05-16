"""Utility functions for working with data in the modeling module."""

from __future__ import annotations

import datasets

from harness.api.modeling import SequenceModelOutput


def sequences_to_dataset(sequences: list[str]) -> datasets.Dataset:
    """Convert a list of sequences to a dataset."""
    return datasets.Dataset.from_dict({'sequences': sequences})


def outputs_to_dataset(
    model_outputs: list[SequenceModelOutput],
    partial_dataset: datasets.Dataset | None = None,
) -> datasets.Dataset:
    """Cache model outputs in a dataset. Will concatenate with an existing dataset."""
    output_dataset = datasets.Dataset.from_list(model_outputs)

    # Concatenate the partial dataset with the output dataset if it exists
    if partial_dataset is not None:
        return datasets.concatenate_datasets([partial_dataset, output_dataset])

    # Else return the output dataset
    return output_dataset
