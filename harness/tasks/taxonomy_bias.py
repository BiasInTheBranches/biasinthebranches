"""Taxonomy Bias task implementations."""

from __future__ import annotations

from typing import Literal

import datasets
import numpy as np
from pydantic import Field
from pydantic import field_serializer
from pydantic import model_validator
from sklearn.model_selection import train_test_split

from harness.api.config import BaseConfig
from harness.api.logging import logger
from harness.api.metric import MetricCollection
from harness.api.modeling import HDF5CachedList
from harness.api.task import Task
from harness.metrics import get_and_instantiate_metric
from harness.tasks.core.downstream import get_downstream_model
from harness.tasks.core.embedding_task import EmbeddingTaskConfig
from harness.tasks.core.utils import find_transformation


def load_dataset_from_path(path: str) -> datasets.Dataset:
    """Load a dataset from a local path or Hugging Face repository.

    Parameters
    ----------
    path : str
        Dataset path, which can be:
        - A local path ending with .hf
        - A HF repo with config in the format "repo_name:config_name"
        - A standard HF dataset path

    Returns
    -------
    datasets.Dataset
        The loaded dataset
    """
    if ':' in path and not path.endswith('.hf'):
        # This is a HF dataset with configuration
        repo_path, config_name = path.split(':', 1)
        return datasets.load_dataset(repo_path, name=config_name)
    elif path.endswith('.hf'):
        # This is a local path to a dataset
        return datasets.load_from_disk(path)
    else:
        # This is a HF dataset without configuration
        return datasets.load_dataset(path)


def parse_dataset_config(config_name: str) -> dict:
    """Parse dataset configuration name to extract metadata.

    Parameters
    ----------
    config_name : str
        Config name in format "domain-taxon_level-split-annotation"
        e.g., "nokingdom-phylum-top0.80-pfam"

    Returns
    -------
    dict or None
        Dictionary with extracted metadata, or None if parsing fails
    """
    parts = config_name.split('-')
    if len(parts) < 4:
        return None

    domain = parts[0]
    taxon_level = parts[1]

    # Extract split info (top/bottom)
    split_part = parts[2]
    if 'top' in split_part:
        split = 'top'
    elif 'bottom' in split_part:
        split = 'bottom'
    else:
        split = split_part

    # Extract annotation type and level
    annotation_part = parts[3]
    if annotation_part.startswith('ec'):
        annotation_type = 'ec'
        level = int(annotation_part[2:]) if len(annotation_part) > 2 else 0
    elif annotation_part.startswith('gene3d'):
        annotation_type = 'gene3d'
        level = int(annotation_part[6:]) if len(annotation_part) > 6 else 0
    elif annotation_part == 'pfam':
        annotation_type = 'pfam'
        level = None
    else:
        return None

    return {
        'domain_filter': domain,
        'taxon_level': taxon_level,
        'split': split,
        'annotation_type': annotation_type,
        'level': level,
    }


def get_opposite_split_path(dataset_path: str) -> str:
    """Generate path to the opposite split dataset.

    Parameters
    ----------
    dataset_path : str
        Original dataset path with either top or bottom split

    Returns
    -------
    str or None
        Path to the opposite split, or None if not applicable
    """
    if ':' not in dataset_path or dataset_path.endswith('.hf'):
        return None

    repo_path, config_name = dataset_path.split(':', 1)

    if 'top' in config_name:
        opposite_config = config_name.replace('top', 'bottom').replace('0.80', '0.20')
    elif 'bottom' in config_name:
        opposite_config = config_name.replace('bottom', 'top').replace('0.20', '0.80')
    else:
        return None

    return f'{repo_path}:{opposite_config}'


def stratified_sample_with_minimum(
    indices: np.ndarray, train_size: int, stratify: np.ndarray, min_samples: int = 5
):
    """Stratified sampling with minimum class representation.

    Select a stratified subsample of indices of total size `train_size` while ensuring
    each class receives at least `min_samples` examples (if available). Two stages:
      1. For each class, select a minimum of min_samples examples (or all if less exist)
      2. From the remaining pool of examples for classes with enough samples, use
         sklearn's train_test_split with stratification to select extra indices,
         so that the final sample (mandatory + extra) is as close as possible to
         the original distribution.

    Parameters
    ----------
    indices : array-like
        Array of indices corresponding to the dataset.
    train_size : int
        Total number of samples to select.
    stratify : array-like
        Array of labels corresponding to each index; must be the same length as indices.
    min_samples : int, default=5
        Minimum number of samples required per class (if available).

    Returns
    -------
    selected_indices : np.ndarray
        Array of indices selected for the subsample.
    remaining_indices : np.ndarray
        Array of indices not selected.

    Raises
    ------
    ValueError
        If the requested train_size is smaller than the total minimum required, or if
        there aren't enough remaining examples to meet train_size.
    """
    indices = np.array(indices)
    stratify = np.array(stratify)

    if len(indices) <= train_size:
        return indices, np.array([])

    # Group indices by label.
    label_to_indices = {}
    for idx, label in zip(indices, stratify, strict=False):
        label_to_indices.setdefault(label, []).append(idx)

    mandatory_indices = []  # will hold the min_samples (or all if not enough)
    extra_pool = []  # extra candidates from classes that have more than min_samples
    extra_labels = []  # corresponding labels for extra_pool (needed for stratification)

    # Process each class separately.
    for label, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        n = len(idxs)
        if n <= min_samples:
            # If a class has fewer than (or equal to) the minimum, include all idxs.
            mandatory_indices.extend(idxs.tolist())
        else:
            # Shuffle indices for reproducibility.
            permuted = np.random.permutation(idxs)
            # Reserve min_samples as mandatory.
            mandatory = permuted[:min_samples]
            extra = permuted[min_samples:]
            mandatory_indices.extend(mandatory.tolist())
            extra_pool.extend(extra.tolist())
            extra_labels.extend([label] * len(extra))

    mandatory_total = len(mandatory_indices)
    if mandatory_total > train_size:
        raise ValueError(
            f'train_size ({train_size}) too small to satisfy the minimum requirement '
            f'for each class (total minimum required = {mandatory_total}).'
        )

    extra_needed = train_size - mandatory_total
    # If no extra samples are needed, we're done.
    if extra_needed == 0:
        final_selection = np.array(mandatory_indices)
    else:
        extra_pool = np.array(extra_pool)
        extra_labels = np.array(extra_labels)
        if extra_needed > len(extra_pool):
            raise ValueError(
                f'Not enough extra samples available. Requested extra {extra_needed} '
                f'but only {len(extra_pool)} available.'
            )
        # Use train_test_split to draw extra_needed samples from the
        # extra pool while stratifying by label.
        extra_selected, _ = train_test_split(
            extra_pool,
            train_size=extra_needed,
            stratify=extra_labels,
        )
        extra_selected = extra_selected.tolist()
        final_selection = np.array(mandatory_indices + extra_selected)

    # Compute remaining indices (those not selected).
    final_set = set(final_selection.tolist())
    remaining_indices = np.array([idx for idx in indices if idx not in final_set])

    return final_selection, remaining_indices


class ClassificationTask(Task):
    """Base class for classification tasks with imbalanced (and many) classes."""

    resolution: str = 'sequence'

    def __init__(self, config: EmbeddingTaskConfig):
        super().__init__(config)

    def evaluate(self, model, cache_dir):
        """Evaluate the model on the task."""
        # Load the dataset
        # NOTE: Originally I set format to numpy, but it prohibits multi-dimension
        # arrays being concatenated into the downstream dataset, removing it does
        # not cause issues.
        task_dataset = load_dataset_from_path(self.config.dataset_name_or_path)
        # Artifact of HF datasets upload
        if 'train' in task_dataset:
            task_dataset = task_dataset['train']
        # Subsample dataset keeping the imbalanced classes same ratio
        if self.config.max_samples is not None:
            indices = np.arange(len(task_dataset))
            selection_indices, _ = stratified_sample_with_minimum(
                indices,
                train_size=self.config.max_samples,
                min_samples=self.config.k_folds,
                stratify=task_dataset[self.config.target_col],
            )
            task_dataset = task_dataset.select(selection_indices)
        # Generate embeddings
        logger.info(
            f'Generating {model.model_input} embeddings ({len(task_dataset):,})'
        )
        input_sequences = task_dataset[model.model_input]

        cache_file = cache_dir / f'{model.config.name}_{self.config.name}.h5'
        with HDF5CachedList(cache_file, mode='w') as model_outputs:
            model_outputs = model.generate_model_outputs(
                input_sequences, model_outputs, return_embeddings=True
            )
            # find and instantiate an output transform object
            transforms = find_transformation(
                model.model_input, model.model_encoding, self.resolution
            )
            logger.info(
                f'Found transformation {[transform.name for transform in transforms]}'
            )
            # Apply the transformations
            for transform in transforms:
                logger.info(f'Applying {transform.name} transformation')
                model_outputs.map(
                    transform.apply_h5,
                    sequences=input_sequences,
                    tokenizer=model.tokenizer,
                )
            # Sequence-level embeddings don't require any transformation of
            # task_dataset, so we can just concatenate the embeddings to the dataset
            embed_dict = {
                'transformed': [output.embedding for output in model_outputs],
            }
            modeling_dataset = datasets.Dataset.from_dict(embed_dict)
            modeling_dataset = datasets.concatenate_datasets(
                [
                    task_dataset,
                    modeling_dataset,
                ],
                axis=1,
            )

            # Setup metrics to pass to downstream prediction model and run modeling
            metrics = MetricCollection(
                [get_and_instantiate_metric(metric) for metric in self.config.metrics]
            )

            # Evaluate with appropriate model
            downstream_modeling = get_downstream_model(
                self.config.task_type, self.config.downstream_model
            )
            downstream_models, metrics = downstream_modeling(
                task_dataset=modeling_dataset,
                input_col='transformed',
                target_col=self.config.target_col,
                metrics=metrics,
                k_fold=self.config.k_folds,
            )

        # Cleanup the cache files if they have been created by this task
        task_dataset.cleanup_cache_files()

        return downstream_models, metrics


class PfamTaxonomyBiasConfig(EmbeddingTaskConfig):
    """Configuration for the Pfam taxonomy bias task."""

    # Name of the task
    name: Literal['PfamTaxonomyBias'] = 'PfamTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])
    # explicitly set the label column since it uses 'pfam' and not 'label'
    target_col: Literal['pfam'] = 'pfam'

    split: Literal['top', 'bottom'] | None = None
    # taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: (
        Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota', 'viruses'] | None
    ) = None

    @model_validator(mode='after')
    def update_from_dataset_path(self):
        """Extract fields from dataset path and update task name."""
        # First try to extract metadata from the dataset path if needed
        if ':' in self.dataset_name_or_path and not self.dataset_name_or_path.endswith(
            '.hf'
        ):
            _, config_name = self.dataset_name_or_path.split(':', 1)
            metadata = parse_dataset_config(config_name)

            if metadata and metadata['annotation_type'] == 'pfam':
                # Only set fields that aren't explicitly provided
                if not hasattr(self, 'domain_filter') or not self.domain_filter:
                    self.domain_filter = metadata['domain_filter']
                if not hasattr(self, 'taxon_level') or not self.taxon_level:
                    self.taxon_level = metadata['taxon_level']
                if not hasattr(self, 'split') or not self.split:
                    self.split = metadata['split']

        # Then update the task name as before
        self.name = f'{self.name}-{self.domain_filter}-{self.split}-{self.taxon_level}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(
            f'-{self.domain_filter}-{self.split}-{self.taxon_level}', ''
        )


class PfamTaxonomyBias(ClassificationTask):
    """Pfam taxonomy bias prediction classification."""

    resolution: str = 'sequence'


class ECTaxonomyBiasConfig(EmbeddingTaskConfig):
    """Configuration for the EC classification taxonomy bias task."""

    # Name of the task
    name: Literal['ECTaxonomyBias'] = 'ECTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])
    # multiple possible target columns, this will implicitly be set by the
    # dataset (as this is the 'ml ready split'), but we might need this here
    target_col: Literal['ec_0', 'ec_1', 'ec_2', 'ec_3'] = 'ec_0'

    split: Literal['top', 'bottom'] | None = None
    # taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: (
        Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota', 'viruses'] | None
    ) = None
    # The EC level we are testing
    ec_level: Literal[0, 1, 2, 3] | None = None

    @model_validator(mode='after')
    def update_from_dataset_path(self):
        """Extract fields from dataset path and update task name."""
        # First try to extract metadata from the dataset path if needed
        if ':' in self.dataset_name_or_path and not self.dataset_name_or_path.endswith(
            '.hf'
        ):
            _, config_name = self.dataset_name_or_path.split(':', 1)
            metadata = parse_dataset_config(config_name)

            if metadata and metadata['annotation_type'] == 'ec':
                # Only set fields that aren't explicitly provided
                if not hasattr(self, 'domain_filter') or not self.domain_filter:
                    self.domain_filter = metadata['domain_filter']
                if not hasattr(self, 'taxon_level') or not self.taxon_level:
                    self.taxon_level = metadata['taxon_level']
                if not hasattr(self, 'split') or not self.split:
                    self.split = metadata['split']
                if not hasattr(self, 'ec_level') or self.ec_level is None:
                    self.ec_level = metadata['level']
                if not hasattr(self, 'target_col') or not self.target_col:
                    self.target_col = f'ec_{metadata["level"]}'

        # Then update the task name as before
        self.name = f'{self.name}-{self.domain_filter}-{self.split}-{self.taxon_level}-ec{self.ec_level}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(
            f'-{self.domain_filter}-{self.split}-{self.taxon_level}-ec{self.ec_level}',
            '',
        )


class ECTaxonomyBias(ClassificationTask):
    """EC taxonomy bias prediction classification."""

    resolution: str = 'sequence'


class Gene3DTaxonomyBiasConfig(EmbeddingTaskConfig):
    """Configuration for the Gene3D classification taxonomy bias task."""

    # Name of the task
    name: Literal['Gene3DTaxonomyBias'] = 'Gene3DTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])
    # multiple possible target columns, this will implicitly be set by the
    # dataset (as this is the 'ml ready split'), but we might need this here
    target_col: Literal['gene3d_0', 'gene3d_1', 'gene3d_2', 'gene3d_3'] = 'gene3d_0'

    split: Literal['top', 'bottom'] | None = None
    # taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: (
        Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota', 'viruses'] | None
    ) = None
    # The EC level we are testing
    gene3d_level: Literal[0, 1, 2, 3] | None = None

    @model_validator(mode='after')
    def update_from_dataset_path(self):
        """Extract fields from dataset path and update task name."""
        # First try to extract metadata from the dataset path if needed
        if ':' in self.dataset_name_or_path and not self.dataset_name_or_path.endswith(
            '.hf'
        ):
            _, config_name = self.dataset_name_or_path.split(':', 1)
            metadata = parse_dataset_config(config_name)

            if metadata and metadata['annotation_type'] == 'gene3d':
                # Only set fields that aren't explicitly provided
                if not hasattr(self, 'domain_filter') or not self.domain_filter:
                    self.domain_filter = metadata['domain_filter']
                if not hasattr(self, 'taxon_level') or not self.taxon_level:
                    self.taxon_level = metadata['taxon_level']
                if not hasattr(self, 'split') or not self.split:
                    self.split = metadata['split']
                if not hasattr(self, 'gene3d_level') or self.gene3d_level is None:
                    self.gene3d_level = metadata['level']
                if not hasattr(self, 'target_col') or not self.target_col:
                    self.target_col = f'gene3d_{metadata["level"]}'

        # Then update the task name as before
        self.name = f'{self.name}-{self.domain_filter}-{self.split}-{self.taxon_level}-gene3d{self.gene3d_level}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(
            f'-{self.domain_filter}-{self.split}-{self.taxon_level}-gene3d{self.gene3d_level}',
            '',
        )


class Gene3DTaxonomyBias(ClassificationTask):
    """Gene3D taxonomy bias prediction classification."""

    resolution: str = 'sequence'


# Classification task where we train models on the top/bottom split
# and evaluate it on the other split.


class CrossEvalClassificationTaskConfig(BaseConfig):
    """Configuration for the cross-eval classification task."""

    # Name of the task - set in the subclass
    name: Literal[''] = ''
    # Task prediction type - also set in the subclass
    task_type: Literal['classification', 'regression', 'multi-label-classification']
    # downstream model - can be set in subclass, defaults to SVM via pipeline
    downstream_model: Literal['mlp', 'svc', 'svr'] | None = None
    # Metrics to measure
    metrics: list[str] = Field(default_factory=lambda: ['accuracy', 'f1'])

    # Dataset paths
    # This should be given a top dataset and a bottom dataset, we are breaking
    # from the API a bit with not having a name_or_path but this will fit better
    top_dataset: str
    bottom_dataset: str

    # Specifics about each individual dataset (num samples, label column etc.)
    # multiple possible target columns, this will implicitly be set by the
    # dataset (as this is the 'ml ready split'), but we might need this here
    target_col: str
    # Whether to balance classes
    balance_classes: bool = False
    # Limit to number of training samples
    max_samples: int | None = None
    # K-fold cross validation
    k_folds: int = 5
    # Truncate ends of sequence embeddings (common for amino acid resolution tasks)
    truncate_end: bool = False

    @model_validator(mode='after')
    def update_from_dataset_paths(self):
        """Extract fields from dataset paths and update task name."""
        # If only top_dataset is specified, try to infer bottom_dataset
        if self.top_dataset and (
            not hasattr(self, 'bottom_dataset') or not self.bottom_dataset
        ):
            opposite_path = get_opposite_split_path(self.top_dataset)
            if opposite_path:
                self.bottom_dataset = opposite_path

        # Extract metadata from top_dataset if needed
        if ':' in self.top_dataset and not self.top_dataset.endswith('.hf'):
            _, config_name = self.top_dataset.split(':', 1)
            metadata = parse_dataset_config(config_name)

            if metadata:
                # Set the common fields if they aren't already set
                if not hasattr(self, 'domain_filter') or not self.domain_filter:
                    self.domain_filter = metadata['domain_filter']
                if not hasattr(self, 'taxon_level') or not self.taxon_level:
                    self.taxon_level = metadata['taxon_level']

                # Set task-specific fields
                if metadata['annotation_type'] == 'ec' and hasattr(self, 'ec_level'):
                    if not self.ec_level:
                        self.ec_level = metadata['level']
                    if not hasattr(self, 'target_col') or not self.target_col:
                        self.target_col = f'ec_{metadata["level"]}'

                elif metadata['annotation_type'] == 'gene3d' and hasattr(
                    self, 'gene3d_level'
                ):
                    if not self.gene3d_level:
                        self.gene3d_level = metadata['level']
                    if not hasattr(self, 'target_col') or not self.target_col:
                        self.target_col = f'gene3d_{metadata["level"]}'

                elif metadata['annotation_type'] == 'pfam' and not hasattr(
                    self, 'target_col'
                ):
                    self.target_col = 'pfam'

        # Then update the task name as before (specific to each cross-eval task)
        # This is already handled in your existing model_validator method
        return self


class CrossEvalClassificationTask(Task):
    """Base for classification tasks trained on one split and evaluated on another.

    This will be a bit different in that we will train two sets of downstream models
    and evaluate them on the other split. We run inference on two sets of sequences,
    train downstream models for both, and evaluate them on the opposing split.

    NOTE: The resultant metrics are implicitly split into 4 groups. The first quarter
    is the training/eval metrics for training/testing on the top dataset, the second
    quarter is the eval metrics for evaluating the top models on the bottom dataset.
    The third quarter is the training/eval metrics for training/testing on the bottom
    dataset, and the final quarter is the eval metrics for evaluating the bottom models
    on the top dataset. This is a bit of a hack, but it allows us to keep the metrics in
    the same order as the original task, and we can easily split them out later.
    """

    resolution: str = 'sequence'

    def __init__(self, config: EmbeddingTaskConfig):
        super().__init__(config)

    def evaluate(self, model, cache_dir):
        """Evaluate the model on the task."""
        # Load the dataset
        # NOTE: Originally I set format to numpy, but it prohibits multi-dimension
        # arrays being concatenated into the downstream dataset, removing it does
        # not seem to cause issues.
        top_dataset = load_dataset_from_path(self.config.top_dataset)
        bottom_dataset = load_dataset_from_path(self.config.bottom_dataset)

        if 'train' in top_dataset:
            top_dataset = top_dataset['train']
        if 'train' in bottom_dataset:
            bottom_dataset = bottom_dataset['train']

        # Subsample dataset keeping the imbalanced classes same ratio
        if self.config.max_samples is not None:
            # Balance the top dataset
            indices = np.arange(len(top_dataset))
            selection_indices, _ = stratified_sample_with_minimum(
                indices,
                train_size=self.config.max_samples,
                min_samples=self.config.k_folds,
                stratify=top_dataset[self.config.target_col],
            )
            top_dataset = top_dataset.select(selection_indices)
            # Balance the bottom dataset
            indices = np.arange(len(bottom_dataset))
            selection_indices, _ = stratified_sample_with_minimum(
                indices,
                train_size=self.config.max_samples,
                min_samples=self.config.k_folds,
                stratify=bottom_dataset[self.config.target_col],
            )
            bottom_dataset = bottom_dataset.select(selection_indices)

        # Generate embeddings
        logger.info(
            f'Generating {model.model_input} embeddings '
            f'({len(top_dataset):,}, {len(bottom_dataset):,})'
        )

        top_cache_file = cache_dir / f'{model.config.name}_{self.config.name}_top.h5'
        bottom_cache_file = (
            cache_dir / f'{model.config.name}_{self.config.name}_bottom.h5'
        )
        with HDF5CachedList(top_cache_file, mode='w') as top_model_outputs:
            with HDF5CachedList(bottom_cache_file, mode='w') as bottom_model_outputs:
                top_sequences = top_dataset[model.model_input]
                bottom_sequences = bottom_dataset[model.model_input]

                # Generate embeddings for both datasets
                top_model_outputs = model.generate_model_outputs(
                    top_sequences, top_model_outputs, return_embeddings=True
                )
                bottom_model_outputs = model.generate_model_outputs(
                    bottom_sequences,
                    bottom_model_outputs,
                    return_embeddings=True,
                )

                # find and instantiate an output transform object
                transforms = find_transformation(
                    model.model_input, model.model_encoding, self.resolution
                )
                logger.info(f'Found transformation {[tr.name for tr in transforms]}')
                # Apply the transformations
                for transform in transforms:
                    logger.info(f'Applying {transform.name} transformation')
                    top_model_outputs.map(
                        transform.apply_h5,
                        sequences=top_sequences,
                        tokenizer=model.tokenizer,
                    )
                    bottom_model_outputs.map(
                        transform.apply_h5,
                        sequences=bottom_sequences,
                        tokenizer=model.tokenizer,
                    )

                # Sequence-level embeddings don't require any transformation of
                # top_dataset, so we can just concatenate the embeddings to the dataset
                top_embed_dict = {
                    'transformed': [output.embedding for output in top_model_outputs],
                }
                bottom_embed_dict = {
                    'transformed': [
                        output.embedding for output in bottom_model_outputs
                    ],
                }
                top_modeling_dataset = datasets.Dataset.from_dict(top_embed_dict)
                bottom_modeling_dataset = datasets.Dataset.from_dict(bottom_embed_dict)

                top_modeling_dataset = datasets.concatenate_datasets(
                    [
                        top_dataset,
                        top_modeling_dataset,
                    ],
                    axis=1,
                )
                bottom_modeling_dataset = datasets.concatenate_datasets(
                    [
                        bottom_dataset,
                        bottom_modeling_dataset,
                    ],
                    axis=1,
                )

                # Setup metrics to pass to downstream prediction model and run modeling
                top_self_metrics = MetricCollection(
                    [
                        get_and_instantiate_metric(metric)
                        for metric in self.config.metrics
                    ]
                )
                bottom_self_metrics = MetricCollection(
                    [
                        get_and_instantiate_metric(metric)
                        for metric in self.config.metrics
                    ]
                )

                downstream_modeling = get_downstream_model(
                    self.config.task_type, self.config.downstream_model
                )

                # create downstream models with each dataset
                logger.info('Training downstream models on top split')
                top_downstream_models, metrics = downstream_modeling(
                    task_dataset=top_modeling_dataset,
                    input_col='transformed',
                    target_col=self.config.target_col,
                    metrics=top_self_metrics,
                    k_fold=self.config.k_folds,
                )
                logger.info('Training downstream models on bottom split')
                bottom_downstream_models, metrics = downstream_modeling(
                    task_dataset=bottom_modeling_dataset,
                    input_col='transformed',
                    target_col=self.config.target_col,
                    metrics=bottom_self_metrics,
                    k_fold=self.config.k_folds,
                )

                # Evaluate the downstream models on the other split
                logger.info('Evaluating top models on bottom split')
                top_val_metrics = self.cross_eval(
                    top_downstream_models,
                    top_modeling_dataset,
                    bottom_modeling_dataset,
                )
                logger.info('Evaluating bottom models on top split')
                bottom_val_metrics = self.cross_eval(
                    bottom_downstream_models,
                    bottom_modeling_dataset,
                    top_modeling_dataset,
                )
                # Combine the metrics from both splits
                result_metric_collection = MetricCollection(
                    [
                        *top_self_metrics.metrics,
                        *top_val_metrics.metrics,
                        *bottom_self_metrics.metrics,
                        *bottom_val_metrics.metrics,
                    ]
                )

        # merge top/bottom models
        downstream_models = {}
        for key, ds_model in top_downstream_models.items():
            downstream_models[f'top_{key}'] = ds_model
        for key, ds_model in bottom_downstream_models.items():
            downstream_models[f'bottom_{key}'] = ds_model

        # Cleanup the cache files if they have been created by this task
        top_dataset.cleanup_cache_files()
        bottom_dataset.cleanup_cache_files()

        return downstream_models, result_metric_collection

    def cross_eval(
        self,
        downstream_models: dict[str, callable],
        train_dataset: datasets.Dataset,
        eval_dataset: datasets.Dataset,
    ) -> MetricCollection:
        """Evaluate the downstream models on the other split."""
        from harness.tasks.core.downstream.classification import object_to_label

        # Evaluate the downstream models on the other split
        metrics = MetricCollection(
            [get_and_instantiate_metric(metric) for metric in self.config.metrics]
        )

        X_train = train_dataset['transformed']
        y_train_raw = train_dataset[self.config.target_col]
        y_train = object_to_label(y_train_raw)

        X_test = eval_dataset['transformed']
        y_test_raw = eval_dataset[self.config.target_col]
        y_test = object_to_label(y_test_raw)

        for _fold_spec, model in downstream_models.items():
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            for metric in metrics:
                metric.evaluate(predicted=y_train_pred, labels=y_train, train=True)
                metric.evaluate(predicted=y_test_pred, labels=y_test, train=False)

        return metrics


class CrossEvalPfamTaxonomyBiasConfig(CrossEvalClassificationTaskConfig):
    """Configuration for the cross-eval Pfam taxonomy bias task."""

    # Name of the task
    name: Literal['CrossEvalPfamTaxonomyBias'] = 'CrossEvalPfamTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'

    # label column
    target_col: Literal['pfam'] = 'pfam'

    # For task identification and serialization
    # Taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota'] | None = (
        None
    )

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the split name, phylum-level.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the splits of
        of the same task.
        """
        self.name = f'{self.name}-{self.domain_filter}-{self.taxon_level}'
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(f'-{self.domain_filter}-{self.taxon_level}', '')


class CrossEvalPfamTaxonomyBias(CrossEvalClassificationTask):
    """Cross-eval Pfam taxonomy bias prediction classification."""

    resolution: str = 'sequence'


class CrossEvalECTaxonomyBiasConfig(CrossEvalClassificationTaskConfig):
    """Configuration for the cross-eval Pfam taxonomy bias task."""

    # Name of the task
    name: Literal['CrossEvalECTaxonomyBias'] = 'CrossEvalECTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'

    # multiple possible target columns, this will implicitly be set by the
    # dataset (as this is the 'ml ready split'), but we might need this here
    target_col: Literal['ec_0', 'ec_1', 'ec_2', 'ec_3'] = 'ec_0'

    # For task identification and serialization
    # Taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota'] | None = (
        None
    )
    # The EC level we are testing
    ec_level: Literal[0, 1, 2, 3] | None = None

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the split name, phylum-level.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the splits of
        of the same task.
        """
        self.name = (
            f'{self.name}-{self.domain_filter}-{self.taxon_level}-ec{self.ec_level}'
        )
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(
            f'-{self.domain_filter}-{self.taxon_level}-ec{self.ec_level}', ''
        )


class CrossEvalECTaxonomyBias(CrossEvalClassificationTask):
    """Cross-eval GeneEC taxonomy bias prediction classification."""

    resolution: str = 'sequence'


class CrossEvalGene3DTaxonomyBiasConfig(CrossEvalClassificationTaskConfig):
    """Configuration for the cross-eval Pfam taxonomy bias task."""

    # Name of the task
    name: Literal['CrossEvalGene3DTaxonomyBias'] = 'CrossEvalGene3DTaxonomyBias'
    # Task prediction type
    task_type: Literal['classification'] = 'classification'

    # multiple possible target columns, this will implicitly be set by the
    # dataset (as this is the 'ml ready split'), but we might need this here
    target_col: Literal['gene3d_0', 'gene3d_1', 'gene3d_2', 'gene3d_3'] = 'gene3d_0'

    # For task identification and serialization
    # Taxon level we are testing
    taxon_level: str | None = None
    # which domain task is filtered on
    domain_filter: Literal['nokingdom', 'bacteria', 'archaea', 'eukaryota'] | None = (
        None
    )
    # The EC level we are testing
    gene3d_level: Literal[0, 1, 2, 3] | None = None

    @model_validator(mode='after')
    def update_task_name(self):
        """Update the task name to have the split name, e.g phylum-level.

        This needs to be done post-init so we can successfully instantiate the task,
        but before the results are saved so that we can differentiate the splits of
        of the same task.
        """
        self.name = f'{self.name}-{self.domain_filter}-{self.taxon_level}-gene3d{self.gene3d_level}'  # noqa: E501
        return self

    @field_serializer('name', check_fields=False, when_used='json')
    def serialize_name(self, name: str):
        """Serialize the task name to remove the split name and taxon name.

        This allows us to dump the model config and reload appropriately.
        """
        return name.replace(
            f'-{self.domain_filter}-{self.taxon_level}-gene3d{self.gene3d_level}', ''
        )


class CrossEvalGene3DTaxonomyBias(CrossEvalClassificationTask):
    """Cross-eval Gene3D taxonomy bias prediction classification."""

    resolution: str = 'sequence'


# Associate the task config with the task class for explicit registration
taxonomy_bias_tasks = {
    PfamTaxonomyBiasConfig: PfamTaxonomyBias,
    ECTaxonomyBiasConfig: ECTaxonomyBias,
    Gene3DTaxonomyBiasConfig: Gene3DTaxonomyBias,
    CrossEvalPfamTaxonomyBiasConfig: CrossEvalPfamTaxonomyBias,
    CrossEvalECTaxonomyBiasConfig: CrossEvalECTaxonomyBias,
    CrossEvalGene3DTaxonomyBiasConfig: CrossEvalGene3DTaxonomyBias,
}
