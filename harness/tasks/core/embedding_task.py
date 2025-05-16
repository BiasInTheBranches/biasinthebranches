"""General implementatino for a prediction task using embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import datasets

from harness.api.logging import logger
from harness.api.metric import Metric
from harness.api.metric import MetricCollection
from harness.api.modeling import HDF5CachedList
from harness.api.modeling import LM
from harness.api.task import DownstreamModel
from harness.api.task import Task
from harness.api.task import TaskConfig
from harness.metrics import get_and_instantiate_metric
from harness.tasks.core.downstream import get_downstream_model
from harness.tasks.core.downstream.classification import balance_classes
from harness.tasks.core.utils import find_transformation
from harness.tasks.core.utils import flatten_to_token_level
from harness.tasks.core.utils import limit_training_samples


class EmbeddingTaskConfig(TaskConfig):
    """Configuration for a general embedding property prediction task."""

    # The implementation of the task should include the name ,
    # task prediction type, metrics, and downstream model type
    name: Literal[''] = ''
    metrics: list[str]
    task_type: Literal['classification', 'regression', 'multi-label-classification']
    downstream_model: Literal['mlp', 'svc', 'svr'] | None = None

    # Task specific information just need the label column for now
    target_col: str = 'label'
    # Whether to balance classes
    balance_classes: bool = False
    # Limit to number of training samples
    max_samples: int | None = None
    # K-fold cross validation
    k_folds: int = 5
    # Truncate ends of sequence embeddings (common for amino acid resolution tasks)
    truncate_end: bool = False


class EmbeddingTask(Task):
    """Base implementation for any prediction task using embeddings.

    This implementation handles _BOTH_ character level and sequence level tasks.
    """

    resolution: str

    def __init__(self, config: TaskConfig):
        super().__init__(config)
        assert hasattr(self, 'resolution'), 'Resolution must be set in the subclass'

    def evaluate(
        self, model: LM, cache_dir: Path
    ) -> tuple[dict[str, DownstreamModel | None], list[Metric]]:
        """Evaluate an embedding task on any input."""
        char_level = self.resolution in ['aminoacid', 'nucleotide']
        logger.info(f'Task resolution: {self.resolution} (char level: {char_level})')
        # Load the dataset
        # NOTE: Originally I set format to numpy, but it prohibits multi-dimension
        # arrays being concatenated into the downstream dataset, removing it does
        # not seem to cause issues.
        task_dataset: datasets.Dataset = datasets.load_from_disk(
            self.config.dataset_name_or_path
        )

        # Preemptively balance the classes if applicable and specified
        if (
            self.config.task_type == 'classification'
            and self.config.balance_classes
            and not char_level
        ):
            task_dataset = balance_classes(task_dataset, self.config.target_col)

        # Limit training samples if specified. Tricky here because in char level tasks
        # we both need to limit the number of sequences we do inference on
        # and the number of samples used in downstream modeling.
        char_level_limit = char_level and 'max_sequences' in self.config.model_fields
        sequence_level_limit = (
            not char_level
            and 'max_samples' in self.config.model_fields
            and self.config.max_samples is not None
        )
        if char_level_limit and not sequence_level_limit:
            logger.debug('Limiting samples for char level task (max_sequences)')
            task_dataset = limit_training_samples(
                task_dataset=task_dataset,
                max_samples=self.config.max_sequences,
                input_col=model.model_input,
                target_col=self.config.target_col
                if self.config.task_type != 'multi-label-classification'
                else None,
                continuous=self.config.task_type == 'regression',
            )
        elif sequence_level_limit and not char_level_limit:
            logger.debug('Limiting samples for sequence level task (max_samples)')
            task_dataset = limit_training_samples(
                task_dataset=task_dataset,
                max_samples=self.config.max_samples,
                input_col=model.model_input,
                target_col=self.config.target_col
                if self.config.task_type != 'multi-label-classification'
                else None,
                continuous=self.config.task_type == 'regression',
            )
        else:
            if (
                'max_sequences' in self.config.model_fields
                and self.config.max_sequences is not None
            ):
                logger.error(
                    'Logic error: could not limit samples using `max_sequences`'
                )
            if (
                'max_samples' in self.config.model_fields
                and self.config.max_samples is not None
                and not char_level
            ):
                logger.error('Logic error: could not limit samples using `max_samples`')

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

            # Create the downstream modeling dataset
            if self.resolution == 'sequence':
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
            elif self.resolution in ['aminoacid', 'nucleotide']:
                # Flatten the dataset to the token level for downstream modeling
                modeling_dataset = flatten_to_token_level(
                    task_dataset=task_dataset,
                    model_outputs=model_outputs,
                    truncate_end=self.config.truncate_end,
                )

                # Balance the classes and limit the number of training samples
                # if applicable
                if (
                    'max_samples' in self.config.model_fields
                    and self.config.max_samples is not None
                ):
                    if self.config.balance_classes:
                        logger.debug('Balancing classes')
                        modeling_dataset = balance_classes(
                            task_dataset=modeling_dataset,
                            target_col=self.config.target_col,
                        )
                    modeling_dataset = limit_training_samples(
                        modeling_dataset,
                        self.config.max_samples,
                        model.model_input,
                        target_col=self.config.target_col,
                        continuous=self.config.task_type == 'regression',
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
