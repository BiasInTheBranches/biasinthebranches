"""Evaluation entrypoint for the benchmarking pipeline."""

from __future__ import annotations

import logging
from argparse import ArgumentParser
from concurrent.futures import as_completed
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

from parsl.concurrent import ParslPoolExecutor
from pydantic import Field
from pydantic import model_validator

from harness.api.config import BaseConfig
from harness.api.logging import logger
from harness.distribution import ParslConfigTypes
from harness.modeling import ModelConfigTypes
from harness.tasks import TaskConfigTypes


class EvalConfig(BaseConfig):
    """Configuration for the benchmarking pipeline."""

    # Language model configuration
    lm_config: ModelConfigTypes

    # List of task configurations
    task_configs: list[TaskConfigTypes]

    # For distributed evaluation using parsl
    parsl_config: ParslConfigTypes | None = None

    #### General evaluation settings ####
    # Results output directory
    output_dir: Path = Field(
        default_factory=lambda: Path(
            f'results-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )
    )
    # Cache dir for intermediate results (different from model cache dir -
    # this is where the model is downloaded)
    cache_dir: Path = None

    # Setup default models for downstream tasks (can be overridden by task config)
    regression_model: Literal['mlp', 'svr'] = 'svr'
    classification_model: Literal['mlp', 'svc'] = 'svc'
    multi_label_classification_model: Literal['mlp'] = 'mlp'

    # Whether or not to save the downstream models
    save_downstream_models: bool = False

    @model_validator(mode='after')
    def set_cache_dir(self):
        """Set the cache directory to be within the output directory if not provided."""
        # Set cache_dir to be within output_dir if not explicitly provided
        if self.cache_dir is None:
            self.cache_dir = Path(self.output_dir) / 'cache'

        return self


def setup_evaluations(eval_config: EvalConfig):
    """Setup environment for the evaluations."""
    eval_config.output_dir.mkdir(parents=True, exist_ok=True)
    eval_config.cache_dir.mkdir(parents=True, exist_ok=True)

    # 'Push down' the global config options that need to be set
    # for the tasks to use.
    for task_config in eval_config.task_configs:
        # Setup the downstream model if task requires it but
        # no model is provided
        if (
            hasattr(task_config, 'downstream_model')
            and task_config.downstream_model is None
        ):
            match task_config.task_type:
                case 'classification':
                    task_config.downstream_model = eval_config.classification_model
                case 'regression':
                    task_config.downstream_model = eval_config.regression_model
                case 'multi-label-classification':
                    task_config.downstream_model = (
                        eval_config.multi_label_classification_model
                    )

    # Dump the original config for reproducibility
    eval_config.write_yaml(eval_config.output_dir / 'config.yaml')


def evaluate_task(
    task_config: TaskConfigTypes,
    model_config: ModelConfigTypes,
    output_dir: Path,
    cache_dir: Path,
    save_downstream_models: bool = False,
):
    """Evaluate a task given a configuration and a model."""
    import joblib

    from harness.api.logging import logger
    from harness.modeling import get_model
    from harness.tasks import get_task

    # Get a model instance, will be instantiated on current device if not already
    model = get_model(model_config=model_config, register=True)
    logger.info(f'Setup {model.config.name}')

    # Find the task class and config class
    task = get_task(task_config)
    logger.info(f'Setup {task.config.name}')

    # Run the evaluation and get metrics
    downstream_models, metrics = task.evaluate(model, cache_dir)

    # Save metrics and report
    metric_save_path = output_dir / f'{model.config.name}_{task_config.name}.metrics'
    metrics.save(metric_save_path)
    for metric in metrics:
        logger.info(metric.report())

    if save_downstream_models:
        # Save the downstream models
        downstream_model_path = (
            output_dir
            / f'{model.config.name}_{task_config.name}.downstreammodels.joblib'
        )
        joblib.dump(downstream_models, downstream_model_path)
        logger.info(f'Saved downstream models to {downstream_model_path}')


def evaluate(eval_config: EvalConfig):
    """Evaluate the models on the tasks."""
    setup_evaluations(eval_config)
    logger.info(f'Language model config: {eval_config.lm_config}')

    evaluate_function = partial(
        evaluate_task,
        model_config=eval_config.lm_config,
        output_dir=eval_config.output_dir,
        cache_dir=eval_config.cache_dir,
        save_downstream_models=eval_config.save_downstream_models,
    )

    if eval_config.parsl_config is not None:
        # Initialize Parsl
        logger.info('Initializing Parsl')
        parsl_run_dir = eval_config.output_dir / 'parsl'
        parsl_config = eval_config.parsl_config.get_config(parsl_run_dir)

        logger.info('Beginning distributed evaluation')

        with ParslPoolExecutor(parsl_config) as pool:
            task_futures = {}
            for task_config in eval_config.task_configs:
                # Submit tasks to be executed
                future = pool.submit(evaluate_function, task_config=task_config)
                task_futures[future] = task_config
                logger.info(f'Submitted evaluation for {task_config.name}')

            # Process futures as they complete
            for completed_future in as_completed(task_futures):
                task_config = task_futures[completed_future]
                try:
                    _ = completed_future.result()  # Get the result or raise exception
                    logger.info(f'Evaluation complete for {task_config.name}')
                except Exception as e:
                    logger.error(f'Task {task_config.name} failed with error: {e!s}')

    else:
        # Evaluate tasks (task logs will be on same stream as main, no need to log)
        for task_config in eval_config.task_configs:
            evaluate_function(task_config=task_config)

    logger.info(f'Evaluation complete (results: {eval_config.output_dir})')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', required=True, help='Path to the evaluation config file'
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logger.debug('Debug logging enabled')

    config = EvalConfig.from_yaml(args.config)
    evaluate(config)
