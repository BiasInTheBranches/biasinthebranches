"""Implementation of tasks for harness."""

from __future__ import annotations

from typing import Union

from harness.api.logging import logger

from .taxonomy_bias import taxonomy_bias_tasks

# Registry of tasks - append when adding new tasks
task_registry = {
    **taxonomy_bias_tasks,
}

TaskConfigTypes = Union[*task_registry.keys(),]


# Utility for getting and instantiating a task from a config
def get_task(task_config: TaskConfigTypes):
    """Get a task instance from a config."""
    # Find the task class and config class
    task_cls = task_registry.get(task_config.__class__)
    if task_cls is None:
        logger.debug(f'Task {task_config.__class__} not found in registry')
        logger.debug(f'Available tasks:\n\t{task_registry.keys()}')
        raise ValueError(f'Task {task_config.__class__} not found in registry')

    return task_cls(task_config)
