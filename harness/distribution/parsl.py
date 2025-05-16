"""Parsl distribution module."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Literal

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider

from harness.api.config import BaseConfig
from harness.api.config import PathLike


class BaseComputeSettings(BaseConfig, ABC):
    """Compute settings (HPC platform, number of GPUs, etc)."""

    name: Literal[''] = ''
    """Name of the platform to use."""

    @abstractmethod
    def get_config(self, run_dir: PathLike) -> Config:
        """Create a new Parsl configuration.

        Parameters
        ----------
        run_dir : PathLike
            Path to store monitoring DB and parsl logs.

        Returns
        -------
        Config
            Parsl configuration.
        """
        ...


class SingleNodeSettings(BaseComputeSettings):
    """Settings for a single node or workstation with one or more GPUs."""

    name: Literal['singlenode'] = 'singlenode'  # type: ignore[assignment]
    """Name of the platform."""
    available_accelerators: int | str = 8
    """Number of GPU accelerators to use."""
    worker_port_range: tuple[int, int] = (55000, 56000)
    """Port range."""
    retries: int = 1
    label: str = 'htex'

    def get_config(self, run_dir: PathLike) -> Config:
        """Create a parsl configuration for running on a workstation."""
        return Config(
            run_dir=str(run_dir),
            retries=self.retries,
            executors=[
                HighThroughputExecutor(
                    label=self.label,
                    cpu_affinity='block',
                    available_accelerators=self.available_accelerators,
                    worker_port_range=self.worker_port_range,
                    provider=LocalProvider(init_blocks=1, max_blocks=1),
                    heartbeat_threshold=300,
                ),
            ],
        )
