"""Configuration module."""

from __future__ import annotations

import json
from hashlib import sha512
from pathlib import Path
from typing import TypeVar
from typing import Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel

PathLike = Union[str, Path]  # noqa:  UP007

T = TypeVar('T')


class BaseConfig(BaseModel):
    """An interface to add JSON/YAML serialization to Pydantic models."""

    def __hash__(self):
        """Allow for hashing of the model."""
        return int.from_bytes(
            sha512(
                f'{self.__class__.__qualname__}::{self.model_fields}'.encode(
                    'utf-8', errors='ignore'
                )
            ).digest()
        )

    def write_json(self, path: PathLike) -> None:
        """Write the model to a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.
        """
        with open(path, 'w') as fp:
            json.dump(self.model_dump(), fp, indent=2)

    @classmethod
    def from_json(cls: type[T], path: PathLike) -> T:
        """Load the model from a JSON file.

        Parameters
        ----------
        path : str
            The path to the JSON file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            data = json.load(fp)
        return cls(**data)

    def write_yaml(self, path: PathLike) -> None:
        """Write the model to a YAML file.

        Parameters
        ----------
        path : str
            The path to the YAML file.
        """
        with open(path, 'w') as fp:
            yaml.dump(
                json.loads(self.model_dump_json()),
                fp,
                indent=4,
                sort_keys=False,
            )

    @classmethod
    def from_yaml(cls: type[T], path: PathLike) -> T:
        """Load the model from a YAML file.

        Parameters
        ----------
        path : PathLike
            The path to the YAML file.

        Returns
        -------
        T
            A specific BaseConfig instance.
        """
        with open(path) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)
