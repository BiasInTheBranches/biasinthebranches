"""API for protein/genome language models and related utilities."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal
from typing import Protocol

import h5py
import numpy as np

from harness.api.config import BaseConfig


class TorchDataloaderConfig(BaseConfig):
    """Config for the torch DataLoader class."""

    # Batch size
    batch_size: int = 1
    # Whether to shuffle the data
    shuffle: bool = False
    # If dataset size // batch_size is not an integer, drop the last incomplete batch
    drop_last: bool = False
    # Pin memory for faster GPU transfer
    pin_memory: bool = True


class TokenizerConfig(BaseConfig):
    """Config for tokenizer encode arguments."""

    # Padding strategy
    padding: str | bool = 'max_length'
    # Truncation strategy
    truncation: str | bool = True
    # Maximum length of the sequence
    max_length: int = 1024
    # Return type of the tokenizer
    return_tensors: str = 'pt'


class LMConfig(BaseConfig):
    """Base configuration class for a language model."""

    # Name of the model - to be set by the instance
    name: Literal[''] = ''

    # Embedding layer index - some models don't support indexing
    embedding_layer: int | None = None

    # Tokenizer encode configuration options
    tokenizer_config: TokenizerConfig = field(default_factory=TokenizerConfig)
    # dataloader config for the forward passes
    dataloader_config: TorchDataloaderConfig = field(
        default_factory=TorchDataloaderConfig
    )


@dataclass
class SequenceModelOutput:
    """Container for outputs of a biology sequence model.

    This behaves minimally like a dictionary so that it can be used in a dataset.
    """

    sequence: str | None = field(
        default=None, metadata={'description': 'Generated sequence.'}
    )

    input_ids: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The input ids of the sequence (shape: [sequence_length]).'
        },
    )

    logits: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The logits of the sequence '
            '(shape: [sequence_length, vocab_size]).'
        },
    )

    embedding: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The sequence embeddings '
            '(shape: [sequence_length, embedding_size]).'
        },
    )
    attention_maps: np.ndarray | None = field(
        default=None,
        metadata={
            'description': 'The attention maps of the sequence '
            '(shape: [num_heads, sequence_length, sequence_length]).'
        },
    )

    def __getitem__(self, key: str) -> Any:
        """Get the value of an attribute if it's not None."""
        value = getattr(self, key)
        if value is None:
            raise KeyError(key)
        return value

    def __iter__(self) -> Iterator[str]:
        """Iterate over the non-None attributes of the class."""
        return (k for k, v in self.__dict__.items() if v is not None)

    def __len__(self) -> int:
        """Get the number of non-None attributes in the class."""
        return sum(1 for v in self.__dict__.values() if v is not None)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value of an attribute with a default value."""
        value = getattr(self, key, default)
        return value if value is not None else default


class HDF5CachedList:
    """A list-like container that caches SequenceModelOutput objects in an HDF5 file."""

    def __init__(self, file_path: str, mode: str = 'w'):
        self.file_path = file_path
        # Open the HDF5 file in write mode, create it if it doesn't exist
        self.hdf5_file = h5py.File(self.file_path, mode)
        # Keep track of the number of items
        self.length = self.hdf5_file.attrs.get('length', 0)

        # Setup compression options
        # https://docs.h5py.org/en/stable/high/dataset.html#h5py.Dataset.compression
        self.compression_options = {'compression': None, 'compression_opts': None}

    def append(self, obj: SequenceModelOutput):
        """Append a SequenceModelOutput object to the list."""
        idx = self.length
        group = self.hdf5_file.create_group(f'{idx}')
        # Store the sequence as an attribute
        if obj.sequence is not None:
            group.attrs['sequence'] = obj.sequence
        # Store the arrays as datasets
        if obj.input_ids is not None:
            group.create_dataset(
                'input_ids', data=obj.input_ids, **self.compression_options
            )
        if obj.logits is not None:
            group.create_dataset('logits', data=obj.logits, **self.compression_options)
        if obj.embedding is not None:
            group.create_dataset(
                'embedding', data=obj.embedding, **self.compression_options
            )
        if obj.attention_maps is not None:
            group.create_dataset(
                'attention_maps', data=obj.attention_maps, **self.compression_options
            )
        # Update the length
        self.length += 1
        self.hdf5_file.attrs['length'] = self.length

    def __getitem__(self, idx: int) -> SequenceModelOutput:
        """Retrieve a SequenceModelOutput object."""
        if idx < 0 or idx >= self.length:
            raise IndexError('Index out of range')
        group = self.hdf5_file[f'{idx}']
        sequence = group.attrs.get('sequence', None)
        input_ids = group['input_ids'][()] if 'input_ids' in group else None
        logits = group['logits'][()] if 'logits' in group else None
        embedding = group['embedding'][()] if 'embedding' in group else None
        attention_maps = (
            group['attention_maps'][()] if 'attention_maps' in group else None
        )
        return SequenceModelOutput(
            sequence=sequence,
            input_ids=input_ids,
            logits=logits,
            embedding=embedding,
            attention_maps=attention_maps,
        )

    def __setitem__(self, idx: int, obj: SequenceModelOutput):
        """Update a SequenceModelOutput object at a given index."""
        if idx < 0 or idx >= self.length:
            raise IndexError('Index out of range')
        group = self.hdf5_file[f'{idx}']
        # Update sequence
        if obj.sequence is not None:
            group.attrs['sequence'] = obj.sequence
        # Update datasets
        for name in ['input_ids', 'logits', 'embedding', 'attention_maps']:
            if getattr(obj, name) is not None:
                data = getattr(obj, name)
                if name in group:
                    del group[name]  # Delete the old dataset
                group.create_dataset(name, data=data, **self.compression_options)
            elif getattr(obj, name) is None and name in group:
                # Delete the dataset if the new value is None
                del group[name]

    def __len__(self):
        """Return the number of items in the list."""
        return self.length

    def __iter__(self):
        """Iterate over the items."""
        for idx in range(self.length):
            yield self[idx]

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()

    def __del__(self):
        """Ensure the HDF5 file is closed upon deletion."""
        try:
            self.hdf5_file.close()
        except Exception:
            pass

    def map(self, func, **kwargs):
        """
        Apply a function to each item in the list and update the item in place.

        The function should accept a SequenceModelOutput object and return a modified
        SequenceModelOutput object.
        """
        # Find iterable kwargs and pass them to the function
        identifiers = [k for k, v in kwargs.items() if isinstance(v, list)]

        assert all(len(kwargs[k]) == self.length for k in identifiers)

        for idx in range(self.length):
            obj = self[idx]  # Retrieve the object
            fnc_kwargs = {k: kwargs[k][idx] for k in identifiers}
            fnc_kwargs.update({k: v for k, v in kwargs.items() if k not in identifiers})
            new_obj = func(obj, **fnc_kwargs)  # Apply the transformation
            self[idx] = new_obj  # Update the object in the HDF5 file

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and close the HDF5 file."""
        self.close()


class LM(Protocol):
    """Interface for a general protein language model."""

    model_input: str
    model_encoding: str

    @property
    def tokenizer(self) -> Any:
        """Get the tokenizer of the encoder."""
        ...

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        ...

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        ...

    @property
    def device(self):
        """Accelerator object model is running on."""
        ...

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = False,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Run the model on a list of input sequences.

        Responsible for getting embeddings, logits, attention masks if specified.
        """
        ...

    def generate_sequences(
        self, input: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        ...


class Transform(ABC):
    """Base class for a transformation."""

    @staticmethod
    @abstractmethod
    def apply(input: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Transform outputs from a sequence model."""
        ...

    @staticmethod
    @abstractmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Transform a single output from a sequence model."""
        ...
