"""Implementations of ESM(2/3) models."""

from __future__ import annotations

from typing import Any
from typing import Literal

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from transformers import PreTrainedTokenizer

from harness.api.logging import logger
from harness.api.modeling import HDF5CachedList
from harness.api.modeling import LM
from harness.api.modeling import LMConfig
from harness.api.modeling import SequenceModelOutput


class ESMConfig(LMConfig):
    """ESM configuration."""

    name: Literal['ESM'] = 'ESM'
    # Model id or path to load the model
    pretrained_model_name_or_path: str
    # path to HF cache if download needed
    cache_dir: str | None = None
    # Use the model in half precision
    half_precision: bool = False

    # Which embedding layer to choose from the model
    # -1 is the last layer, there are 1+num layers in the
    # model as the first layer is the embedding layer
    embedding_layer: int = -1


class ESM(LM):
    """ESM2 wrapper model."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: ESMConfig) -> None:
        """Initialize the Nucleotide transformer."""
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            cache_dir=config.cache_dir,
        )

        # Load model
        model = AutoModelForMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )

        # Convert the model to half precision
        if config.half_precision:
            model.half()

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """HF Tokenizer object."""
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Tokenizer configuration options."""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Dataloader configuration options."""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def device(self) -> torch.device:
        """Torch device the model is placed on."""
        return self.model.device

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = False,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate embeddings, logits, attention masks for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')
        logger.info('Tokenized dataset.')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    # before moving to device
                    input_ids = batch['input_ids']
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    batch = {k: v.to(self.model.device) for k, v in batch.items()}
                    outputs = self.model(
                        **batch,
                        output_hidden_states=return_embeddings,
                        output_attentions=return_attention_maps,
                    )

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()

                    # Extract (hf) optional model outputs
                    if return_embeddings:
                        # Get the last hidden state
                        hidden_state = outputs.hidden_states[
                            self.config.embedding_layer
                        ]
                        embedding = hidden_state.cpu().detach().numpy()
                    else:  # return_embeddings is False
                        embedding = None

                    if return_attention_maps:
                        attention_maps = [
                            layer_attn.cpu().detach().numpy()
                            for layer_attn in outputs.attentions
                        ]
                        attention_maps = np.stack(attention_maps, axis=1)
                    else:
                        attention_maps = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the bos token and the padding
                        if return_input_ids:
                            seq_input_ids = input_ids[i, 1:seq_len]
                        if return_logits:
                            seq_logits = logits[i, 1:seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, 1:seq_len, :]
                        if return_attention_maps:
                            # Attention maps are of shaep (B, L, H, T, T)
                            seq_attention_maps = attention_maps[
                                i, :, :, 1:seq_len, 1:seq_len
                            ]

                        output_fields = {
                            'input_ids': seq_input_ids,
                            'logits': seq_logits,
                            'embedding': seq_embedding,
                            'attention_maps': seq_attention_maps,
                        }

                        # Create the output object
                        model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


class ESMCConfig(LMConfig):
    """ESMC configuration."""

    name: Literal['ESMC'] = 'ESMC'
    # Model id or path to load the model
    pretrained_model_name_or_path: str = 'esmc_300m'
    # path to HF cache if download needed
    cache_dir: str | None = None

    # Model only returns last hidden state, there
    # is no need to specify an embedding layer


class ESMC(LM):
    """ESMC wrapper module."""

    model_input = 'aminoacid'
    model_encoding = 'char'

    def __init__(self, config: ESMCConfig) -> None:
        import os

        from esm.models.esmc import ESMC
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

        # Set the cache directory as its not exposed by the esm api
        os.environ['HF_HOME'] = config.cache_dir

        # Load the model
        model = ESMC.from_pretrained(config.pretrained_model_name_or_path)

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

        # Load the tokenizer
        tokenizer = EsmSequenceTokenizer()

        # Set persistent attributes
        self.config = config
        self.model = model
        self._tokenizer = tokenizer
        self._device = device

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """HF Tokenizer object."""
        return self._tokenizer

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Tokenizer configuration options."""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Dataloader configuration options."""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def device(self) -> torch.device:
        """Torch device the model is placed on."""
        return self._device

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = False,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate embeddings, logits, attention masks for sequence input."""

        # Tokenize the dataset
        def tokenize_input(examples):
            return self.tokenizer(examples['sequences'], **self.tokenizer_config)

        modeling_input = {'sequences': sequences}
        modeling_dataset = Dataset.from_dict(modeling_input)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')
        logger.info('Tokenized dataset')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # The model takes lots of types of inputs in different tracks
                    # Until we can support non-sequence types the only thing
                    # needed is input_ids
                    input_ids = batch['input_ids']
                    # Get the sequence lengths  bos/eos in esm model, remove last token)
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1
                    outputs = self.model(
                        sequence_tokens=batch['input_ids'].to(self.device)
                    )

                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1

                    # Move the outputs to the CPU
                    # Cast to float16 since model is in bfloat16
                    # (this could lead to loss of precision?)
                    logits = (
                        outputs.sequence_logits.detach().cpu().to(torch.float16).numpy()
                    )

                    # Extract (hf) optional model outputs
                    if return_embeddings:
                        # Get the last hidden state
                        last_hidden_state = outputs.embeddings
                        embedding = (
                            last_hidden_state.detach().cpu().to(torch.float16).numpy()
                        )
                    else:  # return_embeddings is False
                        embedding = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the bos token and the padding
                        if return_input_ids:
                            seq_input_ids = input_ids[i, 1:seq_len]
                        if return_logits:
                            seq_logits = logits[i, 1:seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, 1:seq_len, :]
                        if return_attention_maps:
                            # Attention maps are not natively returned by ESMC API
                            seq_attention_maps = None

                        output_fields = {
                            'input_ids': seq_input_ids,
                            'logits': seq_logits,
                            'embedding': seq_embedding,
                            'attention_maps': seq_attention_maps,
                        }

                        # Create the output object
                        model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(self, input: list[str]) -> list[SequenceModelOutput]:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


esm_models = {
    ESMConfig: ESM,
    ESMCConfig: ESMC,
}
