"""Implementation of the AnkH model."""

from __future__ import annotations

from typing import Any
from typing import Literal

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
from harness.modeling.utils.data import sequences_to_dataset


class AnkhConfig(LMConfig):
    """AnkH configuration."""

    name: Literal['Ankh'] = 'Ankh'
    # Model id or path to load the model
    size: str = 'base'
    # path to HF cache if download needed
    cache_dir: str | None = None

    # Embedding layer to use for embeddings AnkH has 48 layers for both configurations
    # The first layer is embedding layer - then 48 transformer layers
    embedding_layer: int = 48


class Ankh(LM):
    """AnkH wrapper class."""

    model_input: str = 'aminoacid'
    model_encoding: str = 'char'

    def __init__(self, config: AnkhConfig) -> None:
        """Initialize the Ankh model. Requires `pip install ankh`."""
        import os

        import ankh

        os.environ['HF_HOME'] = config.cache_dir

        # Load the model and tokenizer
        if config.size.lower() == 'large':
            model, tokenizer = ankh.load_large_model()
        else:
            model, tokenizer = ankh.load_base_model()

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
            seqs = [list(s) for s in examples['sequences']]
            return self.tokenizer(
                seqs,
                add_special_tokens=True,
                is_split_into_words=True,
                **self.tokenizer_config,
            )

        modeling_dataset = sequences_to_dataset(sequences)
        modeling_dataset = modeling_dataset.map(
            tokenize_input,
            batched=True,
            remove_columns=['sequences'],
        ).with_format('torch')

        # turn into dataloader and grab dset info
        dataloader = DataLoader(modeling_dataset, **self.dataloader_config)

        # Generate embeddings
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    input_ids = batch['input_ids']
                    outputs = self.model(
                        input_ids.to(self.device),
                        batch['attention_mask'].to(self.device),
                        output_hidden_states=True,
                    )
                    # Get the sequence lengths, only eos token, remove last
                    seq_lengths = batch['attention_mask'].sum(axis=1) - 1
                    # Extract (hf) optional model outputs
                    if return_embeddings:
                        # Get the last hidden state
                        hidden_state = outputs.hidden_states[
                            self.config.embedding_layer
                        ]
                        # Move the outputs to the CPU
                        embedding = hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None
                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Only an EOS token, removed by subtracting 1 from attn length
                        if return_input_ids:
                            seq_input_ids = input_ids[i, :seq_len]
                        if return_logits:
                            # must feed outputs through decoder for logits
                            seq_logits = None
                        if return_embeddings:
                            seq_embedding = embedding[i, :seq_len]
                        if return_attention_maps:
                            # Model does not seem to support returning attention maps
                            # OOTB, neither the init with output_attentions=True or
                            # forward pass output_attentions=True seems to work
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

    def generate_sequences(self, input: list[str]) -> Dataset:
        """Generate sequences from one or more input prompts."""
        raise NotImplementedError


ankh_models = {
    AnkhConfig: Ankh,
}
