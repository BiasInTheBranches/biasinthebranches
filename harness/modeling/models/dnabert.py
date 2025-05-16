"""Implementation of the DNABERT model."""

from __future__ import annotations

from typing import Any
from typing import Literal

import torch
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


class DNABERT2Config(LMConfig):
    """Config for DNABERT."""

    name: Literal['DNABERT2'] = 'DNABERT2'
    # Model id or path to load the model
    pretrained_model_name_or_path: str = 'zhihan1996/DNABERT-2-117M'
    # path to HF cache if download needed
    cache_dir: str | None = None

    # Model does not support specific layers hidden states,
    # only returns the last hidden state


class DNABERT2(LM):
    """DNABERT2 wrapper model."""

    model_input: str = 'dna'
    model_encoding: str = 'bpe'

    def __init__(self, config: DNABERT2Config) -> None:
        """Initialize the DNABERT."""
        # The version of triton used by the original authors no longer works. Default
        # to the transformers library attention for this specific model only
        import sys

        triton_module = sys.modules.get('triton')
        sys.modules['triton'] = None
        from transformers import AutoModelForMaskedLM
        from transformers import AutoTokenizer
        from transformers.models.bert.configuration_bert import BertConfig

        sys.modules['triton'] = triton_module

        model_kwargs = {}
        if config.cache_dir:
            model_kwargs['cache_dir'] = config.cache_dir

            # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
        )

        # Load model
        model_cfg = BertConfig.from_pretrained(config.pretrained_model_name_or_path)
        model = AutoModelForMaskedLM.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=True,
            config=model_cfg,
            **model_kwargs,
        )

        # Set the model to evaluation mode
        model.eval()

        # Load the model onto the device
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu',
        )
        model.to(device)

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
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**batch, output_hidden_states=True)

                    # Get the required tokenizer data (no bos/eos in DNABERT model)
                    input_ids = batch['input_ids']
                    seq_lengths = batch['attention_mask'].sum(axis=1)

                    # Move the outputs to the CPU
                    logits = outputs.logits.cpu().detach().numpy()
                    if return_embeddings:
                        # Get the last hidden state (only hidden state for this model)
                        last_hidden_state = outputs.hidden_states
                        embedding = last_hidden_state.cpu().detach().numpy()
                    else:
                        embedding = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the the padding (no bos/eos in DNABERT model)
                        if return_input_ids:
                            seq_input_ids = (
                                input_ids[i, :seq_len].cpu().detach().numpy()
                            )
                        if return_logits:
                            seq_logits = logits[i, :seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, :seq_len, :]
                        if return_attention_maps:
                            # Model does not support attention outputs
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


dnabert_models = {DNABERT2Config: DNABERT2}
