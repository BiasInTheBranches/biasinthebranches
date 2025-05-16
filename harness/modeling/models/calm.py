"""CaLM model implementation."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from typing import Literal
import os

import requests
import torch
from pydantic import Field
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from harness.api.logging import logger
from harness.api.modeling import HDF5CachedList
from harness.api.modeling import LM
from harness.api.modeling import LMConfig
from harness.api.modeling import SequenceModelOutput

try:
    from calm.alphabet import Alphabet as _Alphabet
    from calm.alphabet import BatchConverter
    from calm.sequence import CodonSequence
except ImportError:
    # Patch the classes if the CaLM package is not installed
    _Alphabet = Any
    BatchConverter = Any
    CodonSequence = Any


class Alphabet(_Alphabet):
    """Patched version of the CaLM Alphabet class.

    Will fill in unknown tokens instead of erroring out.
    """

    def encode(self, text):
        """Encode text into token indices."""
        return [self.tok_to_idx.get(tok, self.unk_idx) for tok in self.tokenize(text)]


class CaLMDataset(Dataset):
    """Dataset for CaLM model."""

    def __init__(self, sequences: list[str]):
        self.sequences = sequences

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        """Get an item from the dataset."""
        return self.sequences[idx]


class CaLMDataCollator:
    """Data collator for CaLM model."""

    def __init__(self, batch_converter: BatchConverter):
        self.batch_converter = batch_converter

    def __call__(self, sequences: list[str]) -> torch.Tensor:
        """Collate the examples."""
        # Make all sequences modulo 3 otherwise tokenizer will error
        sequences = [seq[: len(seq) - len(seq) % 3] for seq in sequences]

        # Truncate to 1022 codons (max sequence length of model, need start/end tokens)
        sequences = [seq[: 1022 * 3] for seq in sequences]

        # Convert the sequences to CodonSequence objects to handle
        # T/U conversion, 3-mer splitting, and add cls/eos tokens
        seqs = [CodonSequence(seq) for seq in sequences]

        # Tokenize the sequences
        _, _, tokens = self.batch_converter([('', seq.seq) for seq in seqs])

        return tokens


class CaLMConfig(LMConfig):
    """Config for CaLM model."""

    # The name of the encoder
    name: Literal['CaLM'] = 'CaLM'  # type: ignore[assignment]

    checkpoint_path: Path = Field(
        default=Path(os.environ.get('HF_HOME', Path.home()))
        / '.harness'
        / 'models'
        / 'calm_weights.pkl',
        description='Model checkpoint path (/path/to/calm_weights.pkl).',
    )
    half_precision: bool = Field(default=False, description='Use half precision.')

    # Embedding layer to extract embeddings from, CaLM has 12 layers
    embedding_layer: int = 12


class CaLM(LM):
    """CaLM model.

    Module to use the Codon adaptation Language Model (CaLM)
    as published in C. Outeiral and C. M. Deane, "Codon language
    embeddings provide strong signals for protein engineering",
    bioRxiv (2022), doi: 10.1101/2022.12.15.519894.
    """

    model_input: str = 'dna'
    model_encoding: str = '3mer'

    def __init__(self, config: CaLMConfig) -> None:
        """Initialize the CaLM model.

        Note: If the model weights are not found at the checkpoint path,
        they will be downloaded from the CaLM repository.

        Note: Sometimes the model NaNs out for normal (looking) sequences. This is
        handled in the downstream task creation through filtering NaNs. This happens
        regardless of precision.
        """
        from calm.model import ProteinBertModel
        from calm.pretrained import ARGS

        # Load the model and data collator
        alphabet = Alphabet.from_architecture('CodonModel')
        model = ProteinBertModel(args=ARGS, alphabet=alphabet)
        batch_converter = alphabet.get_batch_converter()
        data_collator = CaLMDataCollator(batch_converter=batch_converter)

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

        self.config = config
        self.model = model
        self.data_collator = data_collator
        self._device = device

        # Load the model weights
        self._load_weights(config.checkpoint_path)

    def _load_weights(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.exists():
            print('Downloading CaLM model weights...')

            # Make the parent directory if it doesn't exist
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the model weights and save them to the checkpoint path
            url = 'http://opig.stats.ox.ac.uk/data/downloads/calm_weights.pkl'
            with open(checkpoint_path, 'wb') as fp:
                fp.write(requests.get(url).content)

        with open(checkpoint_path, 'rb') as fp:
            state_dict = pickle.load(fp)
            self.model.load_state_dict(state_dict)

    @property
    def tokenizer(self) -> BatchConverter:
        """Get the tokenizer of the encoder."""
        return self.data_collator.batch_converter

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        return (
            self.config.tokenizer_config.model_dump()
            if self.config.tokenizer_config
            else {}
        )

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        return (
            self.config.dataloader_config.model_dump()
            if self.config.dataloader_config
            else {}
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the encoder."""
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
        # Set up the torch dataset and dataloader
        dataset = CaLMDataset(sequences)
        dataloader = DataLoader(
            dataset, collate_fn=self.data_collator, **self.dataloader_config
        )

        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []
        with torch.no_grad():
            with logging_redirect_tqdm(loggers=[logger]):
                for batch in tqdm(dataloader, desc='Generating embeddings'):
                    # Move the batch to the device
                    tokens = batch.to(self.device)

                    # Run the model inference step
                    outputs = self.model(
                        tokens,
                        repr_layers=[self.config.embedding_layer],
                        need_head_weights=return_attention_maps,
                    )

                    # Get the attention mask (shape: (B, T))
                    attention_mask = ~tokens.eq(self.model.padding_idx)

                    # Get the sequence lengths - 1 for the BOS token
                    seq_lengths = attention_mask.sum(axis=1) - 1

                    # Move the logits and last hidden states to CPU
                    logits = outputs['logits'].cpu().detach().numpy()
                    # Extract (hf) optional model outputs
                    if return_embeddings:
                        # Get the hidden states
                        embedding = (
                            outputs['representations'][self.config.embedding_layer]
                            .cpu()
                            .detach()
                            .numpy()
                        )
                    else:
                        embedding = None

                    if return_attention_maps:
                        # Get the attention maps and move them to CPU
                        attention_maps = outputs['attentions'].cpu().detach().numpy()
                    else:
                        attention_maps = None

                    # Create the output objects
                    for i, seq_len in enumerate(seq_lengths):
                        seq_input_ids = None
                        seq_logits = None
                        seq_embedding = None
                        seq_attention_maps = None

                        # Remove the BOS/EOS token and the padding
                        if return_input_ids:
                            seq_input_ids = tokens[i, 1:seq_len].cpu().detach().numpy()
                        if return_logits:
                            seq_logits = logits[i, 1:seq_len, :]
                        if return_embeddings:
                            seq_embedding = embedding[i, 1:seq_len, :]
                        if return_attention_maps:
                            # Shape is (B, L, H, T, T)
                            seq_attention_maps = attention_maps[:, :, i, 1:seq_len, :]

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


calm_models = {CaLMConfig: CaLM}
