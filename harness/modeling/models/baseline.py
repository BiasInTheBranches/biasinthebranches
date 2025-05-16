"""Simplified baseline sequence encoders (One-Hot and N-gram) implementations."""

from __future__ import annotations

from itertools import product
from typing import Any
from typing import Literal

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder
from tqdm import tqdm

from harness.api.logging import logger
from harness.api.modeling import HDF5CachedList
from harness.api.modeling import LM
from harness.api.modeling import LMConfig
from harness.api.modeling import SequenceModelOutput

CODONS = [''.join(codon) for codon in product('ACGT', repeat=3)]
AMINO_ACIDS = list('ACDEFGHIKLMNPQRSTVWY')


class OneHotEncoderConfig(LMConfig):
    """Configuration for baseline sequence encoders."""

    name: Literal['OneHotEncoder'] = 'OneHotEncoder'

    # Whether to evaluate DNA sequences or Amino acids
    evaluate_dna: bool = False


class OneHotEncoder(LM):
    """One-hot encoding for biological sequences using scikit-learn."""

    def __init__(self, config: OneHotEncoderConfig) -> None:
        """Initialize the One-Hot encoder."""
        self.config = config

        # Set token model input and encoding:
        if config.evaluate_dna:
            self.model_input = 'dna'
            self.model_encoding = 'char'
            self.vocabulary = CODONS
        else:  # Evaluating amino acids
            self.model_input = 'aminoacid'
            self.model_encoding = 'char'
            self.vocabulary = AMINO_ACIDS

        logger.info(f'Using vocabulary: {self.vocabulary}')

        # Initialize scikit-learn's OneHotEncoder
        self.encoder = SklearnOneHotEncoder(
            sparse_output=False,
            categories=[self.vocabulary],
            handle_unknown='ignore',  # Ignore characters not in vocabulary
        )

        # Fit the encoder with the vocabulary
        self.encoder.fit([[char] for char in self.vocabulary])

    @property
    def tokenizer(self) -> Any:
        """Simple pass-through tokenizer."""
        return self

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        return {}

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        return {}

    @property
    def device(self) -> torch.device:
        """Get the device (CPU for this baseline model)."""
        return torch.device('cpu')

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = True,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate one-hot encoded embeddings for each sequence."""
        # Initialize outputs list if not provided
        if model_outputs is None:
            model_outputs = []

        # Process each sequence
        for seq in tqdm(sequences, desc='One-hot encoding sequences'):
            # Tokenize the sequence based on configuration
            if self.config.evaluate_dna:
                # For DNA, consider codons (3-letter groups)
                tokens = [
                    seq[i : i + 3] for i in range(0, len(seq), 3) if i + 3 <= len(seq)
                ]
                # Handle if there are remaining characters
                if len(seq) % 3 != 0:
                    logger.warning(
                        f'Sequence length {len(seq)} is not divisible by 3. '
                        f'Last {len(seq) % 3} characters ignored.'
                    )
            else:
                # For amino acids, consider each character
                tokens = list(seq)

            # Create input_ids (indices into vocabulary)
            if return_input_ids:
                seq_input_ids = np.array(
                    [
                        self.vocabulary.index(token) if token in self.vocabulary else -1
                        for token in tokens
                    ]
                )

                # Handle unknown tokens in input_ids
                # Set to the last index (just beyond vocabulary size)
                unknown_mask = seq_input_ids == -1
                if unknown_mask.any():
                    logger.warning(
                        f'Found {unknown_mask.sum()} unknown tokens in sequence'
                    )
                    seq_input_ids[unknown_mask] = len(self.vocabulary)
            else:
                seq_input_ids = None

            # One-hot encode the sequence if embeddings are requested
            if return_embeddings:
                # Prepare tokens for the encoder - each token needs to be separate
                tokens_for_encoder = [[token] for token in tokens]
                # Transform the tokens into a one-hot encoded array
                seq_embedding = self.encoder.transform(tokens_for_encoder)
            else:
                seq_embedding = None

            # Create SequenceModelOutput object
            output_fields = {
                'sequence': seq,
                'input_ids': seq_input_ids if return_input_ids else None,
                'embedding': seq_embedding if return_embeddings else None,
                'logits': None,
                'attention_maps': None,
            }

            model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(
        self, input: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate sequences (not applicable for encoders)."""
        raise NotImplementedError('Sequence generation not supported for encoders')


class NGramEncoderConfig(LMConfig):
    """Configuration for baseline sequence encoders."""

    name: Literal['NGramEncoder'] = 'NGramEncoder'

    # Whether to evaluate DNA sequences or Amino acids
    evaluate_dna: bool = False

    # N-gram range (min_n, max_n)
    ngram_range: tuple[int, int] = (3, 3)
    # Maximum number of features to keep
    max_features: int | None = 128
    # Whether to use binary counts
    binary: bool = False


class NGramEncoder(LM):
    """N-gram encoding for biological sequences."""

    def __init__(self, config: NGramEncoderConfig) -> None:
        """Initialize the N-gram encoder."""
        self.config = config

        # Set token model input and encoding:
        if config.evaluate_dna:
            self.model_input = 'dna'
            self.model_encoding = 'char'
        else:  # Evaluating amino acids
            self.model_input = 'aminoacid'
            self.model_encoding = 'char'

    @property
    def tokenizer(self) -> Any:
        """Simple pass-through tokenizer."""
        return self

    @property
    def tokenizer_config(self) -> dict[str, Any]:
        """Get the tokenizer configuration."""
        return {}

    @property
    def dataloader_config(self) -> dict[str, Any]:
        """Get the dataloader configuration."""
        return {}

    @property
    def device(self) -> torch.device:
        """Get the device (CPU for this baseline model)."""
        return torch.device('cpu')

    def generate_model_outputs(
        self,
        sequences: list[str],
        model_outputs: HDF5CachedList | None = None,
        return_input_ids: bool = True,
        return_logits: bool = False,
        return_embeddings: bool = True,
        return_attention_maps: bool = False,
    ) -> list[SequenceModelOutput]:
        """Generate n-gram encoded representations."""
        # Create a new vectorizer for this specific set of sequences
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=self.config.ngram_range,
            max_features=self.config.max_features,
            binary=self.config.binary,
            token_pattern=r'\S+',  # Tokenize on non-whitespace characters
        )

        if self.config.evaluate_dna:
            # For DNA, consider codons
            k_mer = 3
            sequences = [
                ' '.join(seq[i : i + k_mer] for i in range(0, len(seq), k_mer))
                for seq in sequences
            ]
        else:
            # For amino acids, consider each character
            sequences = [' '.join(list(seq)) for seq in sequences]

        # Fit and transform on these sequences
        ngram_features = vectorizer.fit_transform(sequences)

        # Log vocabulary size
        logger.info(
            f'N-gram vocabulary size for this batch: {len(vectorizer.vocabulary_)}'
        )
        # Convert to dense array and add a (dummy) seq_len dim (just 1)
        dense_features = ngram_features.toarray()
        dense_features = np.expand_dims(dense_features, axis=1)

        # Initialize outputs list if not provided
        if model_outputs is None:
            model_outputs: list[SequenceModelOutput] = []

        # Create output objects for each sequence
        for i, seq in enumerate(tqdm(sequences, desc='N-gram encoding sequences')):
            # Use the same features for both input_ids and embedding
            # since for n-grams they're equivalent
            features = dense_features[i]

            output_fields = {
                'sequence': seq,
                'input_ids': features if return_input_ids else None,
                'embedding': features if return_embeddings else None,
                'logits': None,  # N-gram encoder doesn't produce logits
                'attention_maps': None,  # N-gram encoder doesn't have attention
            }

            model_outputs.append(SequenceModelOutput(**output_fields))

        return model_outputs

    def generate_sequences(
        self, input: list[str], model_outputs: HDF5CachedList | None = None
    ) -> list[SequenceModelOutput]:
        """Generate sequences (not applicable for encoders)."""
        raise NotImplementedError('Sequence generation not supported for encoders')


baseline_models = {
    OneHotEncoderConfig: OneHotEncoder,
    NGramEncoderConfig: NGramEncoder,
}
