"""Super resolution transformation for transformer embeddings."""

from __future__ import annotations

from itertools import repeat

import numpy as np
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from harness.api.logging import logger
from harness.api.modeling import SequenceModelOutput
from harness.api.modeling import Transform


class SuperResolution(Transform):
    """Up-sample the hidden states of a transformer model.

    This is to recover the original sequence length (in chars) of an input sequence
    regardless of the tokenization scheme of the model.
    """

    name: str = 'super_resolution'
    resolution: str = 'char'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Run the 'super resolution transformation on a set of embeddings.

        Parameters
        ----------
        inputs : list[SequenceModelOutput]
            Modeloutput to pool.
        sequences: str
            list of sequences, sent in as kwargs. Used to get the single char level
            representation out of a coarser input representation.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer sent in as a kwarg. Necessary to get the single char level
            representation out of a coarser input representation.

        Returns
        -------
        list[SequenceModelOutput]:
            Returns the input embeddings averaged over the window size in a
            SequenceModelOutput object.
        """
        sequences: list[str] = kwargs.get('sequences')
        tokenizer: PreTrainedTokenizerFast = kwargs.get('tokenizer')

        assert sequences is not None, 'Sequences must be provided as a kwarg'
        assert tokenizer is not None, 'Tokenizer must be provided as a kwarg'

        tokenized_seqs = [tokenizer.tokenize(seq) for seq in sequences]
        for model_input, tokenized_seq in tqdm(
            zip(inputs, tokenized_seqs, strict=False), desc='Transform'
        ):
            # Iterate over each token and take convex combination of window around token
            super_res_emb = SuperResolution.super_resolution(
                model_input.embedding, tokenized_seq
            )
            model_input.embedding = super_res_emb

        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Run the 'super resolution transformation on a set of embeddings.

        Parameters
        ----------
        model_output : SequenceModelOutput
            Modeloutput to pool.
        sequences: str
            list of sequences, sent in as kwargs. Used to get the single char level
            representation out of a coarser input representation.
        tokenizer : PreTrainedTokenizerFast
            Tokenizer sent in as a kwarg. Necessary to get the single char level
            representation out of a coarser input representation.

        Returns
        -------
        SequenceModelOutput:
            Returns the input embeddings averaged over the window size in a
            SequenceModelOutput object.
        """
        sequence: str = kwargs.get('sequences')
        tokenizer: PreTrainedTokenizerFast = kwargs.get('tokenizer')

        assert sequence is not None, 'Sequences must be provided as a kwarg'
        assert tokenizer is not None, 'Tokenizer must be provided as a kwarg'

        tokenized_seq = tokenizer.tokenize(sequence)
        super_res_emb = SuperResolution.super_resolution(
            model_output.embedding, tokenized_seq
        )
        model_output.embedding = super_res_emb

        return model_output

    @staticmethod
    def super_resolution(embedding, tokens, window_size=None):
        """On a single embedding expand to char level embedding."""
        # Determine location of each token in the sequence
        char_locations = []
        for i, token in enumerate(tokens):
            char_locations.extend(list(repeat(i, len(token))))

        # Determine the maximum token length if window_size is not provided
        # window size is the number of tokens to include on both sides of the ith token
        if window_size is None:
            window_size = max(len(token) for token in tokens) // 2 + 1

        _, hidden_size = embedding.shape
        seq_length = len(''.join(tokens))

        # Initialize the output tensor
        super_res_embedding = np.zeros((seq_length, hidden_size))

        total_window_size = window_size * 2 + 1
        warning_raised = False
        for char_loc in range(seq_length):
            # Initialize the window embedding
            window_embedding = np.zeros((total_window_size, hidden_size))

            # Fill the window embedding with the embedding of the tokens in the window
            for idx in range(total_window_size):
                # Determine the location of the residue in the sequence
                residue_location = char_loc - window_size + idx
                if residue_location < 0 or residue_location >= seq_length:
                    continue
                # Determine the location of the residue in the embeddings
                emb_idx = char_locations[residue_location]
                if emb_idx > embedding.shape[0] - 1:
                    if not warning_raised:
                        logger.warning(
                            'Embedding shorter than tokenized sequence, skipping char '
                            f'locations {residue_location}-{seq_length}'
                        )
                        warning_raised = True
                    break
                window_embedding[idx] = embedding[emb_idx, :]

            # Mean pool the windowed embedding for the char level representation
            super_res_embedding[char_loc, :] = window_embedding.mean(axis=0)

        return super_res_embedding
