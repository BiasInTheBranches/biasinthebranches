"""Window transformer embeddings for adapting sequence lengths."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from harness.api.modeling import SequenceModelOutput
from harness.api.modeling import Transform


class Window3(Transform):
    """Windowed embeddings to shape (num_tokens//3, hidden_dim)."""

    name: str = '3_window'
    resolution: str = '3mer'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Windowed embeddings to shape (num_tokens//3, hidden_dim).

        Parameters
        ----------
        input : list[SequenceModelOutput]
            SequenceModelOutput objects to pool embeddings over.
        window_size : int
            The size of the window to pool over, passed in as keyword argument.

        Returns
        -------
        List[SequenceModelOutput]
            Returns the input embeddings averaged over the window size in a
            SequenceModelOutput object.
        """
        window_size = kwargs.get('window_size', 3)

        for model_out in tqdm(inputs, desc='Transform'):
            # Find output length, if not divisible by window size, add one to
            # capture the remainder
            output_size = model_out.embedding.shape[0] // window_size
            if model_out.embedding.shape[0] % window_size != 0:
                output_size += 1
            windowed_emb = np.zeros((output_size, model_out.embedding.shape[1]))
            # Average over the window size
            for window_i, token_i in enumerate(
                range(0, model_out.embedding.shape[0], window_size)
            ):
                windowed_emb[window_i] = model_out.embedding[
                    token_i : token_i + window_size
                ].mean(axis=0)
            # Update the embedding
            model_out.embedding = windowed_emb

        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Windowed embeddings to shape (num_tokens//3, hidden_dim).

        Parameters
        ----------
        input : SequenceModelOutput
            SequenceModelOutput object to pool embeddings over.
        window_size : int
            The size of the window to pool over, passed in as keyword argument.

        Returns
        -------
        SequenceModelOutput
            Returns the input embeddings averaged over the window size in a
            SequenceModelOutput object.
        """
        window_size = kwargs.get('window_size', 3)

        # Find output length, if not divisible by window size, add one to
        # capture the remainder
        output_size = model_output.embedding.shape[0] // window_size
        if model_output.embedding.shape[0] % window_size != 0:
            output_size += 1
        windowed_emb = np.zeros((output_size, model_output.embedding.shape[1]))
        # Average over the window size
        for window_i, token_i in enumerate(
            range(0, model_output.embedding.shape[0], window_size)
        ):
            windowed_emb[window_i] = model_output.embedding[
                token_i : token_i + window_size
            ].mean(axis=0)
        # Update the embedding
        model_output.embedding = windowed_emb

        return model_output
