"""Full sequence transform for transformer models."""

from __future__ import annotations

from harness.api.modeling import SequenceModelOutput
from harness.api.modeling import Transform


class FullSequence(Transform):
    """Return dense representation of the hidden states of a transformer model."""

    name = 'full_sequence'
    resolution: str = 'token'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Return the dense embeddings for the full sequence.

        Parameters
        ----------
        input : torch.Tensor
            The hidden states to pool (B, SeqLen, HiddenDim).

        Returns
        -------
        list[torch.Tensor]
            The pooled embeddings (B, SeqLen, HiddenDim).
        """
        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Return the dense embeddings for the full sequence.

        Parameters
        ----------
        input : SequenceModelOutput
            The hidden states to pool (B, SeqLen, HiddenDim).

        Returns
        -------
        SequenceModelOutput
            The pooled embeddings (B, SeqLen, HiddenDim).
        """
        return model_output
