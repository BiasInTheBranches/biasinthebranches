"""Average pool the hidden states of a transformer model."""

from __future__ import annotations

from harness.api.modeling import SequenceModelOutput
from harness.api.modeling import Transform


class AveragePool(Transform):
    """Average pool the hidden states of a transformer model."""

    name: str = 'average_pool'
    resolution: str = 'sequence'

    @staticmethod
    def apply(inputs: list[SequenceModelOutput], **kwargs) -> list[SequenceModelOutput]:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        input : torch.Tensor
            The hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        torch.Tensor
            The pooled embeddings (B, HiddenDim).
        """
        for model_out in inputs:
            model_out.embedding = model_out.embedding.mean(axis=0)

        return inputs

    @staticmethod
    def apply_h5(model_output: SequenceModelOutput, **kwargs) -> SequenceModelOutput:
        """Average pool the hidden states using the attention mask.

        Parameters
        ----------
        input : SequenceModelOutput
            The hidden states to pool (B, SeqLen, HiddenDim).
        attention_mask : torch.Tensor
            The attention mask for the hidden states (B, SeqLen).

        Returns
        -------
        SequenceModelOutput
            The pooled embeddings (B, HiddenDim).
        """
        model_output.embedding = model_output.embedding.mean(axis=0)

        return model_output
