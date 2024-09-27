from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, Swinv2Model, SwinModel, ViTModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.swinv2.modeling_swinv2 import (
    Swinv2ModelOutput,
    Swinv2Embeddings,
    Swinv2Encoder,
)
from transformers import CLIPVisionModel
from dataclasses import dataclass
from transformers import ViTConfig, SwinConfig
from typing import Tuple


@dataclass
class VisualOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_loss: Optional[torch.FloatTensor] = None
    progression_loss: Optional[torch.FloatTensor] = None
    observation_det_logits: torch.FloatTensor = None
    observation_cls_logits: torch.FloatTensor = None
    progression_logits: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


class Swinv2ModelwithType(Swinv2Model):
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_types: Optional[torch.FloatTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, len(self.config.depths))

        embedding_output, input_dimensions = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos
        )

        if image_types is not None:
            embedding_output = embedding_output + image_types

        encoder_outputs = self.encoder(
            embedding_output,
            input_dimensions,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.transpose(1, 2))
            pooled_output = torch.flatten(pooled_output, 1)

        if not return_dict:
            output = (sequence_output, pooled_output) + encoder_outputs[1:]

            return output

        return Swinv2ModelOutput(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            reshaped_hidden_states=encoder_outputs.reshaped_hidden_states,
        )
