import torch
import torch.nn as nn
from typing import Optional, Tuple


def _expand_mask(
    mask: torch.Tensor,
    dtype: torch.dtype,
    tgt_len: Optional[int] = None,
    context_mask=None,
):
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, :, None].expand(bsz, 1, src_len, tgt_len).to(dtype)
    if context_mask is not None:
        expanded_mask = expanded_mask * context_mask[:, None, None, :].to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def forward_cross_attn(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    encoder_hidden_states: Optional[torch.FloatTensor] = None,
    encoder_attention_mask: Optional[torch.FloatTensor] = None,
    layer_head_mask: torch.FloatTensor = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
    residual = hidden_states
    hidden_states, attn_weights, _ = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)

    if encoder_hidden_states is not None:
        residual = hidden_states
        hidden_states, _, _ = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            key_value_states=encoder_hidden_states,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

    residual = hidden_states
    hidden_states = self.activation_fn(self.fc1(hidden_states))
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.activation_dropout, training=self.training
    )
    hidden_states = self.fc2(hidden_states)
    hidden_states = nn.functional.dropout(
        hidden_states, p=self.dropout, training=self.training
    )
    hidden_states = residual + hidden_states
    hidden_states = self.final_layer_norm(hidden_states)

    if hidden_states.dtype == torch.float16 and (
        torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
    ):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (attn_weights,)

    return outputs
