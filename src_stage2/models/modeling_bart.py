# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch BART model. """
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from transformers.models.bart.modeling_bart import *
from transformers import Swinv2Config
from transformers.utils import ModelOutput
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
import torch


from models.modeling_swin import Swinv2ModelwithType
from models.modeling_bart_func import forward_cross_attn


@dataclass
class SwapModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    swap_last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    inspect_logits: torch.FloatTensor = None
    prototype_proba: torch.FloatTensor = None


@dataclass
class SwapSeq2SeqModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    swap_last_hidden_state: torch.FloatStorage = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    inspect_logits: torch.FloatTensor = None


def shift_tokens_right(
    input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int
):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class VisionEncoder(nn.Module):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        vision_config = Swinv2Config.from_pretrained(
            self.config.stage1_model_name_or_path
        )
        vision_config.num_observation = self.config.num_observation
        vision_config.num_view = self.config.num_view
        vision_config.pretrained_visual_extractor = (
            self.config.pretrained_visual_extractor
        )
        self.vision_encoder = Swinv2ModelwithType.from_pretrained(
            self.config.stage1_model_name_or_path
        )
        self.lesion_encoder = Swinv2ModelwithType.from_pretrained(
            self.config.stage1_model_name_or_path
        )
        self.embed_image_types = nn.Embedding(
            self.config.num_observation
            + 2,  # 0 for prior, 1 for current, 2-14 for lesion
            vision_config.embed_dim,
            padding_idx=self.config.num_observation + 1,
        )
        mid_dim = config.d_model  # + vision_config.hidden_size
        self.prior_projection = nn.Sequential(
            nn.Linear(vision_config.hidden_size, mid_dim),
            ACT2FN[config.activation_function],
            nn.Linear(mid_dim, config.d_model),
        )
        self.current_projection = nn.Sequential(
            nn.Linear(vision_config.hidden_size, mid_dim),
            ACT2FN[config.activation_function],
            nn.Linear(mid_dim, config.d_model),
        )
        self.lesion_projection = nn.Sequential(
            nn.Linear(vision_config.hidden_size, mid_dim),
            ACT2FN[config.activation_function],
            nn.Linear(mid_dim, config.d_model),
        )
        self.inspect_head = nn.ModuleList(
            nn.Sequential(
                # prior, current, and lesion
                nn.Linear(vision_config.hidden_size * 3, vision_config.hidden_size),
                ACT2FN[config.activation_function],
                nn.Linear(vision_config.hidden_size, self.config.prototype_stat[j]),
            )
            for j in range(self.config.num_observation - 1)
        )

    def encode(
        self,
        input_pixels,
        image_mask,
        gather_index=None,
        encoder=None,
        image_types=None,
        projection=None,
    ):
        outputs = encoder(pixel_values=input_pixels, image_types=image_types)
        last_hidden_state, pooler_output = (
            outputs.last_hidden_state,
            outputs.pooler_output,
        )
        one_hot_index = None
        if gather_index is not None:
            one_hot_index = (
                torch.nn.functional.one_hot(
                    gather_index, num_classes=input_pixels.size(0)
                ).float()
                * image_mask.unsqueeze(-1).float()
            )

        last_hidden_state = projection(last_hidden_state)
        image_attention_mask = None

        if one_hot_index is not None and image_mask is not None:
            last_hidden_state = (
                torch.einsum("bij,jkl->bikl", one_hot_index, last_hidden_state)
                * image_mask.unsqueeze(-1).unsqueeze(-1).float()
            )
            pooler_output = (
                torch.einsum("bij,jk->bik", one_hot_index, pooler_output)
                * image_mask.unsqueeze(-1).float()
            )
            norm = image_mask.sum(dim=1, keepdim=True)
            safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            pooler_output = pooler_output.sum(dim=1) / safe_norm
            image_attention_mask = (
                torch.ones_like(last_hidden_state[:, :, :, 0])
                * image_mask.unsqueeze(-1).float()
            )
            image_attention_mask = image_attention_mask.view(
                last_hidden_state.size(0), -1
            )
            last_hidden_state = last_hidden_state.view(
                last_hidden_state.size(0), -1, last_hidden_state.size(-1)
            )
        else:
            image_attention_mask = torch.ones_like(last_hidden_state[:, :, 0])
        return last_hidden_state, pooler_output, image_attention_mask

    def forward(
        self,
        input_pixels=None,
        image_mask=None,
        gather_index=None,
        prior_input_pixels=None,
        prior_image_mask=None,
        prior_gather_index=None,
        patch_input_pixels=None,
        patch_image_mask=None,
        patch_gather_index=None,
        patch_context_index=None,
        patch_labels=None,
    ):
        encoder_hidden_states = []
        encoder_attention_mask = []
        image_type_ids = torch.ones_like(input_pixels[:, 0, 0, 0]).long()
        flatten_last_hidden_state, pooler_output, image_attention_mask = self.encode(
            input_pixels,
            image_mask,
            gather_index,
            encoder=self.vision_encoder,
            projection=self.current_projection,
            image_types=self.embed_image_types(image_type_ids).unsqueeze(1),
        )
        encoder_hidden_states.append(flatten_last_hidden_state)
        encoder_attention_mask.append(image_attention_mask)
        prior_pooler_output = None
        if (
            prior_input_pixels is not None
            and prior_image_mask is not None
            and prior_image_mask.sum() > 0
        ):
            image_type_ids = torch.zeros_like(prior_input_pixels[:, 0, 0, 0]).long()
            (
                flatten_prior_last_hidden_state,
                prior_pooler_output,
                prior_image_attention_mask,
            ) = self.encode(
                prior_input_pixels,
                prior_image_mask,
                prior_gather_index,
                encoder=self.vision_encoder,
                projection=self.prior_projection,
                image_types=self.embed_image_types(image_type_ids).unsqueeze(1),
            )
            encoder_hidden_states.append(flatten_prior_last_hidden_state)
            encoder_attention_mask.append(prior_image_attention_mask)

        lesion_hidden_states = None
        lesion_inspect_logits = None
        lesion_attention_mask = None
        if patch_image_mask is not None and patch_image_mask.sum() > 0:
            image_type_ids = patch_labels.long() + 2
            (
                lesion_hidden_states,
                lesion_pooler_output,
                lesion_attention_mask,
            ) = self.encode(
                patch_input_pixels,
                image_mask=None,
                gather_index=None,
                encoder=self.lesion_encoder,
                projection=self.lesion_projection,
                image_types=self.embed_image_types(image_type_ids),
            )
            if patch_labels is not None and self.training:
                if prior_pooler_output is None:
                    prior_pooler_output = torch.zeros_like(pooler_output)
                one_hot_index = torch.nn.functional.one_hot(
                    patch_context_index, num_classes=pooler_output.size(0)
                ).float()
                pooler_output = torch.einsum("bi,ik->bk", one_hot_index, pooler_output)
                if prior_pooler_output is None:
                    prior_pooler_output = torch.zeros_like(pooler_output)
                else:
                    prior_pooler_output = torch.einsum(
                        "bi,ik->bk", one_hot_index, prior_pooler_output
                    )
                lesion_pooler_output = torch.cat(
                    (prior_pooler_output, pooler_output, lesion_pooler_output), dim=-1
                )
                lesion_inspect_logits = [
                    self.inspect_head[i](lesion_pooler_output[pos])
                    for pos, i in enumerate(patch_labels.squeeze())
                ]
                lesion_inspect_logits = torch.nn.utils.rnn.pad_sequence(
                    lesion_inspect_logits, batch_first=True, padding_value=-100
                )
        return (
            encoder_hidden_states,
            encoder_attention_mask,
            lesion_hidden_states,
            lesion_inspect_logits,
            lesion_attention_mask,
        )


class ViTBartEncoder(BartEncoder):
    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config, embed_tokens)
        self.vision_encoder = VisionEncoder(config, embed_tokens)
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        self.gradient_checkpointing = False
        self.init_cross_attn()

    def init_cross_attn(self):
        import types

        for layer in self.layers:
            layer.cross_attn = BartAttention(
                embed_dim=layer.embed_dim,
                num_heads=self.config.encoder_attention_heads,
                dropout=self.config.attention_dropout,
                config=self.config,
            )
            layer.cross_attn_layer_norm = nn.LayerNorm(layer.embed_dim)
            layer.forward = types.MethodType(forward_cross_attn, layer)

    def forward(
        self,
        input_ids=None,
        input_pixels=None,
        attention_mask=None,
        image_mask=None,
        gather_index=None,
        prior_input_pixels=None,
        prior_image_mask=None,
        prior_gather_index=None,
        patch_input_pixels=None,
        patch_image_mask=None,
        patch_gather_index=None,
        patch_context_index=None,
        patch_input_ids=None,
        patch_attention_mask=None,
        patch_labels=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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

        (
            input_embeds_list,
            attention_mask_list,
            lesion_embeds,
            lesion_inspect_logits,
            lesion_attention_mask,
        ) = self.vision_encoder(
            input_pixels=input_pixels,
            image_mask=image_mask,
            gather_index=gather_index,
            prior_input_pixels=prior_input_pixels,
            prior_image_mask=prior_image_mask,
            prior_gather_index=prior_gather_index,
            patch_input_pixels=patch_input_pixels,
            patch_image_mask=patch_image_mask,
            patch_gather_index=patch_gather_index,
            patch_context_index=patch_context_index,
            patch_labels=patch_labels,
        )

        if len(input_embeds_list) == 1:
            encoder_hidden_states = input_embeds_list[0]
            encoder_attention_mask = attention_mask_list[0]
        else:
            encoder_hidden_states = torch.cat(input_embeds_list, dim=1)
            encoder_attention_mask = torch.cat(attention_mask_list, dim=1)

        if lesion_embeds is not None:
            lesion_hidden_states = self.forward_with_cross_attn(
                input_ids=patch_input_ids,
                attention_mask=patch_attention_mask,
                encoder_hidden_states=lesion_embeds,
                encoder_attention_mask=lesion_attention_mask,
            )[0]
            lesion_hidden_states = torch.cat(
                (lesion_embeds, lesion_hidden_states), dim=1
            )
            lesion_attention_mask = torch.cat(
                (lesion_attention_mask, patch_attention_mask), dim=1
            )
            one_hot_index = (
                torch.nn.functional.one_hot(
                    patch_gather_index, num_classes=lesion_hidden_states.size(0)
                ).float()
                * patch_image_mask.unsqueeze(-1).float()
            )

            lesion_hidden_states = (
                torch.einsum("bij,jkl->bikl", one_hot_index, lesion_hidden_states)
                * patch_image_mask.unsqueeze(-1).unsqueeze(-1).float()
            )
            lesion_attention_mask = (
                torch.einsum("bij,jk->bik", one_hot_index, lesion_attention_mask)
                * patch_image_mask.unsqueeze(-1).float()
            )
            lesion_attention_mask = lesion_attention_mask.view(
                lesion_hidden_states.size(0), -1
            )
            lesion_hidden_states = lesion_hidden_states.view(
                lesion_hidden_states.size(0), -1, lesion_hidden_states.size(-1)
            )
            encoder_hidden_states = torch.cat(
                (encoder_hidden_states, lesion_hidden_states), dim=1
            )
            encoder_attention_mask = torch.cat(
                (encoder_attention_mask, lesion_attention_mask), dim=1
            )

        return SwapModelOutput(
            last_hidden_state=encoder_hidden_states,
            attentions=encoder_attention_mask,
            inspect_logits=lesion_inspect_logits,
        )

    def forward_with_cross_attn(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(
            hidden_states, p=self.dropout, training=self.training
        )
        # expand attention_mask
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(
                attention_mask,
                inputs_embeds.dtype,
            )
        if encoder_attention_mask is not None:
            encoder_attention_mask = _prepare_4d_attention_mask(
                encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_ids.shape[-1]
            )
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )[0]
        return BaseModelOutput(last_hidden_state=hidden_states)


class ViTBartModel(BartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = ViTBartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_pixels=None,
        image_mask=None,
        gather_index=None,
        prior_input_pixels=None,
        prior_image_mask=None,
        prior_gather_index=None,
        patch_input_ids=None,
        patch_attention_mask=None,
        patch_input_pixels=None,
        patch_image_mask=None,
        patch_gather_index=None,
        patch_context_index=None,
        patch_labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        inspect_logits = None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_pixels=input_pixels,
                image_mask=image_mask,
                gather_index=gather_index,
                prior_input_pixels=prior_input_pixels,
                prior_image_mask=prior_image_mask,
                prior_gather_index=prior_gather_index,
                patch_input_ids=patch_input_ids,
                patch_attention_mask=patch_attention_mask,
                patch_input_pixels=patch_input_pixels,
                patch_image_mask=patch_image_mask,
                patch_gather_index=patch_gather_index,
                patch_context_index=patch_context_index,
                patch_labels=patch_labels,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            inspect_logits = encoder_outputs.inspect_logits
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=encoder_outputs.attentions,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if not return_dict:
            return decoder_outputs + encoder_outputs

        return SwapSeq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            inspect_logits=inspect_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class ViTBartForConditionalGeneration(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = ViTBartModel(config)
        self.register_buffer(
            "final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings))
        )
        self.lm_head = nn.Linear(
            config.d_model, self.model.shared.num_embeddings, bias=False
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_pixels=None,
        image_mask=None,
        gather_index=None,
        prior_input_pixels=None,
        prior_image_mask=None,
        prior_gather_index=None,
        patch_input_ids=None,
        patch_attention_mask=None,
        patch_input_pixels=None,
        patch_image_mask=None,
        patch_gather_index=None,
        patch_context_index=None,
        patch_labels=None,
        swap_patch_labels=None,
        patch_factor=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            input_pixels=input_pixels,
            image_mask=image_mask,
            gather_index=gather_index,
            prior_input_pixels=prior_input_pixels,
            prior_image_mask=prior_image_mask,
            prior_gather_index=prior_gather_index,
            patch_input_ids=patch_input_ids,
            patch_attention_mask=patch_attention_mask,
            patch_input_pixels=patch_input_pixels,
            patch_image_mask=patch_image_mask,
            patch_gather_index=patch_gather_index,
            patch_context_index=patch_context_index,
            patch_labels=patch_labels[:, :1] if patch_labels is not None else None,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(
                lm_logits.view(-1, self.config.vocab_size), labels.view(-1)
            )
            if patch_labels is not None:
                observation = patch_labels[:, 0]
                patch_labels = patch_labels[:, 1:]
            if (
                outputs.inspect_logits is not None
                and (patch_labels != -100).float().sum() > 0
            ):
                raw_inspect_loss = self.weighted_bce(
                    outputs.inspect_logits,
                    patch_labels,
                    observation,
                    reduce=False,
                )
                mix_inspect_loss = self.weighted_bce(
                    outputs.inspect_logits,
                    swap_patch_labels,
                    observation,
                    reduce=True,
                )
                patch_factor = patch_factor.unsqueeze(-1)
                inspect_loss = (
                    patch_factor * raw_inspect_loss
                    + (1.0 - patch_factor) * mix_inspect_loss
                )
                num_elem = inspect_loss.size(0) * inspect_loss.size(1)
                inspect_loss = inspect_loss.sum() / num_elem

                lm_loss = lm_loss + inspect_loss

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def weighted_bce(self, logits, labels, observation, reduce=True):
        weight = (labels != -100).float()
        plug_weight = torch.nn.utils.rnn.pad_sequence(
            [
                torch.FloatTensor(self.config.attribute_weight[o.item()])
                for o in observation
            ],
            batch_first=True,
            padding_value=0,
        ).to(weight.device)
        min_size = min(weight.size(1), plug_weight.size(1))
        weight = weight[:, :min_size]
        plug_weight = plug_weight[:, :min_size]
        labels = labels[:, :min_size]
        weight = plug_weight * (labels == 1).float() + weight
        reduction = "mean" if reduce else "none"
        loss_fct = nn.BCEWithLogitsLoss(weight=weight, reduction=reduction)
        inspect_labels = labels.masked_fill(labels == -100, 0).float()
        loss = loss_fct(logits, inspect_labels)
        return loss

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        image_mask=None,
        gather_index=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "image_mask": image_mask,
            "gather_index": gather_index,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    @staticmethod
    def _expand_inputs_for_generation(
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0])
            .view(-1, 1)
            .repeat(1, expand_size)
            .view(-1)
            .to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(
                0, expanded_return_idx
            )

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError(
                    "If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined."
                )
            encoder_outputs["last_hidden_state"] = (
                encoder_outputs.last_hidden_state.index_select(
                    0,
                    expanded_return_idx.to(encoder_outputs.last_hidden_state.device),
                )
            )
            encoder_outputs["attentions"] = encoder_outputs.attentions.index_select(
                0,
                expanded_return_idx.to(encoder_outputs.attentions.device),
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
