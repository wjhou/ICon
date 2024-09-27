from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel, Swinv2Model
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from transformers import SwinConfig


@dataclass
class VisualOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    observation_loss: Optional[torch.FloatTensor] = None
    pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    observation_cls_logits: torch.FloatTensor = None
    normal_cls_logits: torch.FloatTensor = None


class VisualEncoder(PreTrainedModel):
    def __init__(self, config: SwinConfig):
        super().__init__(config)
        self.observation_cls = nn.Linear(
            self.config.hidden_size,
            self.config.num_observation,
        )
        self.normal_cls = nn.Linear(
            self.config.hidden_size, self.config.num_observation - 1
        )
        self.post_init()
        self.visual_extractor = Swinv2Model.from_pretrained(
            self.config.pretrained_visual_extractor
        )
        self.observation_weight = torch.FloatTensor(
            self.config.observation_weight
        ).unsqueeze(0)
        self.normal_weight = torch.FloatTensor(self.config.normal_weight).unsqueeze(0)

    def _init_weights(self, module):
        # std = self.config.init_std
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def encode_image(
        self,
        input_pixels,
        image_mask,
        gather_index=None,
    ):
        outputs = self.visual_extractor(
            input_pixels,
        )
        pooler_output = outputs.pooler_output
        last_hidden_state = None
        if gather_index is not None:
            pooler_output = torch.einsum("bij,jk->bik", gather_index, pooler_output)
        if image_mask is not None:
            pooler_output = pooler_output * image_mask.unsqueeze(-1).float()
            pooler_output = pooler_output.sum(dim=1) / image_mask.sum(
                dim=1, keepdim=True
            )
        return pooler_output, last_hidden_state

    def forward(
        self,
        input_pixels: torch.FloatTensor = None,
        image_mask: torch.LongTensor = None,
        gather_index: torch.LongTensor = None,
        observations: Optional[torch.FloatTensor] = None,
        normals: Optional[torch.FloatTensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        one_hot_index = None
        if gather_index is not None:
            one_hot_index = torch.nn.functional.one_hot(
                gather_index, num_classes=input_pixels.size(0)
            ).float()
        pooler_output, last_hidden_state = self.encode_image(
            input_pixels,
            image_mask,
            one_hot_index,
        )

        observation_cls_logits = self.observation_cls(pooler_output)
        normal_cls_logits = self.normal_cls(pooler_output.detach())

        loss = None
        if observations is not None:
            weight = (
                torch.ones_like(observations)
                + self.observation_weight.to(observations.device) * observations
            )
            loss_fct = nn.BCEWithLogitsLoss(weight=weight.view(-1))
            loss = loss_fct(
                observation_cls_logits.view(-1),
                observations.float().view(-1),
            )
            if normals is not None:
                mask = (normals != -100).float()
                if mask.sum() > 0:
                    weight = (
                        torch.ones_like(normals)
                        + self.normal_weight.to(normals.device) * normals
                    ) * mask
                    loss_fct = nn.BCEWithLogitsLoss(weight=weight.view(-1))
                    normal_loss = loss_fct(
                        normal_cls_logits.view(-1),
                        normals.float().view(-1),
                    )
                    loss = loss + normal_loss

        return VisualOutput(
            loss=loss,
            observation_cls_logits=observation_cls_logits,
            normal_cls_logits=normal_cls_logits,
            pooler_output=pooler_output,
            last_hidden_state=last_hidden_state,
        )
