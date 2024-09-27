#!/usr/bin/env python
# coding=utf-8
from dataclasses import dataclass
from typing import Any, Optional, Union
import itertools
import torch
from transformers import DataCollatorForSeq2Seq
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
import sys


@dataclass
class DataCollatorForEnDe(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    observation_pad_id: int = 13
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        input_ids = (
            [feature["input_ids"] for feature in features]
            if "input_ids" in features[0].keys()
            else None
        )
        patch_input_ids = (
            [feature["patch_input_ids"] for feature in features]
            if "patch_input_ids" in features[0].keys()
            else None
        )
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        input_pixels = (
            [feature["input_pixels"] for feature in features]
            if "input_pixels" in features[0].keys()
            else None
        )
        prior_input_pixels = (
            [feature["prior_input_pixels"] for feature in features]
            if "prior_input_pixels" in features[0].keys()
            else None
        )
        patch_input_pixels = (
            [feature["patch_input_pixels"] for feature in features]
            if "patch_input_pixels" in features[0].keys()
            else None
        )
        patch_labels = (
            [feature["patch_labels"] for feature in features]
            if "patch_labels" in features[0].keys()
            else None
        )
        swap_patch_labels = (
            [feature["swap_patch_labels"] for feature in features]
            if "swap_patch_labels" in features[0].keys()
            else None
        )
        patch_factor = (
            [feature["patch_factor"] for feature in features]
            if "patch_factor" in features[0].keys()
            else None
        )
        report_ids = (
            [feature["report_ids"] for feature in features]
            if "report_ids" in features[0].keys()
            else None
        )

        batch_outputs = {}
        if input_ids is not None:
            max_length = max(len(l) for l in list(itertools.chain(*input_ids)))
            batch_outputs["input_ids"] = []
            for input_id_ in input_ids:
                batch_outputs["input_ids"].append(
                    self.process_input_ids(input_id_, max_length)
                )
            max_obs_length = max(len(l) for l in batch_outputs["input_ids"])
            for i in range(len(batch_outputs["input_ids"])):
                diff = max_obs_length - len(batch_outputs["input_ids"][i])
                if diff > 0:
                    remainder = [[self.tokenizer.pad_token_id] * max_length] * diff
                    batch_outputs["input_ids"][i].extend(remainder)

        if patch_input_ids is not None:
            max_length = max(max(len(z) for z in l) for l in patch_input_ids)
            batch_outputs["patch_input_ids"] = []
            for input_id_ in patch_input_ids:
                batch_outputs["patch_input_ids"].extend(
                    self.process_input_ids(input_id_, max_length)
                )

        if patch_labels is not None:
            batch_outputs["patch_labels"] = []
            patch_labels = list(itertools.chain(*patch_labels))
            max_label_length = max(len(l) for l in patch_labels)
            for patch_label in patch_labels:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(patch_label)
                )
                patch_label = patch_label + remainder
                batch_outputs["patch_labels"].append(patch_label)
            if swap_patch_labels is not None:
                batch_outputs["swap_patch_labels"] = []
                swap_patch_labels = list(itertools.chain(*swap_patch_labels))
                for swap_patch_label in swap_patch_labels:
                    remainder = [self.label_pad_token_id] * (
                        max_label_length - len(swap_patch_label)
                    )
                    swap_patch_label = swap_patch_label + remainder
                    batch_outputs["swap_patch_labels"].append(swap_patch_label)
                batch_outputs["patch_factor"] = list(itertools.chain(*patch_factor))

        if labels is not None:
            batch_outputs["labels"] = []
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if self.tokenizer.padding_side == "right":
                    feature["labels"] = feature["labels"] + remainder
                else:
                    feature["labels"] = remainder + feature["labels"]
                batch_outputs["labels"].append(feature["labels"])

        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)

        features["attention_mask"] = torch.ones_like(features["input_ids"]).masked_fill(
            features["input_ids"] == self.tokenizer.pad_token_id, 0
        )
        features["patch_attention_mask"] = torch.ones_like(
            features["patch_input_ids"]
        ).masked_fill(features["patch_input_ids"] == self.tokenizer.pad_token_id, 0)

        if input_pixels is not None:
            (
                input_pixels,
                image_mask,
                gather_index,
                _,
            ) = DataCollatorForEnDe.load_images(
                input_pixels,
            )
            features["input_pixels"] = input_pixels
            features["image_mask"] = image_mask
            features["gather_index"] = gather_index

        if prior_input_pixels is not None:
            (
                prior_input_pixels,
                prior_image_mask,
                prior_gather_index,
                _,
            ) = DataCollatorForEnDe.load_images(
                prior_input_pixels,
            )

            features["prior_input_pixels"] = prior_input_pixels
            features["prior_image_mask"] = prior_image_mask
            features["prior_gather_index"] = prior_gather_index

        if patch_input_pixels is not None:
            (
                patch_input_pixels,
                patch_image_mask,
                patch_gather_index,
                patch_context_index,
            ) = DataCollatorForEnDe.load_images(
                patch_input_pixels,
            )
            features["patch_input_pixels"] = patch_input_pixels
            features["patch_image_mask"] = patch_image_mask
            features["patch_gather_index"] = patch_gather_index
            features["patch_context_index"] = patch_context_index

        if report_ids is not None:
            features["report_ids"] = torch.LongTensor(report_ids)
            features["report_pos"] = torch.LongTensor(list(range(len(report_ids))))

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids
        return features

    def process_input_ids(self, features, max_length):
        input_ids = []
        for feature in features:
            diff = max_length - len(feature)
            remainder = [self.tokenizer.pad_token_id] * (diff)
            feature = feature + remainder
            input_ids.append(feature)
        return input_ids

    @staticmethod
    def load_images(input_pixels):
        max_img_num = max([len(img) for img in input_pixels])
        image_mask = []
        gather_index = []
        context_index = []
        count = 0
        for i in range(len(input_pixels)):
            num_img = len(input_pixels[i])
            if (input_pixels[i] != 0).float().sum() != 0:
                diff = max_img_num - num_img
                image_mask.append([1] * num_img + [0] * diff)
            else:
                image_mask.append([0] * max_img_num)
            gather_idx = [0] * max_img_num
            for j in range(num_img):
                gather_idx[j] = count
                count += 1
                context_index.append(i)

            gather_index.append(gather_idx)
        return (
            torch.cat(input_pixels, dim=0),
            torch.LongTensor(image_mask),
            torch.LongTensor(gather_index),
            torch.LongTensor(context_index),
        )
