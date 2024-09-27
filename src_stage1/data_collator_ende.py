#!/usr/bin/env python
# coding=utf-8
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from transformers import DataCollatorForSeq2Seq
import itertools

from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase


@dataclass
class DataCollatorForEnDeStage1(DataCollatorForSeq2Seq):
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        report_ids = (
            [feature["report_ids"] for feature in features]
            if "report_ids" in features[0].keys()
            else None
        )
        observations = (
            [feature["observations"] for feature in features]
            if "observations" in features[0].keys()
            else None
        )
        normals = (
            [feature["normals"] for feature in features]
            if "normals" in features[0].keys()
            else None
        )
        input_pixels = (
            [feature["input_pixels"] for feature in features]
            if "input_pixels" in features[0].keys()
            else None
        )
        batch_outputs = {}

        if observations is not None:
            batch_outputs["observations"] = []
            for feature in features:
                batch_outputs["observations"].append(feature["observations"])

        if normals is not None:
            batch_outputs["normals"] = []
            for feature in features:
                batch_outputs["normals"].append(feature["normals"])

        features = BatchEncoding(batch_outputs, tensor_type=return_tensors)
        if input_pixels is not None:
            (
                input_pixels,
                image_mask,
                gather_index,
            ) = DataCollatorForEnDeStage1.load_images(
                input_pixels,
            )
            features["input_pixels"] = input_pixels
            features["image_mask"] = image_mask
            features["gather_index"] = gather_index

        if report_ids is not None:
            features["report_ids"] = report_ids
        return features

    @staticmethod
    def load_images(input_pixels):
        max_img_num = max([len(img) for img in input_pixels])
        image_mask = []
        gather_index = []
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
            gather_index.append(gather_idx)
        return (
            torch.cat(input_pixels, dim=0),
            torch.LongTensor(image_mask),
            torch.LongTensor(gather_index),
        )
