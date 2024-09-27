import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
import copy
from tokenizer import Tokenizer
from transformers import BartTokenizer


def process_examples(
    examples,
    max_tgt_length,
    tokenizer: BartTokenizer,
    word_tokenizer,
):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    labels = []
    idxs = []
    image_paths = []
    id2labels = {}
    for index in progress:
        report_id = examples["id"][index]
        image_path = examples["image_path"][index]
        report = word_tokenizer.decode(
            word_tokenizer.encode(examples["report"][index], add_special_tokens=False)[
                :max_tgt_length
            ],
            skip_special_tokens=False,
        )
        report = report.replace(" .", ".")
        report = tokenizer(
            report + tokenizer.eos_token,
            truncation=True,
            max_length=max_tgt_length * 2,
            add_special_tokens=False,
        )
        labels.append(report["input_ids"])
        idxs.append(report_id)
        image_paths.append(image_path)
        id2labels[report_id] = labels[-1]
    return (
        idxs,
        image_paths,
        labels,
        id2labels,
    )
