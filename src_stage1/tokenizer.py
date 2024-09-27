import json
import re
from collections import Counter, defaultdict

import pandas as pd
import os
import pickle
import math


class Tokenizer:
    def __init__(self, config):
        self.model_input_names = ["nodes"]
        self.padding_side = "right"
        self.ann_path = config.annotation_file
        self.threshold = config.threshold
        self.dataset = config.dataset
        if self.dataset == "iu_xray":
            self.clean_report = Tokenizer.clean_report_iu_xray
        else:
            self.clean_report = Tokenizer.clean_report_mimic_cxr
        print(self.clean_report)
        self.ann = json.loads(open(self.ann_path, "r").read())
        (
            self.token2idx,
            self.idx2token,
            self.special_tokens,
            self.total_tokens,
        ) = self.create_vocabulary()
        self.bos_token_id = self.eos_token_id = self.decoder_start_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2

    def create_vocabulary(self):
        total_tokens = []
        total_tokens_ = []
        for example in self.ann["train"]:
            tokens = self.clean_report(example["report"]).split()
            total_tokens_.extend(tokens)
            flag = True
            for token in tokens:
                token = token.strip()
                if len(token) == 0:
                    continue
                if flag or token == ".":
                    total_tokens.append(token)
                    flag = False
                    continue
                total_tokens.append(" " + token)
        total_tokens = set(total_tokens)
        counter = Counter(total_tokens_)
        vocab = [k for k, v in counter.items() if v >= self.threshold]
        vocab.sort()
        special_tokens = ["<unk>"]  # custom for bart
        vocab = special_tokens + vocab
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx
            idx2token[idx] = token
        total_tokens = {z for z in total_tokens if z.strip() in token2idx}
        # total_tokens.add("[UNK]")
        return token2idx, idx2token, special_tokens[:-1], total_tokens

    @staticmethod
    def clean_report_iu_xray(report):
        def report_cleaner(t):
            return (
                t.replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .strip()
                .lower()
                .split(". ")
            )

        def sent_cleaner(t):
            return re.sub(
                "[.,?;*!%^&_+():-\[\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .strip()
                .lower(),
            )

        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def clean_report_mimic_cxr(report):
        def report_cleaner(t):
            return (
                t.replace("\n", " ")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("__", "_")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("  ", " ")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("..", ".")
                .replace("1. ", "")
                .replace(". 2. ", ". ")
                .replace(". 3. ", ". ")
                .replace(". 4. ", ". ")
                .replace(". 5. ", ". ")
                .replace(" 2. ", ". ")
                .replace(" 3. ", ". ")
                .replace(" 4. ", ". ")
                .replace(" 5. ", ". ")
                .strip()
                .lower()
                .split(". ")
            )

        def sent_cleaner(t):
            return re.sub(
                "[.,?;*!%^&_+():-\[\]{}]",
                "",
                t.replace('"', "")
                .replace("/", "")
                .replace("\\", "")
                .replace("'", "")
                .lower()
                .strip(),
            )

        tokens = [
            sent_cleaner(sent).strip() + " ."
            for sent in report_cleaner(report)
            if len(sent_cleaner(sent).strip()) > 0
        ]
        report = " ".join(tokens)
        return report

    @staticmethod
    def load_tag2ids(
        tag_path,
        train_idxs=None,
        need_header=False,
        flip=False,
    ):
        cached_path = tag_path + ".pkl"
        if os.path.exists(cached_path):
            with open(cached_path, "rb") as f:
                tags = pickle.load(f)
        else:
            tags = pd.read_csv(tag_path)
            with open(cached_path, "wb") as f:
                pickle.dump(tags, file=f)
        # tags = tags.replace(-1, 1).fillna(2)
        tags = tags.fillna(2)
        diseases = list(tags)[2:]
        id2tags = defaultdict(list)
        weight = [0] * len(diseases)
        count = 0

        def cal_val(v, flip):
            if flip:
                return 1 if v == 0 else 0
            else:
                return 1 if v == -1 or v == 1 else 0

        for i in range(len(tags)):
            tag = tags.iloc[i]
            idx = tag[1]
            id2tags[idx] = list(tag[2:].values)
            if train_idxs is not None and idx in train_idxs:
                weight = [w + cal_val(v, flip) for w, v in zip(weight, id2tags[idx])]
                count += 1
        if train_idxs is not None:
            weight = [math.log(max((count - w) / max(1, w), 1)) for w in weight]

        if not need_header:
            return id2tags, weight
        else:
            return id2tags, diseases, weight

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            # return self.token2idx["[UNK]"]
            return self.token2idx["<unk>"]
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def encode(
        self,
        report,
        add_special_tokens=True,
    ):
        ids = []
        tokens = self.clean_report(report).split()
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        if add_special_tokens:
            ids = [self.decoder_start_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, ids, skip_special_tokens=True, separator=" "):
        txt = []
        for i, idx in enumerate(ids):
            if idx not in self.idx2token:
                idx = self.unk_token_id
            token = self.idx2token[idx]
            if skip_special_tokens and token in self.special_tokens:
                continue
            txt.append(token)
        return separator.join(txt)

    def batch_decode(self, ids_batch, skip_special_tokens=True, separator=" "):
        out = []
        for ids in ids_batch:
            out.append(
                self.decode(
                    ids,
                    skip_special_tokens=skip_special_tokens,
                    separator=separator,
                )
            )
        return out

    def save_pretrained(self, save_directory):
        return ""
