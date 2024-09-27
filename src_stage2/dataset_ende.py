import os
from torch.utils.data import Dataset
import torch
from data_arguments import DataTrainingArguments
from data_process_ende import process_examples
from PIL import Image
import random


class DatasetCustom(Dataset):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        annotation,
        temporal_ids,
        sentence_annotation,
        patch_annotation,
        swap_annotation,
        processor,
        split: str,
        tokenizer,
        word_tokenizer,
        transform=None,
        observation_category=None,
        id2tags=None,
        id2tags_valid=None,
        keep_columns={
            "id",
            "report",
            "image_path",
        },
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.word_tokenizer = word_tokenizer
        self.data_args = data_args
        self.split = split
        self.dataset = data_args.dataset
        self.processor = processor
        self.observation_category = observation_category
        self.id2tags = id2tags
        self.id2tags_valid = id2tags_valid
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
        self.temporal_ids = temporal_ids
        for sample in samples:
            for key in sample:
                if key not in examples:
                    continue
                examples[key].append(sample[key])
        for key in examples:
            print(key, examples[key][:3])
        (
            idxs,
            image_paths,
            labels,
            self.id2labels,
        ) = process_examples(
            examples=examples,
            max_tgt_length=data_args.max_tgt_length,
            tokenizer=tokenizer,
            word_tokenizer=word_tokenizer,
        )
        if self.data_args.debug_model:
            idxs = idxs[:16]  # if self.split == "train" else idxs
        self.patch_annotation = patch_annotation[split]
        self.swap_annotation = (
            None if swap_annotation is None else swap_annotation[split]
        )
        self.sentence_annotation = (
            None if sentence_annotation is None else sentence_annotation[split]
        )
        self.data = [
            {
                "id": a,
                "image_paths": b,
                "labels": c,
            }
            for a, b, c in zip(
                idxs,
                image_paths,
                labels,
            )
        ]
        self.study2image_paths = {
            (
                int(self.data[i]["id"])
                if self.dataset != "iu_xray"
                else self.data[i]["id"]
            ): self.data[i]["image_paths"]
            for i in range(len(self.data))
        }
        self.all_index = list(range(len(self.data)))

        self.report2index = {
            self.data[index]["id"]: index for index in range(len(self.data))
        }
        self.index2report = {
            index: self.data[index]["id"] for index in range(len(self.data))
        }

        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, index):
        labels = self.data[index]["labels"]
        idx = self.data[index]["id"]
        patch_idx = str(idx)
        prior_image_paths = []
        image_paths = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["image_paths"]
        ]
        if self.temporal_ids is not None and patch_idx in self.temporal_ids:
            prior_image_paths = [
                os.path.join(self.data_args.image_path, a)
                for a in self.temporal_ids[patch_idx]["image_path"]
            ]
        patch_paths = []
        swap_patch_paths = []
        patch_input_ids = []
        patch_labels = []
        swap_patch_labels = []
        swap_patch_lambda = []
        input_ids = None
        topk = 30
        keep_obs = [
            obs.split("_")[0]
            for obs in self.id2tags[patch_idx]
            if "No Finding" not in obs and "POS" in obs
        ]
        all_obs = []
        sorted_obs = sorted(
            self.id2tags[patch_idx],
            key=lambda x: self.observation_category.index(x.split("_")[0]),
        )
        image_labels = [0] * len(self.observation_category)
        for obs in sorted_obs:
            if "No Finding" in obs:
                if "POS" in obs:
                    all_obs.append("<Positive No Finding>")
                else:
                    all_obs.append("<Negative No Finding>")
                continue
            obs_ = obs.split("_")
            obs_str = (
                f"<Positive {obs_[0]}>" if "POS" in obs else f"<Negative {obs_[0]}>"
            )
            if "NEG" in obs:
                all_obs.append(obs_str)

        input_ids = [
            self.tokenizer(
                o_str,
                add_special_tokens=True,
            ).input_ids
            for o_str in all_obs
        ]

        if patch_idx in self.patch_annotation:
            patches = self.patch_annotation[patch_idx]
            for obs in keep_obs:
                patch_info = patches[obs]
                img_path = os.path.join(
                    self.data_args.patch_path,
                    self.split,
                    patch_idx,
                    patch_info["patch_path"],
                )
                patch_paths.append(img_path)
                patch_label = [0] * len(self.tokenizer.attribute2id[obs])
                for e in patch_info["aligned_entities"]:
                    if e not in self.tokenizer.attribute2id[obs]:
                        continue
                    patch_label[self.tokenizer.attribute2id[obs][e]] = 1

                chosen_aligned_entites = []
                if self.split == "train" and patch_idx in self.swap_annotation:
                    swap_info = self.swap_annotation[patch_idx]
                    candidate_patchs = [
                        (key, val[obs]) for key, val in swap_info.items() if obs in val
                    ]
                    if len(candidate_patchs) > 0:
                        chosen_patch = random.choice(candidate_patchs)
                        swap_img_path = os.path.join(
                            self.data_args.patch_path,
                            self.split,
                            chosen_patch[0],
                            chosen_patch[1]["patch_path"],
                        )
                        chosen_aligned_entites = chosen_patch[1]["aligned_entities"]
                        swap_patch_label = [0] * len(patch_label)
                        lambda_factor = 0.75
                        for e in chosen_aligned_entites:
                            if e not in self.tokenizer.attribute2id[obs]:
                                continue
                            swap_patch_label[self.tokenizer.attribute2id[obs][e]] = 1
                        swap_patch_lambda.append(lambda_factor)
                        swap_patch_paths.append(swap_img_path)
                        swap_patch_labels.append(swap_patch_label)
                    else:
                        swap_patch_paths.append(patch_paths[-1])
                        swap_patch_lambda.append(1.0)
                        swap_patch_labels.append(patch_label)

                if self.split == "train":
                    attributes = sorted(
                        patch_info["aligned_entities"],
                        key=lambda x: self.tokenizer.attribute2id[obs][x],
                    )
                    attributes_str = f"<Positive {obs}>"
                    if len(attributes) > 0:
                        attributes_str += self.tokenizer.sep_token + " ".join(
                            attributes
                        )

                    patch_input_ids.append(
                        self.tokenizer(
                            attributes_str, add_special_tokens=True
                        ).input_ids
                    )
                else:
                    patch_input_ids.append(
                        self.tokenizer(
                            f"<Positive {obs}>", add_special_tokens=True
                        ).input_ids
                    )

                patch_label = [self.observation_category.index(obs)] + patch_label
                patch_labels.append(patch_label)
        pixel_value = self.load_images(image_paths, transform=self.transform)
        prior_pixel_value = self.load_images(
            prior_image_paths, transform=self.transform
        )
        patch_pixel_value = self.load_images(patch_paths, transform=self.transform)
        if patch_pixel_value is not None and self.split == "train":
            swap_pixel_value = self.load_images(
                swap_patch_paths, transform=self.transform
            )
            if swap_pixel_value is not None:
                # Lesion-aware Mix-up Augmentation
                lambda_factor = (
                    torch.tensor(swap_patch_lambda)
                    .unsqueeze(1)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                    .to(patch_pixel_value.dtype)
                )
                patch_pixel_value = (
                    lambda_factor * patch_pixel_value
                    + (1.0 - lambda_factor) * swap_pixel_value
                )

        if prior_pixel_value is None:
            prior_pixel_value = torch.zeros_like(pixel_value[:1])

        if patch_pixel_value is None:
            patch_pixel_value = torch.zeros_like(pixel_value[:1])
            patch_input_ids = [
                self.tokenizer(
                    self.tokenizer.pad_token, add_special_tokens=True
                ).input_ids
            ]
            # inspect_head is not set for padding category, use the last head instead
            patch_labels = [[len(self.observation_category) - 2] + [-100] * topk]
            swap_patch_labels = [[-100] * topk]
            swap_patch_lambda = [1.0]

        if input_ids is None:
            input_ids = [self.tokenizer("", add_special_tokens=True).input_ids]

        item = {
            "labels": labels,
            "input_pixels": pixel_value,
            "prior_input_pixels": prior_pixel_value,
            "patch_input_pixels": patch_pixel_value,
            "input_ids": input_ids,
            "image_labels": image_labels,
            "patch_input_ids": patch_input_ids,
            "patch_labels": patch_labels,
        }
        if self.split != "train":
            item["report_ids"] = self.report2index[idx]
        else:
            item["swap_patch_labels"] = swap_patch_labels
            item["patch_factor"] = swap_patch_lambda

        return item

    def __len__(self):
        return len(self.data)

    def load_images(self, paths, transform=None):
        images = []
        pixel_value = None
        for path in paths:
            image = Image.open(path).convert("RGB")
            images.append(image)
        if len(images) > 0:
            pixel_value = self.processor(images=images, return_tensors="pt")[
                "pixel_values"
            ]
        return pixel_value
