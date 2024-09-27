import os

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from PIL import Image


def process_examples(examples):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    idxs = []
    image_paths = []
    for index in progress:
        report_id = examples["id"][index]
        image_path = examples["image_path"][index]
        idxs.append(report_id)
        image_paths.append(image_path)

    return (
        idxs,
        image_paths,
    )


def load_images(root_path, image_paths):
    images = {}
    for image_path in tqdm(image_paths, desc="Loading Images"):
        for img_path in image_path:
            img_path = os.path.join(root_path, img_path)
            image = Image.open(img_path).convert("RGB")
            images[img_path] = image
    return images


def extract_temporal_info(
    samples,
    ref_samples,
    temporal_ids,
):
    id2sample = {sample["id"]: sample for sample in samples}
    if ref_samples is not None:
        ref_id2sample = {sample["id"]: sample for sample in ref_samples}
        for subject_id in temporal_ids:
            object_id = temporal_ids[subject_id]["object_id"]
            if object_id not in id2sample:
                id2sample[object_id] = ref_id2sample[object_id]

    for sample in samples:
        sample["temporal_image_path"] = []
        sample["temporal_report"] = ""
        sample["temporal_predicate"] = []

    for subject_id in tqdm(temporal_ids, desc="Updating Temooral Info"):
        predicate_object = temporal_ids[subject_id]
        predicate = predicate_object["predicate"]

        object_id = predicate_object["object_id"]
        if object_id not in id2sample:
            print(object_id, "Not Found")
        else:
            object_example = id2sample[object_id]
            subject_example = id2sample[subject_id]
            subject_example["temporal_image_path"] = object_example["image_path"]
            subject_example["temporal_report"] = object_example["report"]
            subject_example["temporal_predicate"] = predicate
    return samples


class DatasetCustom(Dataset):
    def __init__(
        self,
        data_args,
        annotation,
        split: str,
        id2tags,
        processor,
        observation_category,
        transform=None,
        dataset=None,
        keep_columns={
            "id",
            "image_path",
        },
    ) -> None:
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        self.split = split
        self.dataset = (
            dataset
            if dataset is not None
            else ("iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr")
        )
        self.id2tags = id2tags
        examples = {kc: [] for kc in keep_columns}
        samples = annotation[split.replace("valid", "val")]
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
        ) = process_examples(examples=examples)
        self.data = [{"id": a, "image_path": b} for a, b in zip(idxs, image_paths)]
        self.transform = transform
        self.observation_category = observation_category
        self.return_raw_img = self.data_args.return_raw_img

    def __getitem__(self, index):
        idx = self.data[index]["id"]

        if self.dataset == "iu_xray":
            observation_id = idx
        else:
            observation_id = (
                self.data[index]["image_path"][0].split("/")[-1].split(".")[0]
            )
        image_path = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["image_path"]
        ]
        pixel_value = []
        if self.return_raw_img:
            raw_imgs = []

        for img_path in image_path:
            image = Image.open(img_path)
            image = image.convert("RGB")
            if self.return_raw_img:
                raw_imgs.append(image)
            if self.transform is not None:
                image = self.transform(image)
            image = self.processor(images=image, return_tensors="pt")["pixel_values"]
            pixel_value.append(image)
        pixel_value = torch.cat(pixel_value, dim=0)

        observations = [0] * len(self.observation_category)
        normals = [0] * (len(self.observation_category) - 1)
        for tag, obs in zip(self.id2tags[observation_id], self.observation_category):
            if tag == 2 or tag == 0:
                continue
            observations[self.observation_category.index(obs)] = 1

        for tag, obs in zip(self.id2tags[observation_id], self.observation_category[:-1]):
            if tag == 0:
                normals[self.observation_category.index(obs)] = 1
            elif tag != 2:
                normals[self.observation_category.index(obs)] = -100

        item = {
            "input_pixels": pixel_value,
            "split": self.split,
            "observations": observations,
            "normals": normals,
        }
        if self.split != "train" or self.return_raw_img:
            item["report_ids"] = idx
        if self.return_raw_img:
            item["raw_imgs"] = raw_imgs
        return item

    def __len__(self):
        return len(self.data)
