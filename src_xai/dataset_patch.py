import os

from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from PIL import Image
from numpy import asarray


def process_examples(examples):
    progress = tqdm(
        range(len(examples["id"])),
        desc="Processing Samples",
    )
    idxs = []
    image_paths = []
    views = []
    for index in progress:
        report_id = examples["id"][index]
        image_path = examples["image_path"][index]
        idxs.append(report_id)
        image_paths.append(image_path)
        views.append(examples["view"][index])

    return (
        idxs,
        image_paths,
        views,
    )


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
        patch_size=None,
        step=128,
        keep_columns={
            "id",
            "image_path",
            "view",
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
            views,
        ) = process_examples(examples=examples)
        self.data = [
            {
                "id": a,
                "image_path": b,
                "views": c,
            }
            for a, b, c in zip(
                idxs,
                image_paths,
                views,
            )
        ]
        self.transform = transform
        self.observation_category = observation_category
        if self.dataset == "iu_xray":
            self.view2id = {
                "FRONTAL": 0,
                "LATERAL": 1,
                "UNKNOWN": 2,
            }
        else:
            self.view2id = {
                "PA": 0,
                "AP": 1,
                "LATERAL": 2,
                "UNKNOWN": 3,
            }
        self.id2view = {v: k for k, v in self.view2id.items()}
        self.patch_size = patch_size
        self.step = step

    def __getitem__(self, index):
        idx = self.data[index]["id"]

        observation_id = (
            self.data[index]["image_path"][0].split("/")[-1].split(".")[0]
            if "mimic" in self.dataset
            else self.data[index]["id"]
        )
        image_path = [
            os.path.join(self.data_args.image_path, a)
            for a in self.data[index]["image_path"]
        ]
        pixel_value = []
        views = []
        raw_patches = []
        patch_path = []
        patch_pos = []
        for img_path, view in zip(image_path, self.data[index]["views"]):
            image = Image.open(img_path).convert("RGB")
            image = asarray(image)
            h, w, _ = image.shape
            ratio = self.patch_size // self.step
            pos = 0
            patches = []
            image_name = img_path.split("/")[-1].split(".")[0]
            for i in range(0, h, self.step):
                if i + self.step * ratio > h:
                    continue
                for j in range(0, w, self.step):
                    if j + self.step * ratio > w:
                        continue
                    patch = image[i : i + self.patch_size, j : j + self.patch_size]
                    patches.append(patch)
                    raw_patches.append(patch[:, :, 0])
                    views.append(self.view2id.get(view, 3))
                    patch_pos.append(pos)
                    pos += 1
                    patch_path.append(image_name)
            patches = self.processor(images=patches, return_tensors="pt")[
                "pixel_values"
            ]
            pixel_value.append(patches)
        pixel_value = torch.cat(pixel_value, dim=0)

        observations = [0] * len(self.observation_category)
        for tag, obs in zip(self.id2tags[observation_id], self.observation_category):
            if tag == 2 or tag == 0 or "No Finding" in obs:
                continue
            observations[self.observation_category.index(obs)] = 1

        item = {
            "input_pixels": pixel_value,
            "split": self.split,
            "observations": observations,
            "views": views,
            "raw_patches": raw_patches,
            "patch_path": patch_path,
            "patch_pos": patch_pos,
            "report_ids": idx,
        }
        return item

    def __len__(self):
        return len(self.data)
