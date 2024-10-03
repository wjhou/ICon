import warnings

warnings.filterwarnings("ignore")

from PIL import Image
import json
import torch

from transformers import ViTFeatureExtractor, SwinConfig
import argparse
import sys
import os
from safetensors.torch import load_model
from tqdm import tqdm
from dataset_patch import DatasetCustom

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_stage1.models.modeling_swin import VisualEncoder
from src_stage1.tokenizer import Tokenizer


def reshape_transform(tensor, height=8, width=8):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def visualize_image(
    model: VisualEncoder,
    dataset,
    split,
    labels=None,
    predicted_observation=None,
    output_dir=None,
    batch_size=8,
):
    result = {}
    for i in tqdm(range(0, len(dataset), batch_size), desc="Selecting Patches ......"):
        batch = [dataset[i + j] for j in range(batch_size) if i + j < len(dataset)]
        input_pixels = torch.cat([sample["input_pixels"] for sample in batch], dim=0)
        outputs = []
        mini_batch_size = 512
        for j in range(0, len(input_pixels), mini_batch_size):
            inputs = {
                "input_pixels": input_pixels[j : j + mini_batch_size],
            }

            inputs = {k: v.cuda() if v is not None else v for k, v in inputs.items()}
            output = model(**inputs).observation_cls_logits
            outputs.append(output)
        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = torch.cat(outputs, dim=0)
        count = 0
        for sample in batch:
            sample_id = sample["report_ids"]
            count += len(sample["views"])
            if predicted_observation is not None:
                obs_hyp = predicted_observation[sample_id]
                observation = [0] * len(labels)
                for obs in obs_hyp:
                    if "POS" not in obs:
                        continue
                    observation[labels.index(obs.split("_")[0])] = 1
            else:
                observation = sample["observations"]
            sample_folder = os.path.join(output_dir, str(sample_id))
            if not os.path.exists(sample_folder):
                os.makedirs(sample_folder)
            outputs_ = outputs[count - len(sample["views"]) : count]
            raw_patches = sample["raw_patches"]
            obs2patch = {}

            for obs_idx, obs in enumerate(labels):
                if observation[obs_idx] == 0 or "No Finding" in obs:
                    continue
                out = outputs_[:, obs_idx].topk(1, dim=0)
                selected_idx = out[1][0].item()
                selected_score = out[0][0].item()
                selected_position = sample["patch_pos"][selected_idx]
                obs2patch[obs] = {
                    "id": selected_idx,
                    "score": round(selected_score, 4),
                    "patch_path": sample["patch_path"][selected_idx]
                    + "_p"
                    + str(selected_position)
                    + ".jpg",
                    "patch_view": dataset.id2view[sample["views"][selected_idx]],
                    "patch_position": selected_position,
                }
            for selected_idx, p in {
                val["id"]: val["patch_path"] for val in obs2patch.values()
            }.items():
                Image.fromarray(raw_patches[selected_idx], mode="L").save(
                    os.path.join(sample_folder, p)
                )
            result[sample_id] = obs2patch
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mimic_cxr")
    parser.add_argument(
        "--image_path",
        type=str,
        default="./data/mimic_cxr/images",
    )
    parser.add_argument("--patch_size", type=int, default=True)
    parser.add_argument("--output_dir", type=str, default="./data/mimic_cxr/")
    parser.add_argument(
        "--chexbert_label",
        type=str,
        default="./data/mimic_cxr/id2tags.json",
    )
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--mimic_cxr_model_path", default=None, type=str)
    parser.add_argument("--best_checkpoint", type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    id2tags, observation_category, observation_weight = Tokenizer.load_tag2ids(
        args.chexbert_label,
        need_header=True,
    )
    # checkpoint = "GanjinZero/biobart-base"
    # config = BartConfig.from_pretrained(checkpoint)
    vision_config = SwinConfig.from_pretrained(args.model_path)
    vision_config.num_observation = len(observation_category)
    vision_config.num_view = (2 if args.dataset == "iu_xray" else 3) + 1
    vision_config.pretrained_visual_extractor = (
        "microsoft/swinv2-small-patch4-window8-256"
    )
    vision_config.observation_weight = observation_weight
    model = VisualEncoder(vision_config)
    processor = ViTFeatureExtractor.from_pretrained(
        vision_config.pretrained_visual_extractor, size=256
    )
    load_model(
        model,
        os.path.join(
            (
                args.model_path
                if args.mimic_cxr_model_path is None
                else args.mimic_cxr_model_path
            ),
            "model.safetensors",
        ),
    )
    model.eval()
    model = model.cuda()
    annotation = json.load(
        open(
            "./data/{dataset}/annotation.json".format(dataset=args.dataset),
            "r",
            encoding="utf-8",
        )
    )
    results = {}
    for split in [
        "train",
        "valid",
        "test",
    ]:
        observation_file, predicted_observation = None, None
        if split == "valid":
            observation_file = "results_eval_step_{step}.json".format(
                step=args.best_checkpoint
            )
        elif split == "test":
            observation_file = "results.json"
        if observation_file is not None:
            predicted_observation = json.load(
                open(
                    os.path.join(
                        args.model_path,
                        observation_file,
                    ),
                    "r",
                    encoding="utf-8",
                )
            )
            predicted_observation = {
                int(k) if "mimic" in args.dataset else k: v["obs_hyp"]
                for k, v in predicted_observation.items()
            }
        data = DatasetCustom(
            data_args=args,
            split=split,
            annotation=annotation,
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
            dataset=args.dataset,
            patch_size=args.patch_size,
        )
        print("Datasize:", len(data))
        output_dir = os.path.join(args.output_dir, split)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with torch.no_grad():
            result = visualize_image(
                model=model,
                dataset=data,
                split=split,
                labels=observation_category,
                predicted_observation=predicted_observation,
                output_dir=output_dir,
                batch_size=args.batch_size,
            )
        results[split] = result

    with open(
        os.path.join(args.output_dir, "patch_annotation.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
