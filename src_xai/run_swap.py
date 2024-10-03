import argparse
import json
import os


def main(args):
    annotation = json.load(open(args.annotation_path, "r", encoding="utf-8"))
    patch_annotation = json.load(open(args.patch_path, "r", encoding="utf-8"))
    pseudo_mapping = json.load(open(args.retrieval_path, "r", encoding="utf-8"))

    patch_database = patch_annotation["train"]
    swap_patch_annotation = {}
    for split in annotation:
        swap_patch_annotation[split] = {}
        data = annotation[split]
        patch_data = patch_annotation[split.replace("val", "valid")]
        for sample in data:
            study_id = str(sample["id"])
            if study_id not in patch_data or study_id not in pseudo_mapping:
                continue
            patch_info = patch_data[study_id]
            pseudo_study_ids = pseudo_mapping[study_id][:4]
            swap_patch_infos = {}
            for pseudo_study_id in pseudo_study_ids:
                if pseudo_study_id not in patch_database:
                    continue
                pseudo_patch_info = patch_database[pseudo_study_id]
                swap_patch_info = {}
                for obs in patch_info:
                    if obs not in pseudo_patch_info:
                        continue
                    swap_patch_info[obs] = pseudo_patch_info[obs]
                if len(swap_patch_info) > 0:
                    swap_patch_infos[pseudo_study_id] = swap_patch_info
            if len(swap_patch_infos) > 0:
                swap_patch_annotation[split][study_id] = swap_patch_infos

    with open(args.output_dir, "w", encoding="utf-8") as f:
        json.dump(swap_patch_annotation, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mimic_cxr")
    parser.add_argument(
        "--annotation_path",
        type=str,
        default="/data/mimic_cxr/gather_annotation.json",
    )
    parser.add_argument(
        "--patch_path",
        type=str,
        default="./data/mimic_cxr/",
    )
    parser.add_argument(
        "--retrieval_path",
        type=str,
        default="./data/20240101/",
    )
    parser.add_argument("--output_dir", type=str, default="./data/mimic_cxr/")
    args = parser.parse_args()
    main(args)
