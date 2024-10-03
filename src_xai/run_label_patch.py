import json
from tqdm import tqdm
from collections import defaultdict
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_stage1.tokenizer import Tokenizer

def func_iu_xray(
    sentence_level_annotation,
    triples,
    patch_annotation,
    report_level_annotation,
    align_patch_annotation,
    topk=30,
):
    with open(report_level_annotation, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    with open(sentence_level_annotation, "r", encoding="utf-8") as f:
        sentence_annotation = json.load(f)
    with open(patch_annotation, "r", encoding="utf-8") as f:
        patch_annotation = json.load(f)
    with open(triples, "r", encoding="utf-8") as f:
        triples = json.load(f)
    triples_ = {
        k: v[:topk]
        for k, v in triples.items()
        if "Positive" in k and "No Finding" not in k
    }
    updated_triples = defaultdict(set)
    for obs in triples_:
        updated_triples[obs.split(":")[0]].update(triples_[obs])

    normal_triples = {
        k: v[:topk]
        for k, v in triples.items()
        if "No Finding" not in k and "Negative" in k
    }
    normal_entities = set()
    for obs in normal_triples:
        normal_entities.update(normal_triples[obs][:10])

    clean_fn = (
        Tokenizer.clean_report_mimic_cxr
        if "mimic" in align_patch_annotation
        else Tokenizer.clean_report_iu_xray
    )
    aligned_entities = {}
    for split in annotation:
        samples = annotation[split]
        aligned_entities[split] = {}
        for sample in tqdm(samples, desc=f"Loading {split}"):
            study_id = sample["id"]
            report = set(clean_fn(sample["report"]).split())
            entities = report.intersection(normal_entities)
            aligned_entities[split][str(study_id)] = list(entities)

    for split in patch_annotation:
        sentence_samples = sentence_annotation[split.replace("valid", "val")]
        samples = patch_annotation[split]
        for study_id in tqdm(samples, desc=f"Loading {split}"):
            sentence_observation = sentence_samples.get(study_id, {})
            obs2text = defaultdict(set)
            report = set()
            all_obs = samples[study_id].keys()
            for pos in sentence_observation:
                info = sentence_observation[pos]
                tokens = info["sentence"].split()
                report.update(tokens)
                for obs in info["observation"]:
                    if "Positive" in obs and obs.split(":")[0] in all_obs:
                        obs2text[obs.split(":")[0]].update(tokens)

            for obs in all_obs:
                if obs not in obs2text:
                    obs2text[obs] = report

            for obs in obs2text:
                entities = updated_triples[obs]
                candidate_entities = obs2text[obs]
                samples[study_id][obs]["aligned_entities"] = list(
                    entities.intersection(candidate_entities)
                )
            samples[study_id]["aligned_entities"] = aligned_entities[
                split.replace("valid", "val")
            ][str(study_id)]

    with open(align_patch_annotation, "w", encoding="utf-8") as f:
        json.dump(patch_annotation, f, indent=4)

def func(
    sentence_level_annotation,
    triples,
    patch_annotation,
    report_level_annotation,
    align_patch_annotation,
    topk=30,
):
    with open(report_level_annotation, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    with open(sentence_level_annotation, "r", encoding="utf-8") as f:
        sentence_annotation = json.load(f)
    with open(patch_annotation, "r", encoding="utf-8") as f:
        patch_annotation = json.load(f)
    with open(triples, "r", encoding="utf-8") as f:
        triples = json.load(f)
    triples_ = {
        k: v[:topk]
        for k, v in triples.items()
        if "Positive" in k and "No Finding" not in k
    }
    updated_triples = defaultdict(set)
    for obs in triples_:
        updated_triples[obs.split(":")[0]].update(triples_[obs])

    normal_triples = {
        k: v[:topk]
        for k, v in triples.items()
        if "No Finding" not in k and "Negative" in k
    }
    # normal_updated_triples = defaultdict(set)
    # for obs in normal_triples:
    #     normal_updated_triples[obs.split(":")[0]].update(normal_triples[obs])
    normal_entities = set()
    for obs in normal_triples:
        normal_entities.update(normal_triples[obs][:10])

    clean_fn = (
        Tokenizer.clean_report_mimic_cxr
        if "mimic" in align_patch_annotation
        else Tokenizer.clean_report_iu_xray
    )
    image2study = {}
    study2image = {}
    aligned_entities = {}
    for split in annotation:
        samples = annotation[split]
        aligned_entities[split] = {}
        for sample in tqdm(samples, desc=f"Loading {split}"):
            study_id = sample["study_id"] if "study_id" in sample else sample["id"]
            image_id = sample["image_path"][0].split("/")[-1].split(".")[0]
            image2study[image_id] = study_id
            study2image[study_id] = image_id
            report = set(clean_fn(sample["report"]).split())
            entities = report.intersection(normal_entities)
            aligned_entities[split][str(study_id)] = list(entities)

    for split in patch_annotation:
        sentence_samples = sentence_annotation[split.replace("valid", "val")]
        samples = patch_annotation[split]
        for study_id in tqdm(samples, desc=f"Loading {split}"):
            fake_study_id = (
                int(study_id) if "mimic" in align_patch_annotation else study_id
            )
            image_id = study2image[fake_study_id]
            sentence_observation = sentence_samples.get(image_id, {})
            obs2text = defaultdict(set)
            report = set()
            all_obs = samples[study_id].keys()
            for pos in sentence_observation:
                info = sentence_observation[pos]
                tokens = info["sentence"].split()
                report.update(tokens)
                for obs in info["observation"]:
                    if "Positive" in obs and obs.split(":")[0] in all_obs:
                        obs2text[obs.split(":")[0]].update(tokens)

            for obs in all_obs:
                if obs not in obs2text:
                    obs2text[obs] = report

            for obs in obs2text:
                entities = updated_triples[obs]
                candidate_entities = obs2text[obs]
                samples[study_id][obs]["aligned_entities"] = list(
                    entities.intersection(candidate_entities)
                )
            samples[study_id]["aligned_entities"] = aligned_entities[
                split.replace("valid", "val")
            ][str(study_id)]

    with open(align_patch_annotation, "w", encoding="utf-8") as f:
        json.dump(patch_annotation, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mimic_cxr")
    parser.add_argument("--triple_path", type=str)
    parser.add_argument("--sentence_annotation", type=str)
    parser.add_argument("--patch_annotation", type=str)
    parser.add_argument("--report_annotation", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    func(
        args.sentence_annotation,
        args.triple_path,
        args.patch_annotation,
        args.report_annotation,
        args.output_dir,
        30,
    )
