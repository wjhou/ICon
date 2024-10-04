from tqdm import tqdm
from collections import defaultdict
from nltk.corpus import stopwords
import json
from transformers import AutoTokenizer
import torch
import copy

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_stage2.chexbert_eval import load_chexbert, CONDITIONS


class Retriever:
    def __init__(self, corpus):
        self.score_matrix = self.build_matrix(corpus)

    def build_matrix(self, corpus):
        samples = [set(s) for s in corpus]
        matrix = []
        for i in range(len(samples)):
            row = []
            for j in range(len(samples)):
                row.append(token_overlap(samples[i], samples[j]))
            matrix.append(row)
        return matrix

    def get_scores(self, idx):
        return self.score_matrix[idx]


def label_sentences(text, model, tokenizer):
    if isinstance(text, list):
        text = text[0]
    sentences = text.split(".")
    sentences = [s.strip() + "." for s in sentences if len(s.strip()) > 0]
    inputs = tokenizer(
        sentences,
        padding="longest",
        return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    logits = model(
        source_padded=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )[:-1]
    new_text = []
    for i in range(len(logits)):
        obs = CONDITIONS[i]
        if obs not in keep_obs:
            continue
        obs_logits = logits[i]
        for j in range(len(obs_logits)):
            cur_obs_logits = obs_logits[j]
            pred = cur_obs_logits.argmax(dim=0).item()
            if pred == 1 or pred == 3:
                s = sentences[j].replace(".", " .")
                if s in new_text:
                    continue
                new_text.append(s)
    return " ".join(new_text)


def label_func(texts, model, tokenizer, max_batch_size=768):
    batch_logits = None
    for i in tqdm(range(0, len(texts), max_batch_size), desc="Labeling Reports"):
        batch_hyps = texts[i : i + max_batch_size]
        inputs = tokenizer(batch_hyps, return_tensors="pt", padding="longest")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        logits = model(
            source_padded=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        if batch_logits is None:
            batch_logits = logits
        else:
            batch_logits = [
                torch.cat([a, b], dim=0) for a, b in zip(batch_logits, logits)
            ]
    return batch_logits


def label_reports(results, model, tokenizer):
    new_results = {}
    idxs = list(results.keys())
    hyps = []
    refs = []
    for idx in idxs:
        new_results[idx] = {
            "hyp": results[idx]["hyp"],
        }
        hyp = results[idx]["hyp"].replace(" .", ".")
        hyps.append(hyp)
        new_results[idx]["ref"] = results[idx]["ref"]
        ref = results[idx]["ref"].replace(" .", ".")
        refs.append(ref)

    batch_size = 768
    hyp_logits = label_func(hyps, model, tokenizer, max_batch_size=batch_size)
    ref_logits = label_func(refs, model, tokenizer, max_batch_size=batch_size)

    hyp_logits = [logits.cpu() for logits in hyp_logits]
    ref_logits = [logits.cpu() for logits in ref_logits]
    for pos, idx in enumerate(idxs):
        for logits, key in zip([hyp_logits, ref_logits], ["hyp_obs", "ref_obs"]):
            obs_list = []
            for i in range(len(logits)):
                obs = CONDITIONS[i]
                cur_obs_logits = logits[i][pos]
                pred = cur_obs_logits.argmax(dim=0).item()
                if pred == 1 or pred == 3:
                    pred_obs = f"{obs}:Positive"
                elif pred == 2:
                    pred_obs = f"{obs}:Negative"
                else:
                    pred_obs = None
                if pred_obs:
                    obs_list.append(pred_obs)
            new_results[idx][key] = obs_list
    return new_results


def token_overlap(t1, t2):
    t1 = set(t1)
    t2 = set(t2)
    tp = len(t1.intersection(t2))
    if len(t1) == 0 or len(t2) == 0:
        return 0
    f1 = tp / min(len(t1), len(t2))
    return f1


def compute_consistency(results, sim_refs, merged_attributes):
    for idx in results:
        hyp = results[idx]["hyp"]
        if isinstance(hyp, list):
            hyp = hyp[0]
        ref = results[idx]["ref"]
        hyp = hyp.replace("<unk>", "[UNK]")
        ref = ref.replace("<unk>", "[UNK]")
        results[idx]["hyp"] = hyp
        results[idx]["ref"] = ref
    count = 0
    for idx in tqdm(results, desc="Labeling reports"):
        count += 1
        results[idx]["hyp"] = label_sentences(results[idx]["hyp"], model, tokenizer)
        results[idx]["ref"] = label_sentences(results[idx]["ref"], model, tokenizer)
        if count < 3:
            print(f"{count} Hyp:", results[idx]["hyp"])
            print(f"{count} Ref:", results[idx]["ref"])
            print()
    ids = []
    hyps = []
    refs = []
    swords = set(stopwords.words("english"))
    punc = set(".,?;*!%^&_+():-\[\]\{\}")
    numbers = set("0123456789")
    swords.update(punc)
    swords.update(numbers)
    for study_id in results:
        hyp = [
            token
            for token in results[study_id]["hyp"].split()
            if token in merged_attributes
        ]
        ref = [
            token
            for token in results[study_id]["ref"].split()
            if token in merged_attributes
        ]
        hyps.append(hyp)
        refs.append(ref)
        ids.append(study_id)

    overlap_scores = []
    overlap_ref_scores = []
    gt_overlap_scores = []
    for idx in tqdm(sim_refs, desc="Computing Consistency"):
        pos = ids.index(idx)
        overlap_score = []
        gt_overlap_score = []
        hyp_i = hyps[pos]
        ref_score = token_overlap(hyp_i, refs[pos])
        equ_ids = sim_refs[idx]
        for equ_id in equ_ids:
            ref_j = refs[ids.index(equ_id)]
            gt_sim = token_overlap(refs[pos], ref_j)
            if gt_sim < 0.5:
                continue
            hyp_j = hyps[ids.index(equ_id)]
            sim = token_overlap(hyp_i, hyp_j)
            overlap_score.append(sim)
            gt_overlap_score.append(gt_sim)
        if (
            len(gt_overlap_score) == 0
            or sum(gt_overlap_score) / len(gt_overlap_score) == 0
        ):
            continue
        gt_ref_score = sum(gt_overlap_score) / len(gt_overlap_score)
        overlap_scores.append(sum(overlap_score) / len(overlap_score))
        overlap_ref_scores.append(ref_score)
        gt_overlap_scores.append(gt_ref_score)
    print(sum([1 if s == 0 else 0 for s in overlap_ref_scores]))
    print("--------------------------------------------------------------")
    print(path)
    print(
        f"Number of sample: {len(overlap_scores)}",
        sum(overlap_ref_scores) / len(overlap_ref_scores),
    )
    print(
        "Consistency for Hypothesis:",
        round(sum(overlap_scores) / len(overlap_scores), 4),
    )
    print(
        "Ref-Consistency for Hypothesis:",
        round(
            sum([a * b for a, b in zip(overlap_scores, overlap_ref_scores)])
            / len(overlap_scores),
            4,
        ),
    )
    print("--------------------------------------------------------------")
    print()


def retrieve(image2study, results, merged_attributes, topk=10, need_label=False):
    results = copy.deepcopy(results)
    swords = set(stopwords.words("english"))
    punc = set(".,?;*!%^&_+():-\[\]\{\}")
    numbers = set("0123456789")
    swords.update(punc)
    swords.update(numbers)
    if need_label:
        with torch.no_grad():
            results = label_reports(results, model, tokenizer)
    refs = []
    ref_obs = []
    ids = []
    for study_id in results:
        ref = [
            token
            for token in results[study_id]["ref"].split()
            if token in merged_attributes
        ]
        ref_o = [
            o
            for o in results[study_id]["ref_obs"]
            if o.split(":")[0] in keep_obs and "Pos" in o
        ]
        if len(ref_o) == 0:
            continue
        refs.append(ref)
        ref_obs.append(ref_o)
        ids.append(study_id)
    retriever = Retriever(ref_obs)

    sim_refs = {}
    for cidx in tqdm(range(len(ref_obs)), desc="Retrieving equivalent reference"):
        idx = ids[cidx]
        study_id = image2study[idx]
        scores = retriever.get_scores(cidx)
        if topk < len(scores):
            scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        else:
            scores = list(enumerate(scores))
        saved_study_ids = set()
        new_scores = []
        for score in scores[: topk + 10]:
            rel_study_id = image2study[ids[score[0]]]
            if (
                rel_study_id not in saved_study_ids
                and rel_study_id != study_id
                and score[1] >= 0.75
            ):
                new_scores.append((ids[score[0]], score[1]))

        scores = sorted(new_scores, key=lambda x: x[1], reverse=True)
        retrieved_idxs = [score[0] for score in scores[:topk]]
        if len(retrieved_idxs) == 0:
            continue
        sim_refs[idx] = retrieved_idxs
    return sim_refs


if __name__ == "__main__":
    import sys

    keep_obs = {
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
        "Support Devices",
    }
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = load_chexbert("./data/CheXbert/chexbert.pth")
    annotation = json.load(
        open("./data/mimic_cxr/annotation.json", "r", encoding="utf-8")
    )
    retrieval_corpus = []

    attributes = json.load(open("./data/mimic_cxr/triples.json", "r"))
    attributes = {
        k: v[:30]
        for k, v in attributes.items()
        if "Positive" in k and k.split(":")[0] in keep_obs
    }
    merged_attributes = defaultdict(set)
    for k, v in attributes.items():
        # merge both spatial and temporal attributes
        merged_attributes[k.split(":")[0]].update(v)

    attributes = set()
    for k, v in merged_attributes.items():
        attributes.update(v)
    merged_attributes = attributes

    # convert multiple images per study to multiple results
    image2study = {}
    for sample in annotation["test"]:
        idx = str(sample["id"])
        image_path = sample["image_path"]
        for image in image_path:
            image = image.split("/")[-1].split(".")[0]
            image2study[image] = idx

    path = sys.argv[1]
    results = json.load(open(path, "r", encoding="utf-8"))
    sim_refs = retrieve(
        image2study, results, merged_attributes, topk=10000, need_label=True
    )
    compute_consistency(results, sim_refs, merged_attributes)
