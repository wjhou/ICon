import json
from tqdm import tqdm
import argparse
from rank_bm25 import BM25Okapi
import os
import sys
from nltk.corpus import stopwords
from multiprocessing import Process, Manager
from typing import Set

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_stage1.tokenizer import Tokenizer


def f1(x: Set, y: Set):
    tp = len(x.intersection(y))
    if len(x) == 0 or len(y) == 0 or tp == 0:
        return 0
    recall = tp / len(y)
    precision = tp / len(x)
    return 2 * recall * precision / (precision + recall)


def retrieval(
    input_ids,
    corpus_idxs,
    corpus,
    id2doc,
    id2observation,
    results,
    temporal_ids,
):
    bm25 = BM25Okapi(corpus)
    for i in tqdm(range(len(input_ids)), desc="Retrieving ......"):
        idx = input_ids[i]
        is_temporal = temporal_ids is not None and str(idx) in temporal_ids
        query = id2doc[idx]
        # rank by tokens
        scores = bm25.get_scores(query)
        scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        scores = [
            (corpus_idxs[score[0]], score[1])
            for score in scores[:50]
            if idx != corpus_idxs[score[0]]
            and (not is_temporal or str(corpus_idxs[score[0]]) in temporal_ids)
        ]
        scores = sorted(
            scores,
            key=lambda x: (f1(x=id2observation[idx], y=id2observation[x[0]]), x[1]),
            reverse=True,
        )
        retrieved_idxs = [score[0] for score in scores[:10]]
        # rank by observation
        results[idx] = retrieved_idxs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation_path", type=str, required=True, help="the name of dataset"
    )
    parser.add_argument(
        "--chexbert_label", type=str, required=True, help="the output path"
    )
    parser.add_argument(
        "--cluster_path", type=str, required=True, help="the output path"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="the output path"
    )
    parser.add_argument(
        "--temporal_file", type=str, required=True, help="the output path"
    )
    args = parser.parse_args()
    id2observation, observation_category, _ = Tokenizer.load_tag2ids(
        args.chexbert_label, need_header=True
    )
    if args.temporal_file:
        temporal_ids = json.load(open(args.temporal_file, "r", encoding="utf-8"))
    else:
        temporal_ids = None
    args.dataset = "mimic_cxr" if "mimic_cxr" in args.annotation_path else "mimic_abn"
    attribute_data = json.load(
        open(
            f"./data/{args.dataset}/triples.json",
            "r",
        )
    )
    attributes = set()
    for obs in attribute_data:
        if "Pos" not in obs or "No Finding" in obs:
            continue
        attributes.update(attribute_data[obs][:30])

    K = 8
    clean_fn = (
        Tokenizer.clean_report_mimic_cxr
        if "mimic" in args.annotation_path
        else Tokenizer.clean_report_iu_xray
    )
    annotation = json.load(open(args.annotation_path, "r", encoding="utf-8"))["train"]
    swords = set(stopwords.words("english"))
    punc = set(".,?;*!%^&_+():-\[\]\{\}")
    numbers = set("0123456789")
    swords.update(punc)
    swords.update(numbers)
    studyid2observation = {}
    studyid2report = {}

    for sample in annotation:
        study_id = str(sample["id"])
        report = [
            tok
            for tok in clean_fn(sample["report"]).split()
            if tok in attributes
        ]
        if "iu_xray" in args.annotation_path:
            image_id = study_id
        else:
            image_id = sample["image_path"][0].split("/")[-1].split(".")[0]
        studyid2observation[study_id] = {
            b
            for a, b in zip(id2observation[image_id], observation_category)
            if a == -1 or a == 1 and "No Finding" not in b
        }

        studyid2observation[study_id] = {o for o in studyid2observation[study_id]}

        studyid2report[study_id] = report

    corpus = []
    idxs = []
    for idx in studyid2observation:
        idxs.append(idx)
        doc = []
        if idx in studyid2report:
            doc.extend(studyid2report[idx])
        doc.extend(studyid2observation[idx])
        corpus.append(doc)
        if len(corpus) < 10:
            print(idx, corpus[-1])
    id2doc = {idx: doc for idx, doc in zip(idxs, corpus)}
    num_process = 64
    size = len(corpus) // num_process
    procs = []
    manager = Manager()
    results = manager.dict()
    for i in range(num_process):
        start = i * size
        end = (i + 1) * size if i < num_process - 1 else len(corpus)
        input_ids = idxs[start:end]
        proc = Process(
            target=retrieval,
            args=(
                input_ids,
                idxs,
                corpus,
                id2doc,
                studyid2observation,
                results,
                temporal_ids,
            ),
        )
        procs.append(proc)
        proc.start()
    for proc in procs:
        proc.join()

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results.copy(), f, indent=4, ensure_ascii=False)
