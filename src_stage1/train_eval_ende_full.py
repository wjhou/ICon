import os

import torch
from tqdm import tqdm

import json
from sklearn.metrics import precision_recall_fscore_support
import numpy as np


def pad_strings(strs):
    max_len = max([len(s) for s in strs])
    return [s + " " * (max_len - len(s)) for s in strs]


def extend_sample(samples, num_views):
    outputs = []
    for sample, num_view in zip(samples, num_views):
        num_view = sum(num_view)
        outputs.extend([sample] * num_view)
    return np.array(outputs)


def train(training_args, data_args, last_checkpoint, trainer, train_dataset):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def index2label(observation_labels, progression_labels, inputs):
    outputs = []
    for input in inputs:
        output = []
        for i, j in enumerate(input):
            if j == -100:
                continue
            o = observation_labels[i]
            p = progression_labels[j]
            output.append(o + "_" + p)
        outputs.append(output)
    return outputs


def clean_normal(normal_preds, abnormal_preds):
    new_normal_preds = []
    for normal_pred, abnormal_pred in zip(normal_preds, abnormal_preds):
        new_normal_pred = []
        for a, b in zip(normal_pred, abnormal_pred):
            if b == 1:
                new_normal_pred.append(0)
            else:
                new_normal_pred.append(a)
        new_normal_preds.append(new_normal_pred)
    return new_normal_preds


def eval_text(
    max_tgt_length: int,
    model,
    test_dataset,
    output_path: str,
    result_file_name: str = "results.txt",
    reference_file_name: str = "references.txt",
    prediction_file_name: str = "predictions.txt",
):
    model.eval()

    max_length = max_tgt_length
    print("******************")
    print("Text generation max length", max_length)
    print("******************")
    # for all report
    predictions = []
    references = []
    report_ids = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model",
    )
    observation_labels = model.config.observation_category
    pad_observation_labels = pad_strings(observation_labels)
    abnormal_preds, normal_preds = [], []
    abnormal_trues, normal_trues = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            model_inputs = {
                "input_pixels": batch["input_pixels"].cuda(),
                "image_mask": batch["image_mask"].cuda(),
                "gather_index": batch["gather_index"].cuda(),
            }
            outputs = model(**model_inputs)
            observation_cls_logits = outputs.observation_cls_logits
            observation_cls_pred = (observation_cls_logits > 0).float().cpu().numpy()
            observation_true = batch["observations"].cpu().numpy()

            normal_cls_logits = outputs.normal_cls_logits
            normal_cls_pred = (normal_cls_logits > 0).float().cpu().numpy()
            normal_true = batch["normals"].cpu().numpy()

            normal_cls_pred = clean_normal(normal_cls_pred, observation_cls_pred)

            num_views = batch["image_mask"].tolist()
            if test_dataset.dataset == "iu_xray":
                num_views = [[1] * len(num_views)]
            observation_cls_pred = extend_sample(observation_cls_pred, num_views)
            observation_true = extend_sample(observation_true, num_views)
            normal_cls_pred = extend_sample(normal_cls_pred, num_views)
            normal_true = extend_sample(normal_true, num_views)
            report_id = extend_sample([str(s) for s in batch["report_ids"]], num_views)
            abnormal_preds.append(observation_cls_pred)
            abnormal_trues.append(observation_true)
            normal_preds.append(normal_cls_pred)
            normal_trues.append(normal_true)
            report_ids.extend(report_id)

    abnormal_preds = np.concatenate(abnormal_preds, axis=0)
    abnormal_trues = np.concatenate(abnormal_trues, axis=0)
    normal_preds = np.concatenate(normal_preds, axis=0)
    normal_trues = np.concatenate(normal_trues, axis=0)
    normal_trues[normal_trues == -100] = 0
    assert len(abnormal_preds) == len(abnormal_trues) == len(report_ids)
    assert len(normal_preds) == len(normal_trues) == len(report_ids)
    ce_scores = [0, 0, 0]
    print("--------------------------------------------------------------")
    for i in range(normal_preds.shape[1]):
        y_pred = normal_preds[:, i]
        y_true = normal_trues[:, i]
        i_ce_score = precision_recall_fscore_support(
            y_pred=y_pred,
            y_true=y_true,
            pos_label=1,
            average="binary",
        )[:-1]
        print(
            "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (pad_observation_labels[i], *i_ce_score)
        )
        ce_scores = [
            ce_scores[i] + i_ce_score[i] / normal_preds.shape[1]
            for i in range(len(ce_scores))
        ]
    print("--------------------------------------------------------------")
    print(
        "Normal CE Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
        % (ce_scores[0], ce_scores[1], ce_scores[2])
    )
    print("--------------------------------------------------------------")

    ce_scores = [0, 0, 0]
    print("--------------------------------------------------------------")
    for i in range(abnormal_preds.shape[1]):
        y_pred = abnormal_preds[:, i]
        y_true = abnormal_trues[:, i]
        i_ce_score = precision_recall_fscore_support(
            y_pred=y_pred,
            y_true=y_true,
            pos_label=1,
            average="binary",
        )[:-1]
        print(
            "%s\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
            % (pad_observation_labels[i], *i_ce_score)
        )
        ce_scores = [
            ce_scores[i] + i_ce_score[i] / abnormal_preds.shape[1]
            for i in range(len(ce_scores))
        ]
    print("--------------------------------------------------------------")
    print(
        "Abnormal CE Scores\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
        % (ce_scores[0], ce_scores[1], ce_scores[2])
    )
    print("--------------------------------------------------------------")
    target = ce_scores[2]
    output_data = {}
    for (
        sample_index,
        report_id,
        abnormal_pred,
        abnormal_true,
        normal_pred,
        normal_true,
    ) in zip(
        range(len(report_ids)),
        report_ids,
        abnormal_preds,
        abnormal_trues,
        normal_preds,
        normal_trues,
    ):
        hyp = [
            a + "_POSITIVE"
            for a, b in zip(
                observation_labels,
                abnormal_pred,
            )
            if b == 1 and "No Finding" not in a
        ]
        hyp.insert(
            0,
            "No Finding_"
            + (
                "POSITIVE"
                if abnormal_pred[observation_labels.index("No Finding")] == 1
                else "NEGATIVE"
            ),
        )
        hyp.extend(
            [a + "_NEGATIVE" for a, b in zip(observation_labels, normal_pred) if b == 1]
        )
        ref = [
            a + "_POSITIVE"
            for a, b in zip(
                observation_labels,
                abnormal_true,
            )
            if b == 1 and "No Finding" not in a
        ]
        ref.insert(
            0,
            "No Finding_"
            + (
                "POSITIVE"
                if abnormal_true[observation_labels.index("No Finding")] == 1
                else "NEGATIVE"
            ),
        )
        ref.extend(
            [a + "_NEGATIVE" for a, b in zip(observation_labels, normal_true) if b == 1]
        )
        output_data[report_id] = {
            "obs_hyp": hyp,
            "obs_ref": ref,
        }

    if output_path:
        with open(
            os.path.join(output_path, result_file_name),
            "w",
            encoding="utf-8",
        ) as f, open(
            os.path.join(output_path, reference_file_name),
            "w",
            encoding="utf-8",
        ) as f2, open(
            os.path.join(output_path, prediction_file_name),
            "w",
            encoding="utf-8",
        ) as f3:
            f.write(",".join(("Reference", "Prediction")) + "\n")
            for idx, pre, ref in zip(
                range(len(predictions)),
                predictions,
                references,
            ):
                f.write("Reference:\t%s\n" % ",".join(ref))
                f.write("Prediction:\t%s\n" % ",".join(pre))
                f3.write(pre + "\n")
                f2.write(ref + "\n")
            f.write("****************\n")
        with open(
            os.path.join(output_path, result_file_name.replace("txt", "json")),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
    return {"eval_score": target}
