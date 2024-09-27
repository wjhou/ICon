import os

import torch
from tqdm import tqdm

from metrics import compute_scores
import json
from models.modeling_bart import ViTBartForConditionalGeneration
from chexbert_eval import CONDITIONS, TEM_keywords
from transformers import BartTokenizer, Trainer
import re
import sys
from accelerate import Accelerator

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src_stage1.train_eval_ende_full import extend_sample


def pad_strings(strs):
    max_len = max([len(s) for s in strs])
    return [s + " " * (max_len - len(s)) for s in strs]


def post_processing_multi(texts, special_tokens, unk_token):
    new_texts = []

    def post_processing_with_unk(text, special_tokens, unk_token):
        for special_token in special_tokens:
            if not isinstance(special_token, str):
                continue
            if special_token == unk_token:
                continue
            text = text.replace(special_token, "")

        text = re.sub(" +", " ", text)
        return text.replace(".", " .").strip()

    for text in texts:
        new_texts.append(post_processing_with_unk(text, special_tokens, unk_token))
    return new_texts


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


def inspect(
    batch, model: ViTBartForConditionalGeneration, tokenizer, observation_category
):
    _, lesion_pooler_output, _ = model.model.encoder.vision_encoder.encode(
        batch["patch_input_pixels"].cuda(),
        None,
        None,
        encoder=model.model.encoder.vision_encoder.lesion_encoder,
        # encoder=model.model.encoder.vision_encoder.vision_encoder,
        projection=model.model.encoder.vision_encoder.lesion_projection,
    )
    prior_pooler_output = None
    if batch["prior_image_mask"].sum() > 0:
        _, prior_pooler_output, _ = model.model.encoder.vision_encoder.encode(
            batch["prior_input_pixels"].cuda(),
            image_mask=batch["prior_image_mask"].cuda(),
            gather_index=batch["prior_gather_index"].cuda(),
            encoder=model.model.encoder.vision_encoder.vision_encoder,
            projection=model.model.encoder.vision_encoder.prior_projection,
        )
    _, pooler_output, _ = model.model.encoder.vision_encoder.encode(
        batch["input_pixels"].cuda(),
        image_mask=batch["image_mask"].cuda(),
        gather_index=batch["gather_index"].cuda(),
        encoder=model.model.encoder.vision_encoder.vision_encoder,
        projection=model.model.encoder.vision_encoder.prior_projection,
    )
    if prior_pooler_output is None:
        prior_pooler_output = torch.zeros_like(pooler_output)
    one_hot_index = torch.nn.functional.one_hot(
        batch["patch_context_index"].cuda(), num_classes=pooler_output.size(0)
    ).float()
    pooler_output = torch.einsum("bi,ik->bk", one_hot_index, pooler_output)
    if prior_pooler_output is None:
        prior_pooler_output = torch.zeros_like(lesion_pooler_output)
    else:
        prior_pooler_output = torch.einsum(
            "bi,ik->bk", one_hot_index, prior_pooler_output
        )
    lesion_pooler_output = torch.cat(
        (prior_pooler_output, pooler_output, lesion_pooler_output), dim=-1
    )
    observations = batch["patch_labels"][:, 0]
    lesion_inspect_logits = [
        model.model.encoder.vision_encoder.inspect_head[i](lesion_pooler_output[pos])
        for pos, i in enumerate(observations)
    ]

    lesion_attributes = []
    for logits, observation_id, mask in zip(
        lesion_inspect_logits, observations, batch["patch_labels"][:, 1:]
    ):
        observation = observation_category[observation_id]
        attributes = [
            tokenizer.id2attribute[observation][i]
            for i in [
                a_idx
                for a_idx, (logit, m) in enumerate(zip(logits, mask))
                if logit > 0 and m != -100
            ]
        ]
        attribute_str = f"<Positive {observation}>"
        if len(attributes) > 0:
            attribute_str += tokenizer.sep_token + " ".join(attributes)
        lesion_attributes.append(attribute_str)

    patch_outputs = tokenizer.batch_encode_plus(
        lesion_attributes,
        padding="longest",
        return_tensors="pt",
    )
    return patch_outputs.input_ids.cuda(), patch_outputs.attention_mask.cuda()


def eval_text(
    max_tgt_length: int,
    model: ViTBartForConditionalGeneration,
    tokenizer,
    test_dataset,
    output_path: str,
    result_file_name: str = "results.txt",
    reference_file_name: str = "references.txt",
    prediction_file_name: str = "predictions.txt",
    num_beams=None,
    compute_ce_metric=None,
    chexbert=None,
    bert_tokenizer=None,
    decoder_tokenizer: BartTokenizer = None,
    accelerator: Accelerator = None,
    trainer: Trainer = None,
    validation=True,
    report_ids_map=None,
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
    is_temporals = []
    test_progress = tqdm(
        test_dataset,
        desc="Evaluating Model (Report Generation)",
    )
    if num_beams is None:
        num_beams = 1

    print("******************")
    print("Beam Size", num_beams)
    print("******************")

    with torch.no_grad():
        for i, batch in enumerate(test_progress):
            max_length = max_tgt_length * 2
            min_length = 2
            patch_input_pixels = batch["patch_input_pixels"].cuda()
            patch_image_mask = batch["patch_image_mask"].cuda()
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            (
                patch_input_ids,
                patch_attention_mask,
            ) = inspect(batch, model, tokenizer, CONDITIONS)
            model_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "input_pixels": batch["input_pixels"].cuda(),
                "image_mask": batch["image_mask"].cuda(),
                "gather_index": batch["gather_index"].cuda(),
                "prior_input_pixels": batch["prior_input_pixels"].cuda(),
                "prior_image_mask": batch["prior_image_mask"].cuda(),
                "prior_gather_index": batch["prior_gather_index"].cuda(),
                "patch_input_ids": patch_input_ids,
                "patch_attention_mask": patch_attention_mask,
                "patch_input_pixels": patch_input_pixels,
                "patch_image_mask": patch_image_mask,
                "patch_labels": batch["patch_labels"][:, :1].cuda(),
                "patch_gather_index": batch["patch_gather_index"].cuda(),
                "num_beams": num_beams,
                "no_repeat_ngram_size": 0,
                "max_length": max_length,
                "min_length": min_length,
                "decoder_start_token_id": model.config.decoder_start_token_id,
                "bos_token_id": model.config.bos_token_id,
                "eos_token_id": model.config.eos_token_id,
                "pad_token_id": model.config.pad_token_id,
                "early_stopping": True,
                "return_dict_in_generate": True,
            }
            outputs = model.generate(**model_inputs)
            output_sequences = outputs["sequences"]
            labels = batch["labels"]
            report_id = batch["report_ids"]
            is_temporal = (batch["patch_image_mask"].sum(dim=-1) > 0).float()
            num_views = batch["image_mask"]
            num_views = accelerator.pad_across_processes(num_views, dim=1, pad_index=0)
            output_sequences = accelerator.pad_across_processes(
                output_sequences, dim=1, pad_index=tokenizer.pad_token_id
            )
            labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

            output_sequences, labels, num_views, is_temporal, report_id = (
                accelerator.gather_for_metrics(
                    (
                        output_sequences,
                        labels,
                        num_views,
                        is_temporal,
                        report_id,
                    )
                )
            )
            report_id = [str(report_ids_map[pos]) for pos in report_id.cpu().numpy()]

            labels = labels.masked_fill(labels == -100, tokenizer.pad_token_id).cpu()
            output_sequences = output_sequences.cpu()
            num_views = num_views.cpu().numpy()
            is_temporal = is_temporal.cpu().numpy()
            prediction = decoder_tokenizer.batch_decode(
                output_sequences, skip_special_tokens=False
            )
            reference = decoder_tokenizer.batch_decode(
                labels, skip_special_tokens=False
            )

            prediction = post_processing_multi(
                prediction,
                decoder_tokenizer.special_tokens_map.values(),
                decoder_tokenizer.unk_token,
            )
            reference = post_processing_multi(
                reference,
                decoder_tokenizer.special_tokens_map.values(),
                decoder_tokenizer.unk_token,
            )
            prediction = extend_sample(prediction, num_views)
            reference = extend_sample(reference, num_views)
            report_id = extend_sample(report_id, num_views)
            is_temporal = extend_sample(is_temporal, num_views)
            predictions.extend(prediction)
            references.extend(reference)
            report_ids.extend(report_id)
            is_temporals.extend(is_temporal)
            trainer.control = trainer.callback_handler.on_prediction_step(
                trainer.args, trainer.state, trainer.control
            )
    assert len(references) == len(predictions), "Prediction Num != Reference Num"
    accelerator.wait_for_everyone()
    print(len(set(report_ids)))
    bleu_scores = compute_scores(
        gts={index: [gt] for index, gt in enumerate(references)},
        res={index: [re] for index, re in enumerate(predictions)},
    )
    ce_scores = [0, 0, 0]
    if accelerator.is_main_process:
        with torch.no_grad():
            (
                ref_observations,
                hyp_observations,
                ce_scores,
                macro_ce_scores,
            ) = compute_ce_metric(
                references=references,
                hypotheses=predictions,
                chexbert=chexbert,
                bert_tokenizer=bert_tokenizer,
            )

            print("--------------------------------------------------------------")
            print(
                "Binary CE Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                % (ce_scores[0], ce_scores[1], ce_scores[2])
            )
            print(
                "Macro CE Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                % (macro_ce_scores[0], macro_ce_scores[1], macro_ce_scores[2])
            )
            temporal_references = [
                ref
                for ref, is_temporal in zip(references, is_temporals)
                if is_temporal == 1
            ]
            temporal_predictions = [
                pre
                for pre, is_temporal in zip(predictions, is_temporals)
                if is_temporal == 1
            ]
            (
                _,
                _,
                temporal_ce_scores,
                _,
            ) = compute_ce_metric(
                references=temporal_references,
                hypotheses=temporal_predictions,
                chexbert=chexbert,
                bert_tokenizer=bert_tokenizer,
                verbose=False,
            )
            tp = 0
            count_gen = 0
            count_ref = 0
            for ref, hyp in zip(temporal_references, temporal_predictions):
                ref_tem = set([z for z in ref.split() if z in TEM_keywords])
                hyp_tem = set([z for z in hyp.split() if z in TEM_keywords])
                tp += len(ref_tem & hyp_tem)
                count_gen += len(hyp_tem)
                count_ref += len(ref_tem)
            tem_prec = tp / max(count_gen, 1)
            tem_rec = tp / max(count_ref, 1)
            tem_f1 = 2 * tem_prec * tem_rec / max((tem_prec + tem_rec), 0.1)
            print(
                "Temporal Binary CE Score\t\t\t Prec. %0.4f\tRec. %0.4f\tF1 %0.4f"
                % (temporal_ce_scores[0], temporal_ce_scores[1], temporal_ce_scores[2])
            )
            print(
                "TEM\tPrec. %0.4f\tRec. %0.4f\tF1 %0.4f" % (tem_prec, tem_rec, tem_f1)
            )
        print("--------------------------------------------------------------")
        for i in range(5):
            print("Sample Prediction\t%d:" % i, predictions[i])
            print("Sample Reference\t%d:" % i, references[i])
        print("--------------------------------------------------------------")
        for score in bleu_scores:
            print("%s\t%0.4f" % (score, bleu_scores[score]))
        print("--------------------------------------------------------------")
        output_data = {}
        generated_obs = set()
        reference_obs = set()
        for sample_index, report_id, hyp, ref in zip(
            range(len(report_ids)), report_ids, predictions, references
        ):
            output_data[report_id] = {"hyp": hyp, "ref": ref}
            ref_obs, hyp_obs = [], []
            if len(ref_observations) > 0:
                ref_obs = ref_observations[sample_index]
                hyp_obs = hyp_observations[sample_index]
            output_data[report_id]["ref_obs"] = ref_obs
            output_data[report_id]["hyp_obs"] = hyp_obs
            reference_obs.add(",".join(sorted(ref_obs)))
            generated_obs.add(",".join(sorted(hyp_obs)))
        print("-------------------------------")
        print(
            "DISTINCT Generated Report %d, Avg Length %d"
            % (
                len(set(predictions)),
                sum([len(p.split()) for p in predictions]) / len(predictions),
            )
        )
        print(
            "DISTINCT Reference Report %d, Avg Length %d"
            % (
                len(set(references)),
                sum([len(r.split()) for r in references]) / len(references),
            )
        )
        print(
            "DISTINCT Generated Obs Sequence %d, Avg Length %d, Avg POS Length %d, Avg NEG Length %d"
            % (
                len(generated_obs),
                sum([len(p.split(",")) for p in generated_obs]) / len(generated_obs),
                sum(
                    [
                        len([z for z in p.split(",") if "Pos" in z])
                        for p in generated_obs
                    ]
                )
                / len(generated_obs),
                sum(
                    [
                        len([z for z in p.split(",") if "Neg" in z])
                        for p in generated_obs
                    ]
                )
                / len(generated_obs),
            )
        )
        print(
            "DISTINCT Reference Obs Sequence %d, Avg Length %d, Avg POS Length %d, Avg NEG Length %d"
            % (
                len(reference_obs),
                sum([len(r.split(",")) for r in reference_obs]) / len(reference_obs),
                sum(
                    [
                        len([z for z in p.split(",") if "Pos" in z])
                        for p in reference_obs
                    ]
                )
                / len(reference_obs),
                sum(
                    [
                        len([z for z in p.split(",") if "Neg" in z])
                        for p in reference_obs
                    ]
                )
                / len(reference_obs),
            )
        )
        print("-------------------------------")
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
                    f.write("Reference:\t%s\n" % ref)
                    f.write("Prediction:\t%s\n" % pre)
                    f3.write(pre + "\n")
                    f2.write(ref + "\n")
                f.write("****************\n")
                f.write("For All Report\n")
                for key, value in bleu_scores.items():
                    f.write(str(key) + ":\t" + str(value) + "\n")
            with open(
                os.path.join(output_path, result_file_name.replace("txt", "json")),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
    return bleu_scores
