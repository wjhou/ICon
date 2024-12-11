#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BertTokenizer,
    AutoTokenizer,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoConfig,
)
from collections import defaultdict
from transformers.utils import WEIGHTS_NAME
from transformers.trainer_utils import get_last_checkpoint
from data_collator_ende import DataCollatorForEnDe as DataCollatorForSeq2Seq
from dataset_ende import DatasetCustom
from seq2seqtrainer_metrics_ende import Seq2SeqTrainerGenMetrics
from train_eval_ende_full import train
from chexbert_eval import compute_ce_metric, load_chexbert
from sklearn.exceptions import UndefinedMetricWarning
import warnings
import sys

sys.path.append("../")
from src_stage1.data_arguments import DataTrainingArguments
from src_stage1.model_arguments import ModelArguments
from src_stage1.tokenizer import Tokenizer

warnings.filterwarnings(
    action="ignore", category=UndefinedMetricWarning, module="sklearn"
)

logger = logging.getLogger(__name__)


def load_stage1_output(path):
    with open(path, "r", encoding="utf-8") as f:
        eval_output = json.load(f)
        id2tags = {}
        for study_id in eval_output:
            # obs = [
            # o.split("_")[0] for o in eval_output[study_id]["obs_hyp"] if "POS" in o
            # ]
            obs = eval_output[study_id]["obs_hyp"]
            id2tags[study_id] = obs
    return id2tags


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    Seq2SeqTrainer = Seq2SeqTrainerGenMetrics

    data_args.dataset = (
        "mimic_abn"
        if "mimic_abn" in data_args.annotation_file
        else ("iu_xray" if "iu_xray" in data_args.annotation_file else "mimic_cxr")
    )

    logger.info("***************************")
    logger.info("***************************")
    logger.info(data_args)
    logger.info("***************************")
    logger.info("***************************")

    logger.info("***************************")
    logger.info("***************************")
    logger.info(model_args)
    logger.info("***************************")
    logger.info("***************************")

    # load necessary data
    with open(data_args.annotation_file, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    temporal_ids = None
    if data_args.temporal_file is not None:
        with open(data_args.temporal_file, "r", encoding="utf-8") as f:
            temporal_ids = json.load(f)

    if data_args.dataset == "iu_xray":
        train_idxs = {sample["id"] for sample in annotation["train"]}
    else:
        train_idxs = {
            sample["image_path"][0].split("/")[-1].split(".")[0]
            for sample in annotation["train"]
        }
    sentence_annotation = None
    if data_args.sentence_annotation_file is not None:
        with open(data_args.sentence_annotation_file, "r", encoding="utf-8") as f:
            sentence_annotation = json.load(f)

    with open(data_args.patch_annotation_file, "r", encoding="utf-8") as f:
        patch_annotation = json.load(f)

    with open(data_args.attribute_file, "r", encoding="utf-8") as f:
        attribute = json.load(f)
    attribute = {
        k: v[:30]
        for k, v in attribute.items()
        if "Positive" in k and "No Finding" not in k
    }
    merged_attribute = defaultdict(set)
    for k, v in attribute.items():
        merged_attribute[k.split(":")[0]].update(v)
    id2attribute_weight = defaultdict(dict)
    patch_stat = defaultdict(int)
    for study_id in patch_annotation["train"]:
        sample = patch_annotation["train"][study_id]
        for obs in sample:
            if obs == "aligned_entities":
                continue
            patch_stat[obs] += 1
            for entity in sample[obs]["aligned_entities"]:
                if entity not in merged_attribute[obs]:
                    continue
                if entity not in id2attribute_weight[obs]:
                    id2attribute_weight[obs][entity] = 0
                id2attribute_weight[obs][entity] += 1

    merged_attribute = {
        k: {z for z in v if z in id2attribute_weight[k]}
        for k, v in merged_attribute.items()
    }

    attribute = {k: sorted(list(v)) for k, v in merged_attribute.items()}
    attribute2id = {k: {a: i for i, a in enumerate(v)} for k, v in attribute.items()}
    id2attribute = {k: {i: a for i, a in enumerate(v)} for k, v in attribute.items()}

    import math

    for obs in id2attribute_weight:
        norm_factor = len(patch_annotation["train"])
        id2attribute_weight[obs] = {
            k: math.log(max((norm_factor - v) / max(v, 1), 1))
            for k, v in id2attribute_weight[obs].items()
        }

    weight = {}
    for obs in id2attribute_weight:
        weight[obs] = [0] * len(id2attribute_weight[obs])
        entities = attribute2id[obs].keys()
        for entity in entities:
            weight[obs][attribute2id[obs][entity]] = id2attribute_weight[obs][entity]

    if data_args.do_swap:
        with open(data_args.swap_annotation_file, "r", encoding="utf-8") as f:
            swap_annotation = json.load(f)

    data_args.threshold = {"mimic_abn": 3, "iu_xray": 3, "mimic_cxr": 10}[
        data_args.dataset
    ]
    id2tags, observation_category, observation_weight = Tokenizer.load_tag2ids(
        data_args.chexbert_label,
        need_header=True,
        train_idxs=train_idxs,
        flip=True,
    )
    word_tokenizer = Tokenizer(data_args)
    chexbert = load_chexbert(model_args.chexbert_model_name_or_path)
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    language_checkpoint = "facebook/bart-base"
    language_checkpoint = "GanjinZero/biobart-v2-base"
    tokenizer = AutoTokenizer.from_pretrained(language_checkpoint)
    special_tokens_dict = {"additional_special_tokens": []}
    for obs in observation_category:
        if "No Finding" in obs:
            continue
        special_tokens_dict["additional_special_tokens"].extend([f"<Positive {obs}>"])

    tokenizer.add_special_tokens(special_tokens_dict)
    model_config = AutoConfig.from_pretrained(language_checkpoint)
    tokenizer.id2attribute = id2attribute
    tokenizer.attribute2id = attribute2id

    processor = AutoFeatureExtractor.from_pretrained(
        model_args.vision_model_name_or_path
    )
    model_config.num_observation_ = 14
    model_config.num_observation = 14
    model_config.observation_weight = [0]

    model_config.num_view = (
        2 if data_args.dataset == "iu_xray" else 3
    ) + 1  # plus one for unknown view
    vision_encoder_checkpoint = model_args.vision_model_name_or_path
    model_config.pretrained_visual_extractor = vision_encoder_checkpoint
    model_config.checkpoint = vision_encoder_checkpoint
    model_config.stage1_model_name_or_path = model_args.vision_model_name_or_path
    model_config.prototype_stat = {
        observation_category.index(k): len(v) for k, v in id2attribute.items()
    }
    model_config.attribute_weight = {
        observation_category.index(k): v for k, v in weight.items()
    }
    model_config.observation_weight = observation_weight

    from models.modeling_bart import (
        ViTBartForConditionalGeneration as BartForConditionalGeneration,
    )

    model_class = BartForConditionalGeneration

    model_config.num_observation = len(observation_category)
    model_config.vit_checkpoint = model_args.vision_model_name_or_path
    model_config.num_prototype = data_args.num_prototype
    model = model_class.from_pretrained(language_checkpoint, config=model_config)
    model.generation_config.no_repeat_ngram_size = 0
    model.resize_token_embeddings(len(tokenizer))
    logger.info("***************************")
    logger.info("***** Model Structure *****")
    logger.info(model)
    logger.info("***************************")
    logger.info("***************************")
    train_dataset = eval_dataset = test_dataset = None
    if data_args.debug_model:
        for key in annotation:
            debug_data_size = 16  # if key == "train" else 100000
            keep_annotation = [ann for ann in annotation[key][:debug_data_size]]
            if key == "train" and swap_annotation is not None:
                swap_annotation[key] = {
                    k: v
                    for k, v in swap_annotation[key].items()
                    if (int(k) if "mimic" in data_args.dataset else k)
                    in [s["id"] for s in keep_annotation]
                }
                swap_ids = set()
                for idx in swap_annotation[key]:
                    if "mimic" in data_args.dataset:
                        swap_ids.update(map(int, swap_annotation[key][idx]))
                    else:
                        swap_ids.update(swap_annotation[key][idx])
                keep_annotation = keep_annotation + [
                    ann for ann in annotation[key] if ann["id"] in swap_ids
                ]
            annotation[key] = keep_annotation
    id2tags_train = {}
    id2tags_valid = {}
    id2tags_test = {}
    for split, id2tags_ in zip(
        ["train", "val", "test"], [id2tags_train, id2tags_valid, id2tags_test]
    ):
        for sample in annotation[split]:
            study_id = sample["id"]
            observation_id = (
                sample["image_path"][0].split("/")[-1].split(".")[0]
                if "mimic" in data_args.dataset
                else study_id
            )
            obs = [
                observation_category[index] + ("_POS" if o == 1 or o == -1 else "_NEG")
                for index, o in enumerate(id2tags[observation_id])
                if o == 1 or o == -1 or o == 0
            ]
            id2tags_[str(study_id)] = obs

    if training_args.do_train:
        transform = None
        print("*" * 20)
        print(
            "Image Augmentation is %s" % ("None" if transform is None else "Not None")
        )
        print("*" * 20)

        train_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            temporal_ids=temporal_ids,
            sentence_annotation=sentence_annotation,
            patch_annotation=patch_annotation,
            swap_annotation=swap_annotation,
            split="train",
            tokenizer=tokenizer,
            word_tokenizer=word_tokenizer,
            processor=processor,
            transform=transform,
            observation_category=observation_category,
            id2tags=id2tags_train,
        )
        id2tags = load_stage1_output(
            os.path.join(
                model_args.stage1_model_name_or_path, model_args.stage1_eval_output
            )
        )
        eval_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            temporal_ids=temporal_ids,
            sentence_annotation=None,
            patch_annotation=patch_annotation,
            swap_annotation=None,
            split="valid",
            tokenizer=tokenizer,
            word_tokenizer=word_tokenizer,
            processor=processor,
            observation_category=observation_category,
            id2tags=id2tags,
            id2tags_valid=id2tags_valid,
        )
    if training_args.do_predict:
        id2tags = load_stage1_output(
            os.path.join(model_args.stage1_model_name_or_path, "results.json")
        )
        test_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            temporal_ids=temporal_ids,
            sentence_annotation=None,
            patch_annotation=patch_annotation,
            swap_annotation=None,
            split="test",
            tokenizer=tokenizer,
            word_tokenizer=word_tokenizer,
            processor=processor,
            observation_category=observation_category,
            id2tags=id2tags,
            id2tags_valid=id2tags_test,
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=data_args.max_context_length,
        pad_to_multiple_of=8,
        observation_pad_id=len(observation_category) - 1,
    )

    training_args.max_tgt_length = data_args.max_tgt_length
    training_args.num_beams = model_args.num_beams
    training_args.fast_lr = model_args.fast_lr
    training_args.remove_unused_columns = False
    data_args.max_steps = training_args.max_steps

    from transformers import EarlyStoppingCallback

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=4)]
            if data_args.dataset == "mimic_abn"
            else None
        ),
    )
    trainer.tokenizer = tokenizer
    trainer.data_args = data_args
    trainer.chexbert = chexbert
    trainer.bert_tokenizer = bert_tokenizer
    trainer.compute_ce_metric = compute_ce_metric
    trainer.decoder_tokenizer = tokenizer

    if training_args.do_train:
        logger.info("*** Train ***")
        train(
            training_args,
            data_args,
            last_checkpoint,
            trainer,
            train_dataset,
        )

    # Prediction
    if training_args.do_predict:
        logger.info("*** Test ***")
        if model_args.test_model_name_or_path is not None:
            logger.info(
                "*** Test: Loading %s ***" % (model_args.test_model_name_or_path)
            )
            from safetensors.torch import load_file

            # state_dict = load_file(
            state_dict = torch.load(
                os.path.join(
                    model_args.test_model_name_or_path,
                    # SAFE_WEIGHTS_NAME,
                    WEIGHTS_NAME,
                ),
            )
            model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        # model = model.to(dtype=torch.float16, device=trainer.args.device)
        from train_eval_ende_full import eval_text

        print(model_args.num_beams)
        eval_text(
            max_tgt_length=data_args.max_tgt_length,
            model=model,
            test_dataset=trainer.get_test_dataloader(test_dataset),
            output_path=training_args.output_dir,
            num_beams=model_args.num_beams,
            compute_ce_metric=compute_ce_metric,
            chexbert=chexbert,
            bert_tokenizer=bert_tokenizer,
            decoder_tokenizer=tokenizer,
            tokenizer=tokenizer,
            accelerator=trainer.accelerator,
            validation=False,
            trainer=trainer,
            report_ids_map=test_dataset.index2report,
        )


if __name__ == "__main__":
    main()
