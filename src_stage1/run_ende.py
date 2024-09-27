#!/usr/bin/env python
# coding=utf-8
import json
import logging
import os
import sys

import datasets
import torch
import transformers
from torchvision import transforms
from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    BartConfig,
    CLIPFeatureExtractor,
    AutoFeatureExtractor,
    SwinConfig,
)
from transformers.utils import WEIGHTS_NAME, SAFE_WEIGHTS_NAME
from transformers.trainer_utils import get_last_checkpoint
from data_arguments import DataTrainingArguments
from data_collator_ende import DataCollatorForEnDeStage1 as DataCollatorForSeq2Seq
from dataset_ende import DatasetCustom
from model_arguments import ModelArguments
from seq2seqtrainer_metrics_ende import Seq2SeqTrainerGenMetrics
from train_eval_ende_full import train
from transformers import ViTFeatureExtractor

from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings(
    action="ignore", category=UndefinedMetricWarning, module="sklearn"
)

logger = logging.getLogger(__name__)


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

    from tokenizer import Tokenizer

    data_args.dataset = (
        "mimic_abn"
        if "mimic_abn" in data_args.annotation_file
        else ("mimic_cxr" if "mimic_cxr" in data_args.annotation_file else "iu_xray")
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

    with open(data_args.annotation_file, "r", encoding="utf-8") as f:
        annotation = json.load(f)
    if data_args.dataset == "iu_xray":
        train_idxs = {sample["id"] for sample in annotation["train"]}
    else:
        train_idxs = {
            sample["image_path"][0].split("/")[-1].split(".")[0]
            for sample in annotation["train"]
        }
    # observation labels
    id2tags, observation_category, observation_weight = Tokenizer.load_tag2ids(
        data_args.chexbert_label,
        need_header=True,
        train_idxs=train_idxs,
    )
    _, _, normal_weight = Tokenizer.load_tag2ids(
        data_args.chexbert_label,
        need_header=True,
        train_idxs=train_idxs,
        flip=True,
    )

    checkpoint = "GanjinZero/biobart-base"
    # config = BartConfig.from_pretrained(checkpoint)
    config = SwinConfig.from_pretrained(model_args.vision_model_name_or_path)
    config.num_observation = len(observation_category)
    config.observation_category = observation_category
    checkpoint = model_args.vision_model_name_or_path
    config.pretrained_visual_extractor = checkpoint
    config.checkpoint = checkpoint
    data_args.return_raw_img = False
    processor = AutoFeatureExtractor.from_pretrained(
        checkpoint, size=data_args.image_size
    )
    from models.modeling_swin import VisualEncoder

    config.observation_weight = observation_weight
    config.normal_weight = normal_weight[:-1]
    model = VisualEncoder(config=config)
    logger.info("***************************")
    logger.info("***** Model Structure *****")
    logger.info(model)
    logger.info("***************************")
    logger.info("***************************")
    train_dataset = eval_dataset = test_dataset = None

    if data_args.debug_model:
        debug_data_size = 64
        for key in annotation:
            annotation[key] = annotation[key][:debug_data_size]

    if training_args.do_train:
        transform = transforms.Compose(
            [
                transforms.Resize(data_args.image_size + 32),
                transforms.RandomCrop(data_args.image_size),
                transforms.RandomHorizontalFlip(),
            ]
        )
        # transform = None
        train_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="train",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
            transform=transform,
        )
        eval_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="valid",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
        )
    if training_args.do_predict:
        test_dataset = DatasetCustom(
            data_args=data_args,
            annotation=annotation,
            split="test",
            id2tags=id2tags,
            processor=processor,
            observation_category=observation_category,
        )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=None,
        model=model,
        padding=True,
        max_length=data_args.max_context_length,
        pad_to_multiple_of=8,
    )

    training_args.max_tgt_length = data_args.max_tgt_length
    training_args.num_beams = model_args.num_beams
    training_args.fast_lr = model_args.fast_lr
    training_args.remove_unused_columns = False
    data_args.max_steps = training_args.max_steps

    from transformers import EarlyStoppingCallback

    patience = {
        "iu_xray": 10,
        "mimic_abn": 5,
        "mimic_cxr": 3,
    }
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=patience[data_args.dataset],
            )
        ],
    )
    trainer.data_args = data_args
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

            state_dict = load_file(
                os.path.join(
                    model_args.test_model_name_or_path,
                    SAFE_WEIGHTS_NAME,
                ),
            )
            model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
        from train_eval_ende_full import eval_text

        print(model_args.num_beams)
        eval_text(
            max_tgt_length=data_args.max_tgt_length,
            model=model,
            test_dataset=trainer.get_test_dataloader(test_dataset),
            output_path=training_args.output_dir,
        )


if __name__ == "__main__":
    main()
