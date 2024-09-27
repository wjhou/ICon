import collections
from typing import List, Optional
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Seq2SeqTrainer
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import logging
import torch
from optimizer import create_optimizer
from train_eval_ende_full import eval_text

logger = logging.get_logger(__name__)


class Seq2SeqTrainerGenMetrics(Seq2SeqTrainer):
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else self.args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        # if not self.is_in_train and self.args.fp16_full_eval:
        #     model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if self.args.past_index >= 0:
            self._past = None

        self.accelerator.prepare_model(model, evaluation_mode=True)

        args = self.args
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)
        metrics = eval_text(
            max_tgt_length=self.args.max_tgt_length,
            model=self.model,
            tokenizer=self.tokenizer,
            test_dataset=dataloader,
            output_path=self.args.output_dir,
            result_file_name="results_eval_step_%d.txt" % (self.state.global_step),
            reference_file_name="reference_eval_step_%d.txt" % (self.state.global_step),
            prediction_file_name="prediction_eval_step_%d.txt"
            % (self.state.global_step),
            num_beams=self.args.num_beams,
            compute_ce_metric=self.compute_ce_metric,
            chexbert=self.chexbert,
            bert_tokenizer=self.bert_tokenizer,
            decoder_tokenizer=self.decoder_tokenizer,
            accelerator=self.accelerator,
            trainer=self,
            report_ids_map=eval_dataset.index2report,
        )

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(eval_dataset),
        )
