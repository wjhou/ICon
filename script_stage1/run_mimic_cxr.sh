#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=false
suffix=""
warmup_ratio=0.0
max_tgt_length=64
num_train_epochs=5
overwrite_output_dir=false
evaluation_strategy=epoch
per_device_train_batch_size=32
per_device_eval_batch_size=32
gradient_accumulation_steps=4
debug_model=false
seed=42
num_beams=4
slow_lr=1e-4
fast_lr=1e-4
weight_decay=0.0
date=$2
image_size=256
dataloader_num_workers=8
log_level="info"
report_to="none"
model_name="swinv2"
vision_model="microsoft/swinv2-small-patch4-window8-256"
chexbert_label="./data/mimic_cxr/id2tag.csv"
annotation_file="./data/mimic_cxr/annotation.json"
image_path="./data/mimic_cxr/images"
output_dir="./tmp_stage1/mimic_cxr_stage1_${model_name}_image${image_size}_${date}/"

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    max_steps=3
    num_train_epochs=1
    eval_steps=1
    save_steps=1
    output_dir="./tmp_stage1/mimic_cxr_stage1_debug"
    overwrite_output_dir=true
    debug_model=true
    report_to="none"
fi

export TOKENIZERS_PARALLELISM=true
python3 -u ./src_stage1/run_ende.py \
    --chexbert_model_name_or_path ./CheXbert/chexbert.pth \
    --vision_model $vision_model \
    --annotation_file $annotation_file \
    --image_path $image_path \
    --chexbert_label $chexbert_label \
    --image_size $image_size \
    --do_train \
    --do_eval \
    --do_predict \
    --log_level $log_level \
    --per_device_train_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --max_tgt_length $max_tgt_length \
    --output_dir $output_dir \
    --warmup_ratio $warmup_ratio \
    --num_train_epochs $num_train_epochs \
    --learning_rate $slow_lr \
    --fast_lr $fast_lr \
    --weight_decay $weight_decay \
    --evaluation_strategy $evaluation_strategy \
    --save_strategy $evaluation_strategy \
    --save_total_limit 1 \
    --seed $seed \
    --logging_steps 100 \
    --report_to $report_to \
    --fp16 \
    --fp16_opt_level O3 \
    --fp16_full_eval \
    --dataloader_num_workers $dataloader_num_workers \
    --load_best_model_at_end true \
    --overwrite_output_dir $overwrite_output_dir \
    --group_by_length false \
    --length_column_name length \
    --eval_on_gen \
    --greater_is_better true \
    --metric_for_best_model eval_score \
    --debug_model $debug_model \
    --num_beams $num_beams
