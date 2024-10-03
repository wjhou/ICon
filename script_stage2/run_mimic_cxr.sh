#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export LOCAL_RANK=0,1,2,3
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=false
suffix=""
warmup_ratio=0.0
max_tgt_length=104
overwrite_output_dir=false
evaluation_strategy=epoch
per_device_train_batch_size=8 # batch size 64
per_device_eval_batch_size=16
gradient_accumulation_steps=1
debug_model=false
seed=42
num_beams=4
slow_lr=5e-5
fast_lr=5e-5
weight_decay=0.0
dataloader_num_workers=12
log_level="info"
report_to="none"
image_size=256
vision_model="microsoft/swinv2-tiny-patch4-window8-256"
attribute_file="./data/mimic_cxr/triples.json"
chexbert_label="./data/mimic_cxr/id2tag.csv"
annotation_file="./data/mimic_cxr/annotation.json"
temporal_file="./data/mimic_cxr/temporal_ids.json"
patch_annotation_file="./data/mimic_cxr/aligned_patch_annotation.json"
swap_annotation_file="./data/mimic_cxr/swap_patch_annotation.json"
image_path="./data/mimic_cxr/images"
patch_path="./data/mimic_cxr/"
num_prototype=30
num_train_epochs=5
nproc_per_node=4 # 4 GPUs
master_port=25642
model_name="swinv2"
date=$2
stage1_model_name_or_path="./tmp_stage1/mimic_cxr_stage1_${model_name}_image${image_size}_${date}/"
stage1_eval_output="results_eval_step_5945.json"
output_dir="./tmp_stage2/mimic_cxr_stage2_${model_name}_image${image_size}_${date}/"

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    suffix="_debug"
    num_train_epochs=1
    output_dir="./tmp_stage2/mimic_cxr_stage2_debug"
    overwrite_output_dir=true
    debug_model=true
    report_to="none"
    master_port=25643
fi

torchrun --standalone --nproc_per_node=$nproc_per_node --nnodes=1 --master_port=$master_port ./src_stage2/run_ende.py \
    --chexbert_model_name_or_path ../CheXbert/chexbert.pth \
    --vision_model_name_or_path $vision_model \
    --stage1_model_name_or_path $stage1_model_name_or_path \
    --stage1_eval_output $stage1_eval_output \
    --annotation_file $annotation_file \
    --attribute_file $attribute_file \
    --temporal_file $temporal_file \
    --patch_annotation_file $patch_annotation_file \
    --swap_annotation_file $swap_annotation_file \
    --image_path $image_path \
    --patch_path $patch_path \
    --chexbert_label $chexbert_label \
    --image_size $image_size \
    --num_prototype $num_prototype \
    --do_swap \
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
    --save_safetensors false \
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
    --save_on_each_node \
    --metric_for_best_model eval_BLEU_4 \
    --debug_model $debug_model \
    --num_beams $num_beams
