#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_OFFLINE=true
version="20240101"
dataset="mimic_cxr"
# path to mimic-cxr-jpg
image_path="./data/$dataset/images"
# image size
patch_szie=384

patch_path="./data/$dataset/aligned_patch_annotation.json"
output_dir="./data/$dataset/"
chexbert_label="./data/$dataset/id2tag.csv"
annotation_path="./data/$dataset/annotation.json"
retrieval_path="./data/$dataset/$version/pseudo_reference_updated.json"
model_path="./tmp_stage1/mimic_cxr_stage1_swinv2_image256_${version}"
triple_path="./data/$dataset/triples.json"
sentence_annotation="./data/$dataset/sentence_level_observation.json"
patch_annotation="./data/$dataset/patch_annotation.json"
best_checkpoint="5945"
batch_size=16

# rm -rf $output_dir

python3 -u ./src_xai/run_select.py \
    --dataset $dataset \
    --image_path $image_path \
    --patch_size $patch_szie \
    --output_dir $output_dir \
    --chexbert_label $chexbert_label \
    --model_path $model_path \
    --best_checkpoint $best_checkpoint \
    --batch_size $batch_size

python3 -u ./src_xai/run_label_patch.py \
    --dataset $dataset \
    --triple_path $triple_path \
    --sentence_annotation $sentence_annotation \
    --patch_annotation $patch_annotation \
    --report_annotation $annotation_path \
    --output_dir $patch_path

output_dir="./data/$dataset/swap_patch_annotation_updated.json"
python3 -u ./src_xai/run_swap.py \
    --dataset $dataset \
    --annotation_path $annotation_path \
    --patch_path $patch_path \
    --retrieval_path $retrieval_path \
    --output_dir $output_dir