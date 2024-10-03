#!/bin/sh
version="20240101"
dataset="mimic_cxr"
output_dir="./data/$dataset/$version/"
annotation_path="./data/$dataset/annotation.json"
radgraph_path="./data/radgraph/1.0.0/MIMIC-CXR_graphs.json"
output_path="./data/"
chexbert_label="./data/$dataset/id2tag.csv"
cluster_path="./data/$dataset/cluster.json"
temporal_file="/data/mimic_cxr/temporal_ids.json"
mkdir -p $output_dir$dataset

echo "================================================"
echo "Retrieving Pseudo Reference from $dataset"
echo "================================================"
python -u ./src_retrieval/retrieval.py \
    --annotation_path $annotation_path \
    --output_path $output_dir$dataset/pseudo_reference_updated.json \
    --chexbert_label $chexbert_label \
    --cluster_path $cluster_path \
    --temporal_file $temporal_file