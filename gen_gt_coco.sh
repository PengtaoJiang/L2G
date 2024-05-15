#!/bin/sh
EXP=exp_coco
TYPE=ms

CUDA_VISIBLE_DEVICES=0 python3 gen_gt.py \
   --dataset=mscoco \
   --datalist=./data/coco14/train.txt \
   --gt_dir=./data/coco14/JPEGImages/ \
   --save_path=./data/coco14/pseudo_seg_labels/train2014/ \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/ \
   --thr=0.25 \
   --num_workers=16