#!/bin/sh
EXP=exp_coco
TYPE=ms

CUDA_VISIBLE_DEVICES=0 python3 gen_gt.py \
   --dataset=mscoco \
   --datalist=data/coco14/train_cls.txt \
   --gt_dir=/root/data/jpt/coco14/SegmentationClass/ \
   --save_path=/root/data/jpt/coco14/pseudo_seg_labels \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/ \
   --num_workers=16