#!/bin/sh
EXP=exp_voc
TYPE=ms

CUDA_VISIBLE_DEVICES=0 python3 gen_gt.py \
   --dataset=mscoco \
   --datalist=data/voc12/train_aug.txt \
   --gt_dir=./data/voc12/JPEGImages/ \
   --save_path=./data/voc12/pseudo_seg_labels/ \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/ \
   --num_workers=16