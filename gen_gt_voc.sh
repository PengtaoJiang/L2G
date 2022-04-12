#!/bin/sh
EXP=exp_voc
TYPE=ms

CUDA_VISIBLE_DEVICES=0 python3 gen_gt.py \
   --dataset=mscoco \
   --datalist=data/voc12/train_aug.txt \
   --gt_dir=/root/data/jpt/VOCdevkit/VOC2012/JPEGImages/ \
   --save_path=/root/data/jpt/VOCdevkit/VOC2012/pseudo_seg_labels/ \
   --pred_dir=./runs/${EXP}/${TYPE}/attention/ \
   --num_workers=16