#!/bin/sh
GPU_ID=0,1,2,3
EXP=exp_coco
RUN_FILE=train_l2g_sal_coco.py
BASH_FILE=train_l2g_sal_coco.sh
CROP_SIZE=320
PATCH_NUM=4
EPOCH=15
BATCH_SIZE=11
DATASET=mscoco
INPUT_SIZE=448
NUM_CLASSES=80
LR=0.1
BG_THR=0.1
KD_WEIGHT=30
DECAY_POINTS='5,10'
mkdir -p runs/${EXP}/model/
cp ${BASH_FILE} runs/${EXP}/model/${BASH_FILE}
cp scripts/${RUN_FILE} runs/${EXP}/model/${RUN_FILE} 


CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ./scripts/${RUN_FILE} \
    --img_dir=./data/coco14/ \
    --train_list=./data/coco14/train_cls.txt \
    --test_list=./data/coco14/val_cls.txt \
    --epoch=${EPOCH} \
    --lr=${LR} \
    --batch_size=${BATCH_SIZE} \
    --dataset=${DATASET} \
    --input_size=${INPUT_SIZE} \
    --crop_size=${CROP_SIZE} \
    --disp_interval=100 \
    --num_classes=${NUM_CLASSES} \
    --num_workers=8 \
    --patch_num=${PATCH_NUM} \
    --snapshot_dir=./runs/${EXP}/model/  \
    --att_dir=./runs/${EXP}/  \
    --kd_weights=${KD_WEIGHT} \
    --bg_thr=${BG_THR} \
    --poly_optimizer
