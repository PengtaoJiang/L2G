#!/bin/sh
EXP=exp_voc
RUN_FILE=train_l2g_sal_voc.py
BASH_FILE=train_l2g_sal_voc.sh
GPU_ID=0
CROP_SIZE=320
PATCH_NUM=6

mkdir -p runs/${EXP}/model/
cp ${BASH_FILE} runs/${EXP}/model/${BASH_FILE}
cp scripts/${RUN_FILE} runs/${EXP}/model/${RUN_FILE} 


CUDA_VISIBLE_DEVICES=0,1 python3 ./scripts/${RUN_FILE} \
    --img_dir=./data/voc12/ \
    --train_list=./data/voc12/train_cls.txt \
    --test_list=./data/voc12/val_cls.txt \
    --epoch=10 \
    --lr=0.001 \
    --batch_size=3 \
    --iter_size=1 \
    --dataset=pascal_voc \
    --input_size=448 \
    --crop_size=${CROP_SIZE} \
    --disp_interval=100 \
    --num_classes=20 \
    --num_workers=8 \
    --patch_size=${PATCH_NUM} \
    --snapshot_dir=./runs/${EXP}/model/  \
    --att_dir=./runs/${EXP}/  \
    --decay_points='5' \
    --kd_weights=10 \
    --bg_thr=0.001
    # --load_checkpoint="./runs/${EXP}/model/pascal_voc_epoch_9.pth" \
    # --current_epoch=10
