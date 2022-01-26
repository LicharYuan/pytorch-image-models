#!/bin/bash
SAVE_FILE=$1
ARCH=$2
DATA=/mnt/truenas/scratch/hzh/datasets/imagenet_raw
python3 -m torch.distributed.launch --nproc_per_node=8 train_spos.py  $DATA --arch $ARCH --save_file $SAVE_FILE   --sched step --epochs 30 \
--warmup-epochs 5 --lr 0.016 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 8 --warmup-lr 1e-3 \
--weight-decay 1e-5 --opt rmsproptf --opt-eps 0.001 --aa rand-m9-mstd0.5 \
--remode pixel --reprob 0.2 --amp --color-jitter 0.06  --decay-epochs 3 --decay-rate 0.963
