#!/bin/bash
DATA=/mnt/truenas/scratch/hzh/datasets/imagenet_raw
python3 -m torch.distributed.launch --nproc_per_node=8 train_manu.py  $DATA  --sched step --epochs 30 \
--warmup-epochs 5 --lr 0.024 --reprob 0.5 --remode pixel --batch-size 256 --amp -j 8 --warmup-lr 1e-6 \
--weight-decay 1e-5 --opt rmsproptf --opt-eps 0.001  --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 \
--remode pixel --reprob 0.2 --amp --color-jitter 0.06  --decay-epochs 3 --decay-rate 0.963
# DNA training
