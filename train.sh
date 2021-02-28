#!/bin/bash

EXP=${1}
N_DEVICE=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
DEVICE=${3:-all}
FOLDS=${4:-all}

echo "EXP=$EXP, N_DEVICE=$N_DEVICE, DEVICE=$DEVICE, FOLDS=$FOLDS"

cd "$(dirname "${BASH_SOURCE[0]}")" &&\
git fetch --all &&\
git reset --hard "$EXP" &&\
make GPUS="'\"device=$DEVICE\"'" \
  COMMAND="python -m torch.distributed.launch --nproc_per_node=$N_DEVICE train_cls.py --experiment $EXP --folds $FOLDS" \
  NAME="ranzcr-clip-$EXP-${FOLDS//[,]/}"
