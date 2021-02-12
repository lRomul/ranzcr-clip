#!/bin/bash

EXP=${1}
DEVICE=${2:-all}
FOLDS=${3:-all}
TASK=${4:-cls}

echo "EXP=$EXP, DEVICE=$DEVICE, FOLDS=$FOLDS, TASK=$TASK"

cd $(dirname "${BASH_SOURCE[0]}") &&\
git fetch --all &&\
git reset --hard "$1" &&\
make GPUS="'\"device=$DEVICE\"'" \
  COMMAND="python -m torch.distributed.launch
  --nproc_per_node=${nvidia-smi --query-gpu=name --format=csv,noheader | wc -l}
  train_$TASK.py --experiment $1 --folds $FOLDS" \
  NAME="ranzcr-clip-$EXP-${FOLDS//[,]/}"
