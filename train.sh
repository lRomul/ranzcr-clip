#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")" || exit

EXP=${1}
N_DEVICE=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}
DEVICE=${3:-all}
FOLDS=${4:-0,1,2,3,4}

echo "EXP=$EXP, N_DEVICE=$N_DEVICE, DEVICE=$DEVICE, FOLDS=$FOLDS"

for fold in ${FOLDS//[,]/ }
do
  git fetch --all &&\
  git reset --hard "$EXP" &&\
  make GPUS="'\"device=$DEVICE\"'" \
    COMMAND="python -m torch.distributed.launch --nproc_per_node=$N_DEVICE train_cls.py --experiment $EXP --folds $fold" \
    NAME="ranzcr-clip-$EXP-$fold"
done
