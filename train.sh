cd $(dirname "${BASH_SOURCE[0]}") &&\
git fetch --all &&\
git reset --hard origin/"$1" &&\
make GPUS="'\"device=$3\"'" COMMAND="python train_$2.py --experiment $1" NAME="ranzcr-clip-$1"
