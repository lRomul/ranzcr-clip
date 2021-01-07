cd $(dirname "${BASH_SOURCE[0]}")

make GPUS="device=$2" COMMAND="python train.py --experiment $1" NAME="ranzcr-clip-$1"
