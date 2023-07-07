set -x

GPUS=${1}
GPUS_PER_NODE=${2}
QUOTATYPE=${3}
PARTITION=${4}
CKPT_PATH=${5}
DATA_PATH=${6}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}

BASENAME=$(basename ${CKPT_PATH})
EXP_NAME=$(basename $(dirname ${CKPT_PATH}))
DIR=./exp/fewshot/${EXP_NAME}
JOB_NAME=fewshot-${EXP}

srun --partition=${PARTITION} \
  --mpi=pmi2 \
  --quotatype=${QUOTATYPE} \
  --job-name=${JOB_NAME} \
  -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=$CPUS_PER_TASK \
  --kill-on-bad-exit=1 \
  python -W ignore -u main_logistic.py \
    --subset-path imagenet_subset1/1percent.txt \
    --root-path ${DATA_PATH} \
    --image-folder imagenet_full_size/061417/ \
    --device cuda:0 \
    --pretrained ${CKPT_PATH} \
    --fname 'fewshot_1percent.pth' \
    --model-name 'vit_base_patch16' \
    --penalty l2 \
    --lambd 0.1