set -x

GPUS=${1}
GPUS_PER_NODE=${2}
QUOTATYPE=${3}
PARTITION=${4}
CKPT_PATH=${5}
DATA_PATH=${6}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}


TOTAL_BATCH_SIZE=16384
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${GPUS}

BASENAME=$(basename ${CKPT_PATH})
EXP_NAME=$(basename $(dirname ${CKPT_PATH}))
DIR=./exp/linear/${EXP_NAME}
JOB_NAME=lin-${EXP}

mkdir -p ${DIR}

srun --partition=${PARTITION} \
  --mpi=pmi2 \
  --open-mode=append \
  --quotatype=${QUOTATYPE} \
  --job-name=${JOB_NAME} \
  -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=$CPUS_PER_TASK \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python -u main_linprobe.py \
    --batch_size ${BATCH_SIZE} \
    --model vit_base_patch16 \
    --finetune ${CKPT_PATH} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --global_pool \
    --data_path ${DATA_PATH} \
    --use_tcs_dataset \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout.txt