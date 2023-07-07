set -x

IP=${1}
RANK=${2}
NNODES=${3}
CKPT_PATH=${4}
DATA_PATH=${5}
PORT=${PORT:-28500}
PY_ARGS=${PY_ARGS:-""}

TOTAL_BATCH_SIZE=16384
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${NNODES}/8

BASENAME=$(basename ${CKPT_PATH})
EXP_NAME=$(basename $(dirname ${CKPT_PATH}))
DIR=./exp/linear/${EXP_NAME}

mkdir -p ${DIR}

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${NNODES} --node_rank=${RANK} --master_addr=${IP} --master_port=${PORT} \
    main_linprobe.py \
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