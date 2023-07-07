set -x

IP=${1}
RANK=${2}
NNODES=${3}
CKPT_PATH=${4}
DATA_PATH=${5}
PORT=${PORT:-28500}
PY_ARGS=${PY_ARGS:-""}

BASENAME=$(basename ${CKPT_PATH})
EXP_NAME=$(basename $(dirname ${CKPT_PATH}))
DIR=./exp/fewshot/${EXP_NAME}

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${NNODES} --node_rank=${RANK} --master_addr=${IP} --master_port=${PORT} \
    main_logistic.py \
    --subset-path imagenet_subset1/1percent.txt \
    --root-path ${DATA_PATH} \
    --image-folder imagenet_full_size/061417/ \
    --device cuda:0 \
    --pretrained ${CKPT_PATH} \
    --fname 'fewshot_1percent.pth' \
    --model-name 'vit_base_patch16' \
    --penalty l2 \
    --lambd 0.1 \
    --preload