set -x

IP=${1}
RANK=${2}
NNODES=${3}
DATA_PATH=${4}
PORT=${PORT:-28500}
PY_ARGS=${PY_ARGS:-""}

BASENAME=`basename ${0} .sh`
DIR=./exp/pretrain/${BASENAME}
mkdir -p ${DIR}

TOTAL_BATCH_SIZE=4096
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${NNODES}/8

EPOCHS=1600

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=${NNODES} --node_rank=${RANK} --master_addr=${IP} --master_port=${PORT} \
    main_pretrain.py \
    --model sim_vit_base_patch16 \
    --decoder_embed_dim 768 \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs 40 \
    --crop_min 0.08 \
    --with_blockwise_mask \
    --blockwise_num_masking_patches 118 \
    --blr 6.25e-5 --weight_decay 0.05 \
    --mm 0.995 \
    --mmschedule 'cosine' \
    --clip_grad 1.0 \
    --loss_type 'sim' \
    --neg_weight 0.02 \
    --save_latest_freq 5 \
    --output_dir ${DIR} \
    --log_dir ${DIR} \
    --data_path ${DATA_PATH} \
    ${PY_ARGS} 2>&1 | tee -a ${DIR}/stdout.txt
