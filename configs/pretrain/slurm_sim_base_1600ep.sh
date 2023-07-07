set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}
DATA_PATH=${6}
CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${PY_ARGS:-""}

BASENAME=`basename ${0} .sh`
DIR=./exp/pretrain/${BASENAME}
mkdir -p ${DIR}

TOTAL_BATCH_SIZE=4096
let BATCH_SIZE=${TOTAL_BATCH_SIZE}/${GPUS}

EPOCHS=1600

srun --partition=${PARTITION} \
  --mpi=pmi2 \
  --quotatype=${QUOTATYPE} \
  --job-name=${JOB_NAME} \
  -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=$CPUS_PER_TASK \
  --kill-on-bad-exit=1 \
  ${SRUN_ARGS} \
  python -u main_pretrain.py \
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
