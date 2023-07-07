set -x

GPUS=${1}
GPUS_PER_NODE=${2}
JOB_NAME=${3}
QUOTATYPE=${4}
PARTITION=${5}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}

DIR=./exp/semisup_sim_large_1600ep
CKPT=./exp/pretrain_sim_large_1600ep/checkpoint-latest.pth

srun --partition=vc_research_${PARTITION} \
  --mpi=pmi2 \
  --quotatype=${QUOTATYPE} \
  --job-name=${JOB_NAME} \
  -n$GPUS \
  --gres=gpu:${GPUS_PER_NODE} \
  --ntasks-per-node=${GPUS_PER_NODE} \
  --cpus-per-task=$CPUS_PER_TASK \
  --kill-on-bad-exit=1 \
  --dependency=singleton \
  -x SH-IDC1-10-142-5-[45,13,70,198],SH-IDC1-10-142-4-[187,93,188,46,165,83,151,146,26] \
  python -W ignore -u main_logistic.py \
    --subset-path imagenet_subset1/1percent.txt \
    --root-path /mnt/cache/share/images \
    --image-folder imagenet_full_size/061417/ \
    --device cuda:0 \
    --pretrained ${CKPT} \
    --fname 'semisup.pth' \
    --model-name 'vit_large_patch16' \
    --penalty l2 \
    --lambd 0.01