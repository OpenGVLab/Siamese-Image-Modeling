# Finetune

We provide the finetuning scripts here. To finetune a SiameseIM model, it is recommended that
* use 1024 batch size, which should fit into 8 V100 gpus with 32G memory;
* We provide the finetuned checkpoint in [checkpoints.md](./checkpoints.md).

## Train with torch.distributed.launch
This method supports training on multi-nodes with torch.distributed.launch. For example, to finetune a SiameseIM model on 2 nodes, run the command below.

On node 1:
```
  sh ./configs/finetune/dist_finetune_sim_base.sh ${MASTER_ADDR} 0 2 ${CKPT_PATH} ${DATA_PATH}
```

On node 2:
```
  sh ./configs/finetune/dist_finetune_sim_base.sh ${MASTER_ADDR} 1 2 ${CKPT_PATH} ${DATA_PATH}
```

Note:
The `${MASTER_ADDR}` is the ip address of rank 0 node. The second and third arguments specify the node rank and node number respectively. You need to adjust them if different node numbders are used.

## Train on a slurm cluster
If you need to run the finetuning on a slurm cluster, use the command below to run on `${GPUS}/${GPUS_PER_NODE}` nodes with `${GPUS_PER_NODE}` gpus on each node:
```
  sh ./configs/finetune/slurm_finetune_sim_base.sh ${GPUS} ${GPUS_PER_NODE} ${QUOTATYPE} ${PARTITION} ${CKPT_PATH} ${DATA_PATH}
```

## Evaluation
We also provide the evaluation scripts as follows.

For torch.distributed.launch, use
```
  sh ./configs/finetune/dist_finetune_sim_base_eval.sh ${MASTER_ADDR} 0 1 ${CKPT_PATH} ${DATA_PATH}
```

For slurm launch, use
```
  sh ./configs/finetune/slurm_finetune_sim_base_eval.sh ${GPUS} ${GPUS_PER_NODE} ${QUOTATYPE} ${PARTITION} ${CKPT_PATH} ${DATA_PATH}
```
You should get
```
* Acc@1 84.118 Acc@5 96.766 loss 0.728
```
for the provided checkpoint.
