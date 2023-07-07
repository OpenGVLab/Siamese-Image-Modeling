# Pretrain

We provide the pretraining scripts here. To pretrain a SiameseIM model, it is recommended that
* use 4096 batch size, which should fit into 32 V100 gpus with 32G memory;
* pretrain for 1600 epochs for better performance. We also note that pretraining SiameseIM for 400 epochs can already match the performances of 1600 epoch MAE on some tasks;
* We provide the 1600 epoch pretrained checkpoint in [checkpoints.md](./checkpoints.md).

## Train with torch.distributed.launch
This method supports training on multi-nodes with torch.distributed.launch. For example, to pretrain a SiameseIM model on 2 nodes, run the command below.

On node 1:
```
  sh ./configs/pretrain/dist_sim_base_1600ep.sh ${MASTER_ADDR} 0 2 ${DATA_PATH}
```

On node 2:
```
  sh ./configs/pretrain/dist_sim_base_1600ep.sh ${MASTER_ADDR} 1 2 ${DATA_PATH}
```

Note:
The `${MASTER_ADDR}` is the ip address of rank 0 node. The second and third arguments specify the node rank and node number respectively. You need to adjust them if different node numbders are used.

## Train on a slurm cluster
If you need to run the pretraining on a slurm cluster, use the command below to run on `${GPUS}/${GPUS_PER_NODE}` nodes with `${GPUS_PER_NODE}` gpus on each node:
```
  sh ./configs/pretrain/slurm_sim_base_1600ep.sh ${GPUS} ${GPUS_PER_NODE} ${JOB_NAME} ${QUOTATYPE} ${PARTITION} ${DATA_PATH}
```
