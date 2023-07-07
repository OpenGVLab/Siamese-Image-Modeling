# Linear Evaluation

We provide the linear evaluation scripts here. The evaluation setting mainly follows MAE, which uses 16384 batch size and LARS optimizer.

## Train with torch.distributed.launch
This method supports training on multi-nodes with torch.distributed.launch. For example, to conduct linear evaluation on 2 nodes, run the command below.

On node 1:
```
  sh ./configs/linprobe/dist_linprobe_sim_base.sh ${MASTER_ADDR} 0 2 ${CKPT_PATH} ${DATA_PATH}
```

On node 2:
```
  sh ./configs/linprobe/dist_linprobe_sim_base.sh ${MASTER_ADDR} 1 2 ${CKPT_PATH} ${DATA_PATH}
```

Note:
The `${MASTER_ADDR}` is the ip address of rank 0 node. The second and third arguments specify the node rank and node number respectively. You need to adjust them if different node numbders are used.

## Train on a slurm cluster
If you need to run the linear evaluation on a slurm cluster, use the command below to run on `${GPUS}/${GPUS_PER_NODE}` nodes with `${GPUS_PER_NODE}` gpus on each node:
```
  sh ./configs/linprobe/slurm_linprobe_sim_base.sh ${GPUS} ${GPUS_PER_NODE} ${QUOTATYPE} ${PARTITION} ${CKPT_PATH} ${DATA_PATH}
```
