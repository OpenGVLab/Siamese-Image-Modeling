# Few-shot Evaluation

We provide the few-shot evaluation scripts here. We only use 1% ImageNet labelled data to train the model. We follow [MSN](https://github.com/facebookresearch/msn/blob/main/logistic_eval.py) to train a linear classifier on the representation, without tuning model's parameters.

## Train with torch.distributed.launch
Few-shot evaluation does not require high computational resources, so it is enough to run the scripts on a single node, shown as follows.

```
  sh ./configs/few-shot/dist_fewshot_sim_base.sh ${MASTER_ADDR} 0 1 ${CKPT_PATH} ${DATA_PATH}
```

Note:
The `${MASTER_ADDR}` is the ip address of rank 0 node. The second and third arguments specify the node rank and node number respectively. You need to adjust them if different node numbders are used.

## Train on a slurm cluster
If you need to run the few-shot evaluation on a slurm cluster, use the command below to run on `${GPUS}/${GPUS_PER_NODE}` nodes with `${GPUS_PER_NODE}` gpus on each node:
```
  sh ./configs/few-shot/slurm_fewshot_sim_base.sh ${GPUS} ${GPUS_PER_NODE} ${QUOTATYPE} ${PARTITION} ${CKPT_PATH} ${DATA_PATH}
```
