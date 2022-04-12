# Training and evaluation

Please specify your config in `config/persformer_openlane.py`, especially the `args.dataset_dir` and `args.data_dir` after downloading the [OpenLane](https://github.com/OpenPerceptionX/OpenLane) Dataset or [Apollo](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection) dataset.  
Other config details please see `utils/utils.py`.  

## Training
- For **training**, set `args.evaluate = False` , and run:  
```
# using torch.distributed.launch to init
python -m torch.distributed.launch --nproc_per_node NUM_GPUS main_persformer.py --mod=EXPR_NAME --batch_size=BATCH_SIZE --nepochs=NUM_EPOCHS
# using slurm to init
srun -p PARTITION --job-name=PersFormer --mpi=pmi2 -n NUM_GPUS --gres=gpu:NUM_GPUS --ntasks-per-node=NUM_GPUS python main_persformer.py --mod=EXPR_NAME --batch_size=BATCH_SIZE --nepochs=NUM_EPOCHS
```

## Evaluation
- For **evaluation**, set `args.evaluate = True` , and run:
```
# using torch.distributed.launch to init
python -m torch.distributed.launch --nproc_per_node NUM_GPUS main_persformer.py --mod=EXPR_NAME --batch_size=BATCH_SIZE
# using slurm to init
srun -p PARTITION --job-name=PersFormer --mpi=pmi2 -n NUM_GPUS --gres=gpu:NUM_GPUS --ntasks-per-node=NUM_GPUS python main_persformer.py --mod=EXPR_NAME --batch_size=BATCH_SIZE
``` 
- We provide a pretrain model [here](https://drive.google.com/file/d/1FzrOn6Y9BifyBXUUVcDw9jYePAgKjTTB/view?usp=sharing). You could download the model here and setup an experiment folder in the following hierarchy.
```
├── data_splits
    └── waymo
        └── PersFormer
            └── model_best_epoch_40.pth.tar
```
And run the following code which specify the *EXPR_NAME*
```
# using torch.distributed.launch to init
python -m torch.distributed.launch --nproc_per_node NUM_GPUS main_persformer.py --mod=PersFormer --batch_size=BATCH_SIZE
# using slurm to init
srun -p PARTITION --job-name=PersFormer --mpi=pmi2 -n NUM_GPUS --gres=gpu:NUM_GPUS --ntasks-per-node=NUM_GPUS python main_persformer.py --mod=PersFormer --batch_size=BATCH_SIZE
```
We provide our results on 4 3090 GPUs and `torch 1.8.1+cu111` as follow:
```
===> Average loss_gflat-loss on validation set is 18.87749481
===> Evaluation laneline F-measure: 0.47771888
===> Evaluation laneline Recall: 0.44152349
===> Evaluation laneline Precision: 0.52037987
===> Evaluation laneline Category Accuracy: 0.87894481
===> Evaluation laneline x error (close): 0.32258739 m
===> Evaluation laneline x error (far): 0.77797498 m
===> Evaluation laneline z error (close): 0.21360199 m
===> Evaluation laneline z error (far): 0.68047328 m
```
And the results on 4 V100 GPUs and `torch 1.8.0+cu101` are as follow:
```
===> Average loss_gflat-loss on validation set is 18.913057
===> Evaluation on validation set: 
laneline F-measure 0.47793327 
laneline Recall  0.4414864 
laneline Precision  0.52094057 
laneline x error (close)  0.32198945 m
laneline x error (far)  0.77837964 m
laneline z error (close)  0.21318495 m
laneline z error (far)  0.68114834 m
```