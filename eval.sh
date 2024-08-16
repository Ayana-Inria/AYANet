#!/usr/bin/env bash

gpu_ids=0
dataset_type=bcd
dataset_root=./../../../Dataset/LEVIR/LEVIR-Cropped # The path to the dataset folder
test_split='test' # The split you want to test on (adjust according to the file name you put in the 'list' folder of the dataset)
checkpoint_dir=./checkpoints # Name of the folder where you store the weights of the model during the training
vis_dir=./vis # Visualization folder
resultdir=./results
checkpoint='All' # Option 'All' will run the evaluation on all weights stored in checkpoint_dir, if you want to evaluate using only 1 checkpoint, specify the name e.g., 'best_ckpt.pt'
project_name='rep_AYANet_LEVIR_double_mtf2iadesv2_1302_double_ayanet_train_val_adamw_e300_b8_lr0.0001_newlrlambda ' # Specify which model/experiment you want to do the evaluation with

batch_size=8
encoder_arc=double # 'double': double encoders, 'gaborencoder': Gabor encoder only, 'efficientnet_ayn': EfficientNet only
decoder_arc=ayanet

CUDA_VISIBLE_DEVICES=0 python eval.py --gpu_ids ${gpu_ids} --dataset ${dataset_type} --test_split ${test_split} --datadir ${dataset_root} --checkpointroot ${checkpoint_dir} --visroot ${vis_dir} --project_name ${project_name} --checkpoint ${checkpoint} --batch-size ${batch_size} --resultdir ${resultdir} --encoder-arch ${encoder_arc} --decoder-arch ${decoder_arc}