#!/usr/bin/env bash

gpu_ids=0
dataset_type=bcd
dataset_root=/home/priscilla/Dataset/S2Looking/All
test_split='test'
checkpoint_dir=./checkpoints
vis_dir=./vis
resultdir=./results
checkpoint='All'
project_name='AYANet_S2Looking_gaborencoderonly_mtf2iadesv2_gaborencoderv2_drtanet_train_val_adamw_e300_b8_lr0.0001_newlrlambda'

batch_size=8
encoder_arc=gaborencoder
decoder_arc=ayanet

CUDA_VISIBLE_DEVICES=0 python eval.py --gpu_ids ${gpu_ids} --dataset ${dataset_type} --test_split ${test_split} --datadir ${dataset_root} --checkpointroot ${checkpoint_dir} --visroot ${vis_dir} --project_name ${project_name} --checkpoint ${checkpoint} --batch-size ${batch_size} --resultdir ${resultdir} --encoder-arch ${encoder_arc} --decoder-arch ${decoder_arc} --store-imgs
