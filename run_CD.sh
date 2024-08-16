#!/usr/bin/env bash

gpu_ids=0
dataset_type=bcd
dataset_root=./../../../Dataset/LEVIR/LEVIR-Cropped
checkpoint_dir=./checkpoints
tb_dir=./tb_vis
vis_dir=./vis
split='train'
val_split='val'

epoch=300
batch_size=8
optimizer=adamw
encoder_arc=double
decoder_arc=ayanet
lr=0.0001

seed=1302

project_name=rep2_AYANet_LEVIR_double_mtf2iadesv2_${seed}_${encoder_arc}_${decoder_arc}_${split}_${val_split}_${optimizer}_e${epoch}_b${batch_size}_lr${lr}_newlrlambda

CUDA_VISIBLE_DEVICES=0 python train.py --seed ${seed} --gpu_ids ${gpu_ids} --dataset ${dataset_type} --split ${split} --val_split ${val_split} --datadir ${dataset_root} --checkpointroot ${checkpoint_dir} --visroot ${vis_dir} --tbroot ${tb_dir} --max-epochs ${epoch} --optimizer ${optimizer} --batch-size ${batch_size} --lr ${lr} --encoder-arch ${encoder_arc} --decoder-arch ${decoder_arc} --project_name ${project_name}

