#!/bin/bash

#$ -N vit_imagenet
#$ -q gpu
#S -m abe
#$ -l gpu=8

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 ViT.py --model_name="ViT" --dataset_name="imagenet" --loss_fn="else" --log=True --batch_size=16