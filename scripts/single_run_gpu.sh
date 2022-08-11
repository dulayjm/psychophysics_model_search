#!/bin/bash

#$ -N resnet_XE_dropout
#$ -q gpu
#S -m abe
#$ -l gpu=1

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 new_main.py --model_name="resnet" --dataset_name="imagenet" --loss_fn="cross_entropy" --log=False --batch_size=16

