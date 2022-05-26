#!/bin/bash

#$ -N imagenet_VGG_XE
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 new_main.py --model_name="VGG" --dataset_name="imagenet" --loss_fn="cross_entropy" --log=True --batch_size=16

