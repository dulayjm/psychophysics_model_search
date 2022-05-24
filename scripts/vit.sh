#!/bin/bash

#$ -N vit_debug_psych
#$ -pe smp 8
#$ -q gpu
#S -m abe
#$ -l gpu=4
#$ -t 1-10

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 vit.py --model_name="ViT" --dataset_name="imagenet" --loss_fn="psych" --log=True --batch_size=16
