#!/usr/bin/env bash
# * smp (shared memory processing on a single machine up to 64 cores)
#$ -pe smp 24
# the queue to bind the job to
#$ -q gpu
# request a single GPU
#$ -l gpu_card=4
# the job name
#$ -N vit_psych
# Gain access to the CRC modules

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 new_main.py --model_name="ViT" --dataset_name="imagenet" --loss_fn="psych" --log=True --batch_size=16
