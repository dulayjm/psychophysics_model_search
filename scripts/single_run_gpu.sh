#!/bin/bash

#$ -N psychophysics_model_search_debug
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/research/psychophysics_model_search" 

module load python
source nas_env/bin/activate
pip install -r requirements.txt

python main.py --model_name="VGG" --dataset_name="tiny-imagenet-200" --loss_fn="cross_entopy" --log=True