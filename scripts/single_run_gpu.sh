#!/bin/bash

#$ -N debug_python
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
pip3 install -r requirements.txt

python3 new_main.py --model_name="VGG" --dataset_name="else" --loss_fn="cross_entopy" --log=True --batch_size=64
