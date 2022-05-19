#!/bin/bash

#$ -N psychophysics_model_search_debug
#$ -q gpu
#S -m abe
#$ -l gpu=4

BASE_PATH="$HOME/research/psychophysics_model_search" 

module load python
source nas_env/bin/activate
pip install -r requirements.txt

declare -a model_names=("VGG" "googlenet" "alexnet" "resnet")
declare -a seeds=(2 3 5 8 13)

for i in "${model_names[@]}"
do
    for j in "${seeds[@]}" 
    do
        echo "running on $i on seed imagenet with seed $j"
        python3 main.py --model_name="$i" --dataset_name="imagenet" --loss_fn="cross_entopy" --seed="$j" --log=True
    done
done
