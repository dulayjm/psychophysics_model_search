#!/bin/zsh

BASE_PATH="/Users/justindulay/research/ViT/psychophysics_model_search" 

declare -a model_names=("VGG" "googlenet" "alexnet" "resnet")
declare -a seeds=(2 3 5 8 13)

for i in "${model_names[@]}"
do
    for j in "${seeds[@]}" 
    do
        echo "running on $i on seed imagenet with seed $j"
        python3 main.py --model_name="$i" --dataset_name="tiny-imagenet-200" --loss_fn="cross_entopy" --seed="$j" --log=False
    done
done