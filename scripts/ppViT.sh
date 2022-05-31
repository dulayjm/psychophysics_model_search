#$ -N vit_Psych_test
#$ -q gpu
#S -m abe
#$ -l gpu=1

BASE_PATH="$HOME/research/psychophysics_model_search" 

source nas_env/bin/activate
# pip3 install -r requirements.txt

python3 vit.py --train=False --model_name="ViT" --dataset_name="imagenet" --loss_fn="psych" --log=True --batch_size=16
