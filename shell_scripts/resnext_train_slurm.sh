#!/bin/bash
#
#SBATCH --job-name=sat_training2
#SBATCH --output=resnext32_training_res.txt
#SBATCH --ntasks=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu

pwd
#cd ../..
#cd projects/LynchGroup/spacewhale

# Let's train our whalefinder model!
# --name  is a new name for your model!
# --model is the pre-trained model (densenet, reneXT, resnet18)
# --data_dir is where your images are
# !!!!: They must be in a directory named training!
# So if --data_dir training_images, then training images must contain
#        a dir called training, and inside is a whale and a water dir
# --epochs is the number of epochs you want

module load shared
module load anaconda/3
module load cuda91/toolkit/9.1

cd ../../..
pwd
source activate ./space_env

cd git_spacewhale/spacewhale
date

python training_tester_weighted.py --name resnext_full_air --model resneXt --data_dir ../../whale/tiled_air32/full_air  --epochs 24
date
