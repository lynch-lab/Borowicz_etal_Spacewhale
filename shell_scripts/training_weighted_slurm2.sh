#!/bin/bash
#
#SBATCH --job-name=sat_training2
#SBATCH --output=resnet18_full32training_res_lr1.txt
#SBATCH --ntasks=28
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH -p gpu

pwd
#cd ../..
#cd projects/LynchGroup/spacewhale

# Let's train our whalefinder model!
# --name  is a new name for your model!
# --data_dir is where your images are
# !!!!: They must be in a directory named train!
# So if --data_dir training_images, then training images must contain
#        a dir called train, and inside is a whale and a water dir
# --epochs is the number of epochs you want

module load shared
module load anaconda/3
module load cuda91/toolkit/9.1

cd ../../..
pwd
source activate ./space_env

cd git_spacewhale/spacewhale
date

python training_tester_weighted2.py --name resnet18_full32_lr1 --model resnet18 --data_dir ../../whale/tiled_air32/full_air/  --epochs 24
date
