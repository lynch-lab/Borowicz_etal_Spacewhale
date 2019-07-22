#!/bin/bash
#
#SBATCH --job-name=1testRes18
#SBATCH --output=resnet18_test_f10check.txt
#SBATCH --ntasks=28
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH -p gpu


module load shared
#module load torque
module load anaconda/3
module load cuda91/toolkit/9.1

pwd
date
cd ../../..
#cd projects/LynchGroup/spacewhale


# Let's test a trained model!
# This script just iteratively calls test_script.py with for each epoch.
# It will be a pain to adapt, and I should write it differently.

source activate ./space_env

cd ./git_spacewhale/spacewhale

for i in {0..23}
do
     python test_script.py --data_dir ../../whale/tiled_air32/10_fold/fold_10/test --modtype resnet18 --model resnet18_f10 --epoch $i
done

date
