#PBS -l nodes=1 ppn=28, walltime=04:00:00
#PBS -N train_whalemodel
#PBS -q short

module load short
module load torque/6.0.2
module load anaconda/3

cd ../..
cd projects/LynchGroup/spacewhale

#Let's train the whale finding model!
#--name is the name of the model we're going to train
#--data_dir is the directory where we have our training images
#this dir should have two dirs in it: water and whale

source activate ./space_env

python transfer_learning.py --name whale_modeltest --data_dir 
