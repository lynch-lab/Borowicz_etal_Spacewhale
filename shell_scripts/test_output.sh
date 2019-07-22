#PBS -l walltime=06:00:00

#PBS -N model_testing_job

#PBS -q gpu

module load shared
module load torque/6.0.2
module load anaconda/3

cd ../..
cd projects/LynchGroup/spacewhale

# Let's test a trained model!
# This script just iteratively calls test_script.py with for each epoch.
# It will be a pain to adapt, and I should write it differently.

source activate ./space_env

cd whale

for i in {0..23}
do
     python test_script.py --data_dir tiled_air32/fold_4/test_4/ --model f4_32_001 --epoch $i
done



