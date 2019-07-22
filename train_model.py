#PBS -l mem=30000MB

#PBS -N whale_train_job

#PBS -q gpu

module load shared
module load torque/6.0.2
module load anaconda/3
module load cuda80/toolkit/8.044

#### gotta figure out what's up with the call Hieu sent me

