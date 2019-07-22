#PBS -l nodes=1 ppn=28, walltime=04:00:00
#PBS -N train_whalemodel
#PBS -q short

module load short
module load torque/6.0.2
module load anaconda/3

cd ../..
cd projects/LynchGroup/spacewhale

#Let's pull out whales in bounding boxes!
#This script takes a directory of png images which are accompanied
#by .shp files that are bounding boxes, encompassing whales in larger images
#It clips out the whales, so we have water images (with a hole in it) and whale images
#USAGE:



source activate osgeo

python pull_out_target.py --target_dir 31_cm_aerialdownsample_bounded
