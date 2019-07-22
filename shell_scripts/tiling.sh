#PBS -l nodes=1 ppn=28, walltime=01:00:00

#PBS -N tile_images_job

#PBS -q gpu

module load shared
module load torque/6.0.2
module load anaconda/3

cd ../..
cd projects/LynchGroup/spacewhale

# Let's tile some images!
# --root is the dir where the image(s) are located
# --step is the step between tiles, so start a new tile every __ pixels
# --size is the size of the tile in px (always square)
# --output is the dir to write the patches in
## E.g. if step is 100, and size is 100, there's no overlap
## if step is 30, and size 100, then there's overlap. A new 100px tile starts every
## 30 px.

source activate ./space_env

cd whale
date

python gen_training_patches.py --root ../sat_imagery/temp/hawaii_r13c1jan/ --step 32 --size 32 --output temp_figure/time_test
#python gen_training_patches.py --root new_pansharp/arg_14Oct_R15C2_2 --step 32 --size 32 --output new_pansharp/tiled_arg_14Oct_R15C2_2
#python gen_training_patches.py --root new_pansharp/arg_14Oct_R15C2_3 --step 32 --size 32 --output new_pansharp/tiled_arg_14Oct_R15C2_3
date
