#!/bin/sh                                                                                                                                                     

#####
# PBS directives need to go first

# Job parameters, throughput:
#PBS -lselect=1:ncpus=16:mem=96gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=06:00:00

# Copy current terminal environment to nodes
# If using conda activate first in submission terminal
#PBS -V
# this can be checked with qstat -f job_ID and checking the 
# Variable_List description in the output
# Some environment variables are copied across regardless though
# such as PATH and LANG

#####

export PATH=~/miniconda/envs/yoloV5_model/bin/:$PATH
source activate yoloV5_model
PBS_O_WORKDIR=/rds/general/user/ga402/home/009_GAN/003_2D_DCGAN


### THIS IS WHAT YOU NEED TO CHANGE
# 1. Accessory infomation
INFO="DCGAN2d; Conditional GAN attempt"

# 2. The project directory name
PROJECT_DIR=/rds/general/user/ga402/home/PROJECTS/001_AMLCD8TC

# project
PROJECT_NAME="${PROJECT_DIR}/DCGAN2d"

# location of the data
# dataroot="${PROJECT_DIR}/IndividualJPG"
image_data=/rds/general/user/ga402/home/PROJECTS/001_AMLCD8TC/IMAGES2D
label_data='/rds/general/user/ga402/home/PROJECTS/001_AMLCD8TC/images128Filtered.csv'

# Number of workers for dataloader
workers=2

# Batch size during training
batch_size=128

# Spatial size of training images. All images will be resized to this
image_size=64

# Number of channels in the training images. For color images this is 3
nc=3

# Size of z latent vector (i.e. size of generator input)
nz=100

# Size of feature maps in generator
#ngf=64

# Size of feature maps in discriminator
#ndf=64

# Number of training epochs
num_epochs=200

# Learning rate for optimizers
# lr=0.0002
lr=0.00001

# Beta1 hyperparameter for Adam optimizers
beta1=0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu=1



####



#####
# Treat as normal bash script, set bash options:
# exit when a command fails:
set -o errexit
# exit if any pipe commands fail:
set -o pipefail
# exit if there are undeclared variables:
set -o nounset
# trace what gets executed:
set -o xtrace
set -o errtrace
#####


#####

# Actual commands of interest to run:
export CUBLAS_WORKSPACE_CONFIG=:16:8 #set env variable in terminal
python ${PBS_O_WORKDIR}/main.py --project $PROJECT_NAME \
       --dataroot $image_data \
       --label_data $label_data \
       --workers $workers \
       --batch_size $batch_size \
       --image_size $image_size \
       --nc $nc --nz $nz \
       --num_epochs $num_epochs \
       --lr $lr --beta1 $beta1 --ngpu $ngpu \
       --accessory_info "$INFO"
