#!/bin/bash
#
#
#SBATCH -J ada_net
#SBATCH --account=XXX####Change to your project
# Let's redirect job's out some other file than default slurm-%jobid-out
#SBATCH --output=log/out_log_%A.txt
#SBATCH --error=log/err_log_%A.txt
#
#SBATCH --ntasks=1
#
## Puhti
## Below commented
##SBATCH --cpus-per-task=4
##SBATCH --mem=128000
##SBATCH --time=24:00:00
##SBATCH --partition=gpu
##SBATCH --gres=gpu:v100:1
#
#
#
## Mahti
## Below commented
##Test
##SBATCH --time=00:15:00
##SBATCH --partition=gputest
#
#SBATCH --time=24:00:00
#SBATCH --partition=gpusmall
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --gres=gpu:a100:1
#
# These commands will be executed on the compute node:
module load pytorch/2.4
source my_env/bin/activate

# Finally run your job. Here's an example of a python script.
python -u train.py
