#!/bin/bash
#SBATCH --job-name train_dp
#SBATCH -c 16              # Number of cores (-c)
#SBATCH -t 0-12:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p kempner # Partition to submit to
#SBATCH --gres=gpu:1        # Number of GPUs (per node)
#SBATCH --mem=200g   # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o scratch_dir/slurm_logs/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e scratch_dir/slurm_logs/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -A kempner_ydu_lab

module load python
mamba activate libero_env
export HYDRA_FULL_ERROR=1
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/xczhang/workspace/SAILOR/diffusion_policy
python train.py
# python diffusion_policy/workspace/train_diffusion_unet_image_workspace.py
# python  diffusion_policy/workspace/train_diffusion_unet_lowdim_idm_workspace.py 
