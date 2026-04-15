#!/bin/bash
#SBATCH -J mlp_dinov2
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=0-3
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_dinov2_%A_%a.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_dinov2_%A_%a.err

# Beta sweep: -2.0, -1.0, 0.0, 1.0
BETA_LIST=(-2.0 -1.0 0.0 1.0)
BETA=${BETA_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID  Array task: $SLURM_ARRAY_TASK_ID  Beta: $BETA"
echo "Node: $HOSTNAME  GPU: $CUDA_VISIBLE_DEVICES"

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"

# Verify CUDA
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_mlp_dinov2 \
    --beta $BETA \
    --output_dir $STORE_DIR/step1_results/mlp_dinov2 \
    --seed 42 \
    --nhidden 512 \
    --nlayers 6 \
    --max_steps 50000 \
    --n_callback_steps 100 \
    --n_eval_samples 1000 \
    --num_ode_steps 30 \
    --k_sigma 100 \
    --batch_size 512 \
    --device cuda

echo "Training done for beta=$BETA"
