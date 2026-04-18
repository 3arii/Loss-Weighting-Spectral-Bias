#!/bin/bash
#SBATCH -J mlp_edm
#SBATCH -t 4:00:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH --array=0-4
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_%A_%a.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_%A_%a.err

# EXPERIMENT Y — beta sweep, realistic diffusion setup.
# Shared MLP + EDM preconditioning, sigma range [0.002, 80], 100k steps.
# Output: alpha_trained vs (alpha_phi [per-sigma], alpha_sharedW) per beta.

BETA_LIST=(-2.0 -1.0 0.0 1.0 2.0)
BETA=${BETA_LIST[$SLURM_ARRAY_TASK_ID]}

echo "Job ID: $SLURM_JOB_ID  Task: $SLURM_ARRAY_TASK_ID  Beta: $BETA"
echo "Node: $HOSTNAME  GPU: $CUDA_VISIBLE_DEVICES"

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
mkdir -p $STORE_DIR/step1_results/slurm_logs

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_mlp_sweep \
    --model_type edm \
    --beta $BETA \
    --alpha_data 1.0 \
    --ndim 20 \
    --seed 42 \
    --lr 1e-3 \
    --warmup_steps 500 \
    --grad_clip 10.0 \
    --max_steps 100000 \
    --batch_size 512 \
    --nhidden 256 \
    --nlayers 4 \
    --k_sigma 50 \
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --weight_norm mean \
    --n_checkpoints 80 \
    --n_eval_samples 5000 \
    --num_ode_steps 40 \
    --output_dir $STORE_DIR/step1_results/mlp_edm_sweep \
    --device cuda

echo "Done for beta=$BETA"
