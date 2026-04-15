#!/bin/bash
#SBATCH -J mlp_dinov2_test
#SBATCH -t 1:00:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_dinov2_test_%j.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_dinov2_test_%j.err

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Setup — use absolute path to avoid conda prefix inheritance from launcher
PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"

export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
mkdir -p $STORE_DIR/step1_results/slurm_logs

# Verify CUDA
$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

# Short test run: 2000 steps, beta=0.0, small model to verify pipeline
$PYTHON -m step1_validation.run_mlp_dinov2 \
    --beta 0.0 \
    --output_dir $STORE_DIR/step1_results/mlp_dinov2_test \
    --seed 42 \
    --nhidden 256 \
    --nlayers 4 \
    --max_steps 2000 \
    --n_callback_steps 30 \
    --n_eval_samples 1000 \
    --num_ode_steps 30 \
    --k_sigma 50 \
    --batch_size 512 \
    --device cuda

echo "Training done. Running analysis..."

$PYTHON -m step1_validation.analyze_mlp_dinov2 \
    --exp_dir $STORE_DIR/step1_results/mlp_dinov2_test/norm_patch_betap00_nhid256_nl4_seed42 \
    --output_dir $STORE_DIR/step1_results/mlp_dinov2_test/analysis

echo "Done!"
