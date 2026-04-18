#!/bin/bash
#SBATCH -J mlp_gauss_test
#SBATCH -t 1:00:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_gauss_test_%j.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_gauss_test_%j.err

# Fail-fast run: 20k steps, beta=0.0, d=20, small MLP.
# Goal: verify pipeline + get a first meaningful alpha_trained fit.
# 5k was too short for the shared MLP to learn spectral ordering.

echo "Job ID: $SLURM_JOB_ID  Node: $HOSTNAME  GPU: $CUDA_VISIBLE_DEVICES"

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
mkdir -p $STORE_DIR/step1_results/slurm_logs

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_mlp_sweep \
    --beta 0.0 \
    --alpha_data 1.0 \
    --ndim 20 \
    --seed 42 \
    --lr 1e-3 \
    --warmup_steps 500 \
    --grad_clip 10.0 \
    --max_steps 20000 \
    --batch_size 512 \
    --nhidden 256 \
    --nlayers 4 \
    --k_sigma 50 \
    --weight_norm mean \
    --n_checkpoints 40 \
    --n_eval_samples 5000 \
    --num_ode_steps 40 \
    --output_dir $STORE_DIR/step1_results/mlp_gaussian_test \
    --device cuda

echo "Done."
