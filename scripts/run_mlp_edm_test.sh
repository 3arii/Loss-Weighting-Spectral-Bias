#!/bin/bash
#SBATCH -J mlp_edm_test
#SBATCH -t 1:00:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_test_%j.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_test_%j.err

# EXPERIMENT Y — realistic diffusion setup.
# - Shared MLP + EDM preconditioning (c_in, c_out, c_skip, c_noise)
# - sigma range [0.002, 80] (EDM default, unchanged)
# - sigma_data auto = sqrt(mean(lambda_k)) ≈ 0.42
# - Weighting w(sigma)=sigma^beta (study variable)
#
# Test whether EDM preconditioning restores per-sigma-like dynamics in a
# shared-weight architecture. Compare alpha_trained against BOTH theory
# predictions (per-sigma phi and shared-W) in the output JSON.

echo "Job ID: $SLURM_JOB_ID  Node: $HOSTNAME  GPU: $CUDA_VISIBLE_DEVICES"

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
mkdir -p $STORE_DIR/step1_results/slurm_logs

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_mlp_sweep \
    --model_type edm \
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
    --sigma_min 0.002 \
    --sigma_max 80.0 \
    --weight_norm mean \
    --n_checkpoints 40 \
    --n_eval_samples 5000 \
    --num_ode_steps 40 \
    --output_dir $STORE_DIR/step1_results/mlp_edm_test \
    --device cuda

echo "Done."
