#!/bin/bash
#SBATCH -J simple_test_save
#SBATCH -t 0:20:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/simple_test_save_%j.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/simple_test_save_%j.err

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_simple_test \
    --d 50 \
    --alpha_data 1.0 \
    --max_steps 5000 \
    --eval_every 50 \
    --output_dir $STORE_DIR/step1_results/simple_test_d50_alpha1 \
    --device cuda
