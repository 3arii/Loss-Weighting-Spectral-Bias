#!/bin/bash
#SBATCH -J mlp_edm_sgd
#SBATCH -t 1:30:00
#SBATCH -p kempner_h100
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=kempner_binxuwang_lab
#SBATCH -o /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_sgd_%j.out
#SBATCH -e /n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/step1_results/slurm_logs/mlp_edm_sgd_%j.err

# ADAM vs SGD DIAGNOSTIC — does Adam's per-parameter second-moment adaptation
# flatten the lambda_k-scaled gradient structure and suppress spectral bias?
#
# Same config as Experiment Y (EDM MLP, sigma range [0.002, 2.0] for shared-W
# regime match) EXCEPT:
#   --optimizer sgd      (plain SGD + momentum, no second-moment adaptation)
#   --lr 1e-2            (much smaller than Adam's 1e-3; SGD needs gentler
#                         steps because updates scale directly with grad norm)
#   --momentum 0.9       (stabilizes training without flattening per-mode rates)
#   --n_checkpoints 200  (dense early checkpoints so we can resolve sequential
#                         emergence if it actually happens in the first ~200 steps)
#   --max_steps 50000    (SGD needs more steps to converge than Adam)
#
# Decision rule for the next meeting with Binxu:
#   - SGD alpha_trained approx alpha_phi (sequential emergence) -> Adam was
#     the culprit. Publish as "theory survives gradient flow on realistic
#     architecture; Adam suppresses it."
#   - SGD alpha_trained still flat -> architecture/capacity is the issue;
#     theory gap is deeper than optimizer choice.

echo "Job ID: $SLURM_JOB_ID  Node: $HOSTNAME  GPU: $CUDA_VISIBLE_DEVICES"

PYTHON=/n/home12/binxuwang/.conda/envs/torch2/bin/python
export PATH="/n/home12/binxuwang/.conda/envs/torch2/bin:${PATH}"
export STORE_DIR="/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang"
mkdir -p $STORE_DIR/step1_results/slurm_logs

$PYTHON -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

cd /n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias

$PYTHON -m step1_validation.run_mlp_sweep \
    --model_type edm \
    --optimizer sgd \
    --momentum 0.9 \
    --beta 0.0 \
    --alpha_data 1.0 \
    --ndim 20 \
    --seed 42 \
    --lr 1e-2 \
    --warmup_steps 1000 \
    --grad_clip 10.0 \
    --max_steps 50000 \
    --batch_size 512 \
    --nhidden 256 \
    --nlayers 4 \
    --k_sigma 50 \
    --sigma_min 0.002 \
    --sigma_max 2.0 \
    --weight_norm mean \
    --n_checkpoints 200 \
    --n_eval_samples 5000 \
    --num_ode_steps 40 \
    --output_dir $STORE_DIR/step1_results/mlp_edm_sgd_test \
    --device cuda

echo "Done."
