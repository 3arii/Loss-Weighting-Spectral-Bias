"""Generate SLURM job scripts for the Kempner cluster.

Creates job scripts for:
1. Pilot study (all betas, 1 seed, d=200+768)
2. d=200 sweep (3 alpha_data x 11 beta x 5 seeds)
3. d=768 sweep (optional)
4. Per-sigma analytic
5. Ablations

Usage:
    python -m step1_validation.launch_slurm --output_dir slurm_jobs
    python -m step1_validation.launch_slurm --output_dir slurm_jobs --partition gpu_requeue
"""

import argparse
import os

from .config import BETA_VALUES, ALPHA_DATA_VALUES, SEEDS, D_VALUES

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time={time}
#SBATCH --output={log_dir}/{job_name}_%j.out
#SBATCH --error={log_dir}/{job_name}_%j.err

module load python
source activate torch_env  # adjust to your env name

cd {repo_dir}

{commands}

echo "Job {job_name} completed"
"""

REPO_DIR = "/n/home12/binxuwang/Github/Loss-Weighting-Spectral-Bias"
STORE_DIR = "/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/DL_Projects"


def make_run_cmd(beta, alpha_data, ndim, seed, output_dir, max_steps=50000):
    return (f"python -m step1_validation.run_sweep "
            f"--beta {beta} --alpha_data {alpha_data} --ndim {ndim} "
            f"--seed {seed} --output_dir {output_dir} --max_steps {max_steps}")


def generate_pilot(output_dir, partition, log_dir):
    """Pilot: all betas, 1 seed, d=200 and d=768."""
    commands = []
    for d in D_VALUES:
        for beta in BETA_VALUES:
            cmd = make_run_cmd(beta, 1.0, d, 42,
                               f"{STORE_DIR}/step1_results/pilot",
                               max_steps=50000)
            commands.append(cmd)

    script = SLURM_TEMPLATE.format(
        job_name="step1_pilot",
        partition=partition,
        time="04:00:00",
        log_dir=log_dir,
        repo_dir=REPO_DIR,
        commands="\n".join(commands),
    )
    path = os.path.join(output_dir, "pilot.sh")
    with open(path, "w") as f:
        f.write(script)
    print(f"  {path} ({len(commands)} runs)")


def generate_sweep(output_dir, partition, log_dir, ndim=200):
    """Full sweep: one job per alpha_data."""
    for alpha_data in ALPHA_DATA_VALUES:
        commands = []
        for beta in BETA_VALUES:
            for seed in SEEDS:
                cmd = make_run_cmd(beta, alpha_data, ndim, seed,
                                   f"{STORE_DIR}/step1_results/d{ndim}")
                commands.append(cmd)

        script = SLURM_TEMPLATE.format(
            job_name=f"step1_d{ndim}_a{alpha_data}",
            partition=partition,
            time="02:00:00" if ndim == 200 else "16:00:00",
            log_dir=log_dir,
            repo_dir=REPO_DIR,
            commands="\n".join(commands),
        )
        path = os.path.join(output_dir, f"sweep_d{ndim}_alpha{alpha_data}.sh")
        with open(path, "w") as f:
            f.write(script)
        print(f"  {path} ({len(commands)} runs)")


def generate_per_sigma(output_dir, partition, log_dir):
    """Per-sigma analytic (CPU) + GD validation."""
    commands = [
        "python -m step1_validation.run_per_sigma "
        f"--output_dir {STORE_DIR}/step1_results/per_sigma --ndim 200",
        "python -m step1_validation.run_per_sigma --gd "
        f"--output_dir {STORE_DIR}/step1_results/per_sigma --ndim 200",
    ]
    script = SLURM_TEMPLATE.format(
        job_name="step1_per_sigma",
        partition=partition,
        time="00:30:00",
        log_dir=log_dir,
        repo_dir=REPO_DIR,
        commands="\n".join(commands),
    )
    path = os.path.join(output_dir, "per_sigma.sh")
    with open(path, "w") as f:
        f.write(script)
    print(f"  {path}")


def generate_ablations(output_dir, partition, log_dir):
    """Ablation runs (optimizer, lr, threshold, K_sigma, clamping)."""
    base_dir = f"{STORE_DIR}/step1_results/ablations"
    commands = []

    # LR sensitivity: beta=-1 and beta=2, 4 lr values
    for beta in [-1.0, 2.0]:
        for lr in [0.1, 0.01, 0.001, 0.0001]:
            cmd = (f"python -m step1_validation.run_sweep "
                   f"--beta {beta} --alpha_data 1.0 --ndim 200 --seed 42 "
                   f"--lr {lr} --output_dir {base_dir}/lr_{lr}")
            commands.append(cmd)

    # Threshold sensitivity: beta=-1 and beta=-2
    # (handled in analyze_results.py by recomputing with different thresholds)

    script = SLURM_TEMPLATE.format(
        job_name="step1_ablations",
        partition=partition,
        time="01:00:00",
        log_dir=log_dir,
        repo_dir=REPO_DIR,
        commands="\n".join(commands),
    )
    path = os.path.join(output_dir, "ablations.sh")
    with open(path, "w") as f:
        f.write(script)
    print(f"  {path} ({len(commands)} runs)")


def main():
    parser = argparse.ArgumentParser(description="Generate SLURM job scripts")
    parser.add_argument("--output_dir", type=str, default="slurm_jobs")
    parser.add_argument("--partition", type=str, default="kempner_h100")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("Generating SLURM scripts:")
    generate_pilot(args.output_dir, args.partition, log_dir)
    generate_sweep(args.output_dir, args.partition, log_dir, ndim=200)
    generate_sweep(args.output_dir, args.partition, log_dir, ndim=768)
    generate_per_sigma(args.output_dir, args.partition, log_dir)
    generate_ablations(args.output_dir, args.partition, log_dir)
    print(f"\nAll scripts in {args.output_dir}/")
    print("Submit with: sbatch <script.sh>")


if __name__ == "__main__":
    main()
