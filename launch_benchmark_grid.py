import itertools
import subprocess
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch a grid search of benchmarks on Slurm.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--partition", type=str, default=None, help="Slurm partition to use.")
    parser.add_argument("--script", type=str, default="benchmark_job.slurm", help="Path to the Slurm script.")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Hyperparameter Grid Definition
    # -------------------------------------------------------------------------
    # Define your grid here. Keys must match the hydra config structure.
    # Lists represent the values to sweep over.
    grid = {
        "attack.optim.step_size": [0.1, 0.5, 1.0],
        "attack.regularization.total_variation.scale": [0.01, 0.1, 1.0],
        "case.data.partition": ["unique-users"],
    }

    # Static arguments that apply to all runs
    static_args = [
        "python", "benchmark_breaches.py",
        "case=1_single_image_small", # Example case
        "attack=gradvit", # Example attack
        "dryrun=True" # Set to False for actual runs
    ]

    # -------------------------------------------------------------------------
    # Grid Search Execution
    # -------------------------------------------------------------------------
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Found {len(combinations)} combinations to run.")

    for i, params in enumerate(combinations):
        # Construct the command arguments
        cmd_args = list(static_args)
        job_name_suffix = []
        
        for key, value in params.items():
            cmd_args.append(f"{key}={value}")
            # Create a compact suffix for the job name
            short_key = key.split(".")[-1]
            job_name_suffix.append(f"{short_key}{value}")

        full_cmd_str = " ".join(cmd_args)
        job_name = f"bench_{i}_{'_'.join(job_name_suffix)}"
        
        # Construct the sbatch command
        sbatch_cmd = ["sbatch", "--job-name", job_name]
        
        if args.partition:
            sbatch_cmd.extend(["--partition", args.partition])
            
        sbatch_cmd.append(args.script)
        sbatch_cmd.extend(cmd_args)

        if args.dry_run:
            print(f"[Dry Run] would execute: {' '.join(sbatch_cmd)}")
        else:
            print(f"Submitting job {i+1}/{len(combinations)}: {job_name}")
            try:
                subprocess.run(sbatch_cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e}")
                sys.exit(1)

if __name__ == "__main__":
    main()
