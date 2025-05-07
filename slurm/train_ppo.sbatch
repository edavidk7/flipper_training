#!/bin/bash
#SBATCH --mem=32G
#SBATCH --cpus-per-task=24 --nodes=1 --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=amdgpufast
#SBATCH --job-name=train_ppo
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

echo "########################################################################"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Script arguments received by sbatch script ($0): $@"
echo "Number of arguments: $#"
echo "########################################################################"

# --- Environment Setup ---
echo "Setting up environment..."
module restore basic_python || {
    echo "Failed to restore module"
    exit 1
}
echo "Modules restored."
source /home/korcadav/.venv/bin/activate || {
    echo "Failed to activate venv"
    exit 1
}
echo "Virtual environment activated."
cd /mnt/personal/korcadav/flipper_training || {
    echo "Failed to cd into directory"
    exit 1
}
echo "Changed directory to $(pwd)"
echo "------------------------------------------------------------------------"

# --- Execution ---
# Pass all script arguments ("$@") directly to the python command.
# The quotes around "$@" are important to handle arguments with spaces correctly.
PYTHON_COMMAND="python flipper_training/experiments/ppo/train.py $@"

echo "Executing command via srun:"
echo "$PYTHON_COMMAND"
echo "------------------------------------------------------------------------"

srun $PYTHON_COMMAND # Use the variable to make it clearer

EXIT_STATUS=$?
echo "########################################################################"
echo "Python script finished with exit status: $EXIT_STATUS"
echo "Job finished."
echo "########################################################################"
exit $EXIT_STATUS # Propagate the exit status of the python script
