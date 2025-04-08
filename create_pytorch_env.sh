#!/bin/bash

# Reference to https://github.com/Global-Ecosystem-Health-Observatory/TreeMort/blob/develop/scripts/install_treemort.sh

# Exit if any command fails
set -e

# Usage:
# source create_pytorch_env.sh <venv_path> <requirements.txt>

MODULE_NAME="pytorch/2.4"

VENV_PATH="$1"
REQUIREMENTS_TXT="$2"

echo "Loading module: $MODULE_NAME"
module load $MODULE_NAME

echo "Creating virtual environment at: $VENV_PATH"
python3 -m venv --system-site-packages $VENV_PATH || { echo "Error: Failed to create virtual environment."; exit 1; }

echo "Activating virtual environment."
source $VENV_PATH/bin/activate || { echo "Error: Failed to activate virtual environment."; exit 1; }

echo "Upgrading pip."
pip install --upgrade pip || { echo "Error: Failed to upgrade pip."; exit 1; }

echo "Installing package from: $TREEMORT_REPO_PATH"
pip install -r $REQUIREMENTS_TXT || { echo "Error: Failed to install the package."; exit 1; }

echo "Script completed successfully."