#!/bin/bash

# Run a python command in a given conda environment.
#
# Usage:
#
#  python_conda.sh ENV_NAME FOLDER COMMAND ARGUMENTS
# 
# Examples:
#
# Run the interactive shell - python
#  python_conda.sh my_env
#
# Run a single python command - python -c print("Hello world!")
#  python_conda.sh my_env -c 'print("Hello world!")'
#
# Source: https://gist.github.com/GioBonvi/0f864717e415bf5bae1ae6516df3b34c

# Adapt according to your installation.
CONDA_BASE_DIR="/mnt/c/Users/LuDoYa/anaconda3"
ls /
cd $CONDA_BASE_DIR
cd "$2"

# Activate the conda environment.
source "$CONDA_BASE_DIR/etc/profile.d/conda_mod.sh"
conda activate "$1"

# Execute the python command.
python inference_one_sample.py --image demo/src/COD_CAMO_camourflage_00012.jpg \
                               --mask demo/src/COD_CAMO_camourflage_00012.png \
                               --log_path demo_res 