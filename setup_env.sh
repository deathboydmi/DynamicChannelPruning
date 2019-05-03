#!/bin/bash
set -e

# installing virtual envariament for project
sudo apt install -y python3-venv
mkdir .env
python3 -m venv ./.env
source ./.env/bin/activate
which python3
which pip3

# installing PyTorch and DP_Plugin
pip3 install torch==1.1.0 torchvision 
pip3 install -e ./dynamic_pruning

# sample launch
pip3 install ipykernel
python -m ipykernel install --user --name=env
jupyter notebook prun_vgg_example.ipynb