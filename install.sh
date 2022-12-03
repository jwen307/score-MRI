#!/bin/bash

# download model weights
mkdir weights
wget -O weights/checkpoint_95.pth https://www.dropbox.com/s/27gtxkmh2dlkho9/checkpoint_95.pth?dl=0

# create env and activate
conda create -n score-POCS python=3.8
conda activate score-POCS

# install dependencies
# conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
