#!/bin/bash

conda create -y --name py37-simswap python=3.7 

conda activate py37-simswap

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

pip install imageio insightface==0.2.1 moviepy onnxruntime-gpu