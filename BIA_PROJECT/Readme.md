# AIRadiologist
AIRadiologist uses three Convolutional Neural Networks (CNNs) to help diagnose tuberculosis from chest X-ray images and perform lung segmentation. 

## Installation
AIRadiologist has built-in CNN models that require CUDA to run. Please confirm that your device has a NVIDIA graphics card with CUDA support >= 11.6.

First, install the requirements. We recommend installing manually in a new `conda` environment with Python version 3.7-3.9.
```commandline
conda install pytorch torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

Then, install the requirements
```commandline
pip install -r requirements.txt
```

## Usage

To start AIRadiologist, just run `GUI.py` with a Python interpreter in the setup environment. After you see the user interface, you can follow the instructions in `Documentation_AIRadiologist.pdf` to start lung segmentation and diagnosing tuberculosis.
