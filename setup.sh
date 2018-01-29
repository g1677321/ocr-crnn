#!/bin/bash

#You should check the following commands applied to your system environment.

## For Linux
sudo apt-get install python-pip
pip install -r requirement.txt

### PyTorch >= 0.1.12, here is 0.3.0

## Linux: python-2.7, pip, no CUDA
pip install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl 
## if the above command does not work, then you have python 2.7 UCS2, use this command 
#pip install http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp27-cp27m-linux_x86_64.whl

## OSX: python-2.7, pip, no CUDA
#pip install http://download.pytorch.org/whl/torch-0.3.0.post4-cp27-none-macosx_10_6_x86_64.whl 
## macOS Binaries dont support CUDA, install from source if CUDA is needed


### PyTorch vision
pip install torchvision 