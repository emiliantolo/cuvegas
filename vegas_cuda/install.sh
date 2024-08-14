#!/bin/bash

ARCH="native" #"sm_80"

pip3 install torch numba
cd src/vegas_cuda_extension/ && nvcc cuda_kernel.cu -arch=$ARCH -m64 -O3 --keep --ptx && cd ../..
python3 setup.py install
