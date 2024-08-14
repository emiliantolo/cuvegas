# cuVegas

cuVegas: Accelerate Multidimensional Monte Carlo Integration through a Parallelized CUDA-based Implementation of the VEGAS Enhanced Algorithm

## Project structure

- **src/** - contains CUDA C source files
    - **config.cuh**: contains integrand definition and parameters
    - **main.cu**: main file, executes integration
    - **commons/**: contains integrands and other helpers
- **vegas_cuda/** - contains the python binding

## Run

Tested on Ubuntu 22.04, Python 3.10, CUDA 11.8. It requires compute capability >= sm_60.

### Get code
    git clone https://github.com/emiliantolo/cuvegas.git
    cd cuvegas

### CUDA C

#### Compile

    make

#### Run

    ./bin/main.bin

### Python extension

#### Create virtual environment

    python3 -m venv venv
    source venv/bin/activate

#### Install

    cd vegas_cuda
    chmod +x install.sh
    ./install.sh

#### Run test

    python3 test.py
