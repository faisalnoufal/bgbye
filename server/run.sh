#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Suppress NNPACK warnings
export OMP_NUM_THREADS=1
export NNPACK_DISABLE=1

# Start the server
python3 server.py