#!/bin/bash

python3 precompute_training_inputs/build_data.py
python3 precompute_training_inputs/build_crops.py
python3 precompute_training_inputs/build_projindices.py