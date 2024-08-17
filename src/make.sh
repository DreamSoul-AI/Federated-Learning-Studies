#!/bin/bash

# python make.py --mode base --run train --num_experiments 1 --round 8
# python make.py --mode base --run test --num_experiments 1 --round 8

python make.py --mode fl --run train --num_experiments 1 --round 5 --split_round 2 --num_gpus 1
python make.py --mode fl --run test --num_experiments 1 --round 5 --split_round 2 --num_gpus 1