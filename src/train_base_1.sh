#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_model.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_cnn_250_1_40000_200_SGD-0.1-0.9-CosineAnnealingLR&
wait
