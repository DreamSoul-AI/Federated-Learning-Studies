#!/bin/bash
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.001~0~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.001~0.5~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.001~0.9~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.001~0.99~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.001~0.999~CosineAnnealingLR --device cpu
wait
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.01~0~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.01~0.5~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.01~0.9~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.01~0.99~CosineAnnealingLR --device cpu&
python train_model_fl.py --init_seed 0 --num_experiments 1 --resume_mode 0 --control_name CIFAR10_linear_250_1_2500_50_SGD-1-0-None_100-horiz-noniid~r~2~1_sync-0.1-50-server-SGD~0.01~0~CosineAnnealingLR~True-SGD~0.01~0.999~CosineAnnealingLR --device cpu
wait
