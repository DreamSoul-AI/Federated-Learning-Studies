# Installation and Setup Guide

## Install Necessary Packages
1. Update and upgrade the system:
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo reboot
   ```

2. Install Python and required libraries:
   ```bash
   sudo apt install python3 python3-pip -y
   ```

3. Install OpenMPI:
   ```bash
   sudo apt install openmpi-bin openmpi-common libopenmpi-dev -y
   ```

4. Install Python packages:
   ```bash
   pip3 install torch --index-url https://download.pytorch.org/whl/cpu
   pip3 install mpi4py
   ```

## Configure SSH
1. Edit the SSH configuration:
   ```bash
   sudo nano ~/.ssh/config
   ```

2. Add the following lines to the configuration file:
   ```
   Host *
       StrictHostKeyChecking accept-new
       IdentityFile ~/.ssh/[your_private_key]
   ```

3. Set the appropriate permissions:
   ```bash
   sudo chmod 700 ~/.ssh 
   sudo chmod 600 ~/.ssh/*
   ```

## Run Commands
1. On each node with a private IP, run:
   ```bash
   python3 test_dist.py --rank 0 --world_size 2 --master_addr 172.31.18.210 --master_port 29500
   python3 test_dist.py --rank 1 --world_size 2 --master_addr 172.31.18.210 --master_port 29500
   ```

2. On the master node with a public IP, run:
   ```bash
   mpirun -np 2 --hostfile hosts python3 test_mpi.py --master_addr 18.223.155.162 --master_port 29500
   ```

## References
- [MPI on Amazon EC2](https://github.com/BaoqianWang/MPI-on-Amazon-EC2)
- [Open MPI FAQ](https://www.open-mpi.org/faq/?category=rsh)
- [Configure SSH for MPI](https://source.ggy.bris.ac.uk/wiki/Configure_ssh_for_MPI)
- [Avoid SSH Asking Permission](https://unix.stackexchange.com/questions/33271/how-to-avoid-ssh-asking-permission)
- [Disable Strict Host Key Checking in SSH](https://askubuntu.com/questions/87449/how-to-disable-strict-host-key-checking-in-ssh)

This revised README should be clearer and more user-friendly.