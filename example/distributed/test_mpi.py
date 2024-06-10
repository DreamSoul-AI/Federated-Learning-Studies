import torch
import torch.distributed as dist
import os
import argparse
import sys
from mpi4py import MPI

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Parsing arguments
    parser = argparse.ArgumentParser(description="PyTorch Distributed Example")
    parser.add_argument('--master_addr', type=str, required=True, help='IP address of the master node')
    parser.add_argument('--master_port', type=str, default='29500', help='Port used by the master node')
    args = parser.parse_args()

    print(f"Starting on rank {rank}, connecting to master at {args.master_addr}:{args.master_port}")

    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port

    try:
        print("Initializing process group...")
        dist.init_process_group('gloo', rank=rank, world_size=world_size)
        print("Process group initialized.")
    except Exception as e:
        print("Failed to initialize process group:", e)
        sys.exit(1)

    # Example tensor operation
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        print("Rank 0 is sending tensor to Rank 1")
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        print("Rank 1 is receiving tensor from Rank 0")
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print(f"Rank {rank} has tensor {tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
