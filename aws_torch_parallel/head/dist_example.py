import torch
import torch.distributed as dist
import os
import argparse
import sys

def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description="PyTorch Distributed Example")
    parser.add_argument('--rank', type=int, required=True, help='The rank of this node in the distributed setup')
    parser.add_argument('--world_size', type=int, required=True, help='The total number of nodes participating in the distributed setup')
    parser.add_argument('--ip', type=str, required=True, help='IP address of the master node')
    args = parser.parse_args()

    print(f"Starting on rank {args.rank}, connecting to master at {args.ip}")

    # Initialize the distributed environment
    os.environ['MASTER_ADDR'] = args.ip
    os.environ['MASTER_PORT'] = '29500'

    try:
        print("Initializing process group...")
        dist.init_process_group('gloo', rank=args.rank, world_size=args.world_size)
        print("Process group initialized.")
    except Exception as e:
        print("Failed to initialize process group:", e)
        sys.exit(1)

    # Example tensor operation
    tensor = torch.zeros(1)
    if args.rank == 0:
        tensor += 1
        print("Rank 0 is sending tensor to Rank 1")
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        print("Rank 1 is receiving tensor from Rank 0")
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)

    print(f"Rank {args.rank} has tensor {tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
