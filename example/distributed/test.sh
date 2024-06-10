python3 test_dist.py --rank 0 --world_size 2 --master_addr 172.31.18.210 --master_port 29500
python3 test_dist.py --rank 1 --world_size 2 --master_addr 172.31.18.210 --master_port 29500

mpirun -np 2 --hostfile hosts python3 test_mpi.py --master_addr 18.223.155.162 --master_port 29500