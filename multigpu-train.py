import os


cmd = 'python -m torch.distributed.run --nproc_per_node 2 train.py'
os.system(cmd)