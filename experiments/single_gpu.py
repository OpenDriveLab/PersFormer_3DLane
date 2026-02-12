# utils/utils.py

import torch
import numpy as np
import random
import os

def init_single_gpu(args):
    """
    Initialize necessary attributes for single-GPU (non-distributed) training.
    Should be called when args.distributed == False.
    """
    args.distributed = False
    args.rank = 0
    args.local_rank = 0
    args.world_size = 1
    args.proc_id = 0
    args.gpu = 0  # 默认使用 GPU 0；也可用 torch.cuda.current_device() 如果已设 CUDA_VISIBLE_DEVICES
    args.is_main_process = True

    # Set device
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')

    # Set random seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # CuDNN settings: enable benchmark for speed in single-GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Optional: log info
    print(f"[Single-GPU] Using device: {args.device}")