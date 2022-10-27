# ==============================================================================
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import subprocess
import numpy as np
import random

def setup_dist_launch(args):
    args.proc_id = args.local_rank
    world_size = int(os.getenv('WORLD_SIZE', 1))*args.nodes
    print("proc_id: " + str(args.proc_id))
    print("world size: " + str(world_size))
    print("local_rank: " + str(args.local_rank))

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(args.proc_id)
    os.environ['LOCAL_RANK'] = str(args.local_rank)

def setup_slurm(args):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    args.proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    local_rank = args.proc_id % num_gpus
    args.local_rank = local_rank

    print("proc_id: " + str(args.proc_id))
    print("world size: " + str(ntasks))
    print("local_rank: " + str(local_rank))

    addr = subprocess.getoutput(
        f'scontrol show hostname {node_list} | head -n1')
    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['MASTER_ADDR'] = addr

    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(args.proc_id)
    os.environ['LOCAL_RANK'] = str(local_rank)

def setup_distributed(args):
    args.gpu = args.local_rank
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl')
    args.world_size = dist.get_world_size()
    torch.set_printoptions(precision=10)

def ddp_init(args):
    args.proc_id, args.gpu, args.world_size = 0, 0, 1
    
    if args.use_slurm == True:
        setup_slurm(args)
    else:
        setup_dist_launch(args)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    if args.distributed:
        setup_distributed(args)

    # deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.proc_id)
    np.random.seed(args.proc_id)
    random.seed(args.proc_id)

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]