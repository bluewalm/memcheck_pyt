# Copyright (c) 2024, BLUEWALM. All rights reserved. 

import math
import torch
import bluewalm
from argparse import ArgumentParser


def gpu_memory():
    memory = torch.cuda.max_memory_allocated() / 1024**2
    memory = math.ceil(memory)
    memory = int(memory)
    return memory


def softplus_combinator_test(bsz, query_len, dim, **kwargs):
    print()
    print("SOFTPLUS COMBINATOR TEST")
    Combinator = bluewalm.softplus_attention.Combinator
    combinator = Combinator(dim)
    print("bsz =", bsz, "query_len =", query_len, "dim =", dim)
    print()
    x = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0,1.0)
    x.requires_grad = True
    output = combinator(x)
    print("forward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    output.sum().backward()
    print("backward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    print()


def parse_args():
    parser = ArgumentParser(description="forward and backward memory benchmark - combinator")
    parser.add_argument('--bsz', type=int, required=True,
                        help='batch size')
    parser.add_argument('--query_len', type=int, required=True,
                        help='length of the query')
    parser.add_argument('--dim', type=int, required=True,
                        help='dimension of the internal representation')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.bsz > 0
    assert args.dim > 0
    assert args.query_len > 0
    assert args.dim % 8 == 0
    kwargs = args.__dict__
    try:
        softplus_combinator_test(**kwargs)
    except torch.OutOfMemoryError:
        print("error : out of GPU memory")

