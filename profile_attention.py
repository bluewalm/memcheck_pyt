# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# memcheck/profile_attention.py 

import math
import torch
import bluewalm
from argparse import ArgumentParser


class NaiveAttention(torch.nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.heads = heads
        assert dim % heads == 0
        self.head_dim = dim // heads
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        bsz, query_len, _ = q.shape
        _, kv_len, _ = k.shape
        
        q = q.view(bsz, query_len, self.heads, self.head_dim)
        k = k.view(bsz, kv_len, self.heads, self.head_dim)
        v = v.view(bsz, kv_len, self.heads, self.head_dim)
        
        q = q.transpose(1, 2)  # (bsz, heads, query_len, self.head_dim) 
        k = k.transpose(1, 2)  # (bsz, heads, cache_len + query_len, self.head_dim) 
        v = v.transpose(1, 2)  # (bsz, heads, cache_len + query_len, self.head_dim) 
        
        x = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        
        cache_len = kv_len - query_len
        mask = torch.full((query_len, query_len + cache_len), -1e4, device='cuda')
        mask = mask.triu(cache_len + 1)
        
        x += mask
        x = self.dropout(x.softmax(-1))
        output = torch.matmul(x, v)
        output = output.transpose(1, 2).contiguous().view(bsz, query_len, self.dim)
        output = self.wo(output)
        
        return output, k, v


def gpu_memory():
    memory = torch.cuda.max_memory_allocated() / 1024**2
    memory = math.ceil(memory)
    memory = int(memory)
    return memory


def naive_attention_test(bsz, query_len, kv_len, dim, heads, **kwargs):
    print()
    print("SOFTMAX ATTENTION TEST [NAIVE]")
    print("bsz =", bsz, "query_len =", query_len, "kv_len =", kv_len, "dim =", dim, "heads =", heads)
    print()
    attention = NaiveAttention(dim, heads).cuda()
    q = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0,1.0)
    k = torch.zeros((bsz, kv_len, dim), device='cuda').uniform_(-1.0,1.0)
    v = torch.zeros((bsz, kv_len, dim), device='cuda').uniform_(-1.0,1.0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    output, _, _ = attention(q, k, v)
    print("forward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    output.sum().backward()
    print("backward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    print()


def softmax_attention_test(bsz, query_len, kv_len, dim, heads, **kwargs):
    print()
    print("SOFTMAX ATTENTION TEST [FLASH]")
    print("bsz =", bsz, "query_len =", query_len, "kv_len =", kv_len, "dim =", dim, "heads =", heads)
    print()
    q = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0,1.0)
    k = torch.zeros((bsz, kv_len, dim), device='cuda').uniform_(-1.0,1.0)
    v = torch.zeros((bsz, kv_len, dim), device='cuda').uniform_(-1.0,1.0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    # cache_len = kv_len - query_len
    # mask = torch.full((query_len, query_len + cache_len), -1e4, device='cuda')
    # mask = mask.triu(cache_len + 1)
    output = torch.nn.functional.scaled_dot_product_attention(
                                            q, k, v,
                                            is_causal=True, 
                                            #attn_mask=mask, 
                                            dropout_p=0.0)
    print("forward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    output.sum().backward()
    print("backward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    print()


def softplus_attention_test(bsz, query_len, kv_len, dim, core_dim, **kwargs):
    print()
    print("SOFTPLUS ATTENTION TEST")
    attention_operator = bluewalm.softplus_attention.attention_operator
    print("bsz =", bsz, "query_len =", query_len, "kv_len =", kv_len, "dim =", dim, "core_dim =", core_dim)
    print()
    q = torch.zeros((bsz, query_len, core_dim), device='cuda').uniform_(-1.0,1.0)
    k = torch.zeros((bsz, core_dim, kv_len), device='cuda').uniform_(-1.0,1.0)
    v = torch.zeros((bsz, core_dim, kv_len), device='cuda').uniform_(-1.0,1.0)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    output = attention_operator(q, k, v)
    print("forward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    output.sum().backward()
    print("backward pass required", "{:06d}".format(gpu_memory()), "MB of GPU memory")
    print()


def parse_args():
    parser = ArgumentParser(description="forward and backward memory benchmark - attention")
    parser.add_argument('--attention_type', type=str, choices=['naive', 'softmax', 'softplus'], required=True, 
                        help='type of attention')
    parser.add_argument('--bsz', type=int, required=True,
                        help='batch size')
    parser.add_argument('--query_len', type=int, required=True,
                        help='length of the query')
    parser.add_argument('--kv_len', type=int, required=True,
                        help='length of the key and value')
    parser.add_argument('--dim', type=int, required=True,
                        help='dimension of the internal representation')
    parser.add_argument('--heads', type=int, default=None,
                        help='number of attention heads (naive and softmax attention only)')
    parser.add_argument('--core_dim', type=int, default=None,
                        help='core dimension of the layer (softplus attention only)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    assert args.query_len > 0
    assert args.query_len <= args.kv_len
    assert args.dim > 0
    if args.heads is not None:
        assert args.heads > 0
    if args.core_dim is not None:
        assert args.core_dim > 0
    kwargs = args.__dict__
    try:
        if args.attention_type == 'naive':
            naive_attention_test(**kwargs)
        elif args.attention_type == 'softmax':
            softmax_attention_test(**kwargs)
        elif args.attention_type == 'softplus':
            softplus_attention_test(**kwargs)
        else:
            raise Exception("no such attention type")
    except torch.OutOfMemoryError:
        print("error : out of GPU memory")

