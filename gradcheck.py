# Copyright (c) 2024, BLUEWALM. All rights reserved. 
# memcheck/gradcheck.py 

import torch
import bluewalm
from torch.autograd.gradcheck import gradcheck


def check_attention():
    attention_operator = bluewalm.softplus_attention.attention_operator
    bsz, query_len, kv_len, core_dim = 8, 16, 128, 16
    q = torch.zeros((bsz, query_len, core_dim), device='cuda').uniform_(-1.0, 1.0).double()
    k = torch.zeros((bsz, core_dim, kv_len), device='cuda').uniform_(-1.0, 1.0).double()
    v = torch.zeros((bsz, core_dim, kv_len), device='cuda').uniform_(-1.0, 1.0).double()
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    
    def f(q,k,v):
        return attention_operator(q, k, v)
    
    if(gradcheck(f, (q, k, v))):
        print("attention - gradients seem ok")


def check_linear_r():
    QueryProjection = bluewalm.softplus_attention.QueryProjection
    q_proj = QueryProjection(32, 16)
    bsz, query_len, dim = 8, 16, 32
    x = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0, 1.0).double()
    x.requires_grad = True
    w = q_proj.weight.double()
    
    def f(w, x):
        return torch.ops.bluewalm_extension.linear_operator_r(w, x, 32, 16)
    
    if(gradcheck(f, (w, x))):
        print("linear operator r - gradients seem ok")


def check_linear_l():
    KeyProjection = bluewalm.softplus_attention.KeyProjection
    k_proj = KeyProjection(32, 16)
    bsz, query_len, dim = 8, 16, 32
    x = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0, 1.0).double()
    x.requires_grad = True
    w = k_proj.weight.double()
    
    def f(w, x):
        return torch.ops.bluewalm_extension.linear_operator_l(w, x, 32, 16)
    
    if(gradcheck(f, (w, x))):
        print("linear operator l - gradients seem ok")


def check_combinator_lm():
    Combinator = bluewalm.softplus_attention.Combinator
    combinator = Combinator(32)
    bsz, query_len, dim = 8, 16, 32
    x = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0, 1.0).double()
    x.requires_grad = True
    w = combinator.weight.double()
    
    def f(w, x):
        return torch.ops.bluewalm_extension.combinator_operator(w, x, 32, True)
    
    if(gradcheck(f, (w, x))):
        print("combinator - low memory - gradients seem ok")


def check_combinator_p():
    Combinator = bluewalm.softplus_attention.Combinator
    combinator = Combinator(32)
    bsz, query_len, dim = 8, 16, 32
    x = torch.zeros((bsz, query_len, dim), device='cuda').uniform_(-1.0, 1.0).double()
    x.requires_grad = True
    w = combinator.weight.double()
    
    def f(w, x):
        return torch.ops.bluewalm_extension.combinator_operator(w, x, 32, False)
    
    if(gradcheck(f, (w, x))):
        print("combinator - performance - gradients seem ok")


def main():
    check_attention()
    check_linear_l()
    check_linear_r()
    check_combinator_lm()
    check_combinator_p()


if __name__ == "__main__":
    main()

