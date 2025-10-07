#!/bin/bash

# Copyright (c) 2024, BLUEWALM. All rights reserved. 

for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for SEQ_LEN in 512
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for SEQ_LEN in 1024
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for SEQ_LEN in 2048
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 128
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 256
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 512
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 1024
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 2048
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done


for BSZ in 64
do
for DIM in 4096
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_combinator.py --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --dim ${DIM}
done
done
done
