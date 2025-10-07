#!/bin/bash

# Copyright (c) 2024, BLUEWALM. All rights reserved. 

for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for HEADS in 8
do
for SEQ_LEN in 512
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for HEADS in 8
do
for SEQ_LEN in 1024
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for HEADS in 8
do
for SEQ_LEN in 2048
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for HEADS in 8
do
for SEQ_LEN in 4096
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 32 64 128 256 512 1024 2048 4096 8192
do
for HEADS in 8
do
for SEQ_LEN in 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 128
do
for HEADS in 8
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 256
do
for HEADS in 8
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 512
do
for HEADS in 8
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 1024
do
for HEADS in 8
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done


for ATTENTION_TYPE in naive softmax softplus
do
for BSZ in 64
do
for DIM in 2048
do
for HEADS in 8
do
for SEQ_LEN in 32 64 128 256 512 1024 2048 4096 8192
do
python profile_attention.py --attention_type ${ATTENTION_TYPE} \
                       --bsz ${BSZ} \
                       --query_len ${SEQ_LEN} \
                       --kv_len ${SEQ_LEN} \
                       --dim ${DIM} \
                       --heads ${HEADS} \
                       --core_dim ${DIM}
done
done
done
done
done

