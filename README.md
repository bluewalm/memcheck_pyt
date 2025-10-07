
# Memory benchmarking attention

This repository contains simple scripts to benchmark various attention mechanisms. 
It was intended to illustrate the memory requirements of the compressed self-attention layer from the bluewalm module. 
The bluewalm module itself can be obtained from [here](https://www.bluewalm.com). 

This repository is tested and maintained by BLUEWALM. 

Table of Contents
=================
  * [Quick Start Guide](#quick-start-guide)
  * [Scripts](#scripts)
  * [Metrics (1x A100 40GB)](#metrics-1x-a100-40gb)
  * [Release notes](#release-notes)
     * [Changelog](#changelog)
     * [Known issues](#known-issues)


## Quick Start Guide

To run the benchmark scripts perform the following steps. 

1. Obtain the bluewalm pytorch module from us. You can find us at [this place](https://www.bluewalm.com). 
Build the bluewalm Docker container with the Dockerfile obtained from us. 

2. Clone the repository. 
```bash
git clone https://github.com/bluewalm/memcheck_pyt.git
```

3. Now, enter the repository. 
```bash
cd memcheck_pyt
```

9. Start the bluewalm Docker container.
```bash
docker run -it --rm --gpus device=all --net=host --shm-size=32gb --ulimit memlock=-1 --cap-add=SYS_ADMIN --ulimit stack=67108864 -v "${PWD}:/workspace" bluewalm_pyt:latest
```
This will launch an interactive container and mount the current directory as a volume to the `/workspace` directory inside the container. 
Any datasets, checkpoints and deployed models saved to `/workspace` will be accessible in the corresponding directory on the host. 

10. Run the gradcheck script to check if the gradient computation in the bluewalm module is correct. 
```bash
python gradcheck.py
```

11. Benchmark the memory use of attention mechanisms. This only benchmarks the forward and backward passes of the attention operators. Nothing else. 
```bash
./profile_attention.sh
```

12. Benchmark the memory use of combinator layer. 
```bash
./profile_combinator.sh
```

## Scripts

| file                  | purpose                                                                 |
|-----------------------|-------------------------------------------------------------------------|
| gradcheck.py          | check the correctness of the gradient computations                      |
| profile_attention.sh  | profiles a bunch of attention operators with a number of configurations |
| profile_combinator.sh | profile the combinator layer with a number of configurations            |                                |
| profile_attention.py  | profile an attention operator with the specified configuration          |
| profile_combinator.py | profile the combinator layer with the specified configuration           |


## Metrics (1x A100 40GB)

|--------------------------------------------------|--------------------------------------------------|
| ![seqlen1024](https://github.com/bluewalm/memcheck_pyt/blob/master/images/sequence%20length%3D1024.png) | ![seqlen2048](https://github.com/bluewalm/memcheck_pyt/blob/master/images/sequence%20length%3D2048.png) |
| ![seqlen4096](https://github.com/bluewalm/memcheck_pyt/blob/master/images/sequence%20length%3D4096.png) | ![seqlen8192](https://github.com/bluewalm/memcheck_pyt/blob/master/images/sequence%20length%3D8192.png) |
|--------------------------------------------------|--------------------------------------------------|
The above figures show how memory requirement changes in the token dimension. 
As we can see, memory requirement is linear in the token dimension. 
Furthermore, softplus attention is storing more intermediate results. 

|---------------------------------------------|-----------------------------------------------|
| ![dim128](https://github.com/bluewalm/memcheck_pyt/blob/master/images/token%20dimension%3D128.png) | ![dim256](https://github.com/bluewalm/memcheck_pyt/blob/master/images/token%20dimension%3D256.png)   |
| ![dim512](https://github.com/bluewalm/memcheck_pyt/blob/master/images/token%20dimension%3D512.png) | ![dim1024](https://github.com/bluewalm/memcheck_pyt/blob/master/images/token%20dimension%3D1024.png) |
|---------------------------------------------|-----------------------------------------------|
The above figures show how memory requirement changes in the sequence length. 
We can see that for softplus attention the memory requirement is linear in the sequence length. 


## Release notes

### Changelog

October 07, 2025
    * Initial release

### Known issues

    * None
