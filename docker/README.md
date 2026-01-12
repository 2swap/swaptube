# SwapTube Docker Setup

This directory contains a Dockerfile to run SwapTube in a containerized environment with all dependencies preinstalled (CUDA, FFmpeg ≥5, graphics/math libraries, MicroTeX).

## TLDR Workflow

```bash
docker build -f docker/Dockerfile -t swaptube:cuda ..
docker volume create swaptube-build
docker run --rm -it --gpus all \
  -u $(id -u):$(id -g) \
  -v "$PWD:/workspace/swaptube" \
  -v swaptube-build:/workspace/swaptube/build \
  -w /workspace/swaptube \
  swaptube:cuda ./go.sh MyProject 640 360
```


## Note from 2swap:

I personally don't use this as I have gotten accustomed to building and running SwapTube natively on my system, and I intend to keep it that way to maintain a level of familiarity with external dependencies. This setup is contributed by the community, and I do not intend to maintain it. YMMV.

## Notes

Test GPU support

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-bookworm nvidia-smi
```

