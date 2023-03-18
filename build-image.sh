#!/usr/bin/env bash
docker pull tensorflow/tensorflow:latest-gpu
docker build . \
 -f Dockerfile.nvflare-hysped-dev \
 -t iisas/nvflare-hysped-dev:latest \
 --build-arg NVFLARE_BRANCH=main
