#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 && nvflare simulator -w workspace -n 4 -t 1 ./app
