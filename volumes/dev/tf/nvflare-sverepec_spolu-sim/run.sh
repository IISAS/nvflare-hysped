#!/usr/bin/env bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=false && nvflare simulator -w workspace -n 4 -t 4 ./app
