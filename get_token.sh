#!/usr/bin/env bash
docker compose exec -it dev jupyter server list | grep -o -P "(?<=\?token=)[a-z0-9]+"
