#!/usr/bin/env bash
docker compose --env-file .env -p nvflare_hysped_fl up --build -d
