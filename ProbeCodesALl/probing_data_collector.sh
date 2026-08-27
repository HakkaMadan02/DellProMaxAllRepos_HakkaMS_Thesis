#!/bin/bash
set -e

BASE_PATH="<#enter your own path0>"
LOG_FILE="run_mappo_20_40.log"

for n in $(seq 20 40); do
    echo "Running n=$n obs=$n" | tee -a "$LOG_FILE"
    python test_mappo.py \
        -n "$n" \
        --obs "$n" \
        --env DoubleIntegrator \
        --algo mappo \
        --area-size 4 \
        --cpu \
        --log \
        --epi 2 \
        --no-video \
        --path "$BASE_PATH" >> "$LOG_FILE" 2>&1
done