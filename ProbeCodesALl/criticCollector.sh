#!/bin/bash
set -e

BASE_PATH="Enter your path"
LOG_FILE="run_probe_collection.log"
EPISODES=20

echo "========================================" | tee -a "$LOG_FILE"
echo "Probe Data Collection - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Config A: n=8, obs=8 
echo "Config A: n=8 obs=6" | tee -a "$LOG_FILE"
python test_mappo.py \
    -n 8 --obs 6 \
    --env DoubleIntegrator --algo mappo \
    --area-size 4 --cpu --log \
    --epi $EPISODES --no-video \
    --path "$BASE_PATH" >> "$LOG_FILE" 2>&1

# Config B: n=6, obs=8
echo "Config B: n=6 obs=8" | tee -a "$LOG_FILE"
python test_mappo.py \
    -n 6 --obs 8 \
    --env DoubleIntegrator --algo mappo \
    --area-size 4 --cpu --log \
    --epi $EPISODES --no-video \
    --path "$BASE_PATH" >> "$LOG_FILE" 2>&1


echo "========================================" | tee -a "$LOG_FILE"
echo "Collection complete - $(date)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
