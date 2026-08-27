#!/bin/bash
set -e


python test_mappo.py --epi 32 --area-size 4 -n 8 --obs 0  --no-video --env DoubleIntegrator --algo mappo --log  --path /home/kunalgarg/Desktop/gcbf-jax-main_trainer/Server_Logs/DoubleIntegrator/mappo/2Layers_15000 --step 14660
python test_mappo.py --epi 32 --area-size 4 -n 16 --obs 0  --no-video --env DoubleIntegrator --algo mappo --log  --path /home/kunalgarg/Desktop/gcbf-jax-main_trainer/Server_Logs/DoubleIntegrator/mappo/2Layers_15000 --step 14660
python test_mappo.py --epi 32 --area-size 4 -n 24 --obs 0 --no-video --env DoubleIntegrator --algo mappo --log  --path /home/kunalgarg/Desktop/gcbf-jax-main_trainer/Server_Logs/DoubleIntegrator/mappo/2Layers_15000 --step 14660         
python test_mappo.py --epi 32 --area-size 4 -n 32 --obs 0 --no-video --env DoubleIntegrator --algo mappo --log  --path /home/kunalgarg/Desktop/gcbf-jax-main_trainer/Server_Logs/DoubleIntegrator/mappo/2Layers_15000 --step 14660         
