#!/bin/bash
set -e


python test.py --epi 32 --area-size 4 -n 8 --obs 8 --no-video --env DoubleIntegrator --cbf 0 --algo gcbf+ --log --path /home/kunalgarg/Downloads/Radial_SingleHead_Ablation/ --step 760
python test.py --epi 32 --area-size 4 -n 16 --obs 16 --no-video --env DoubleIntegrator --cbf 0 --algo gcbf+ --log --path /home/kunalgarg/Downloads/Radial_SingleHead_Ablation/ --step 760
python test.py --epi 32 --area-size 4 -n 24 --obs 24 --no-video --env DoubleIntegrator --cbf 0 --algo gcbf+ --log --path /home/kunalgarg/Downloads/Radial_SingleHead_Ablation/ --step 760
python test.py --epi 32 --area-size 4 -n 32 --obs 32 --no-video --env DoubleIntegrator --cbf 0 --algo gcbf+ --log --path /home/kunalgarg/Downloads/Radial_SingleHead_Ablation/ --step 760