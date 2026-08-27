#!/usr/bin/env bash
set -e

python test.py --epi 2 --area-size 4 -n 65 --obs 50 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000

python test.py --epi 2 --area-size 4 -n 70 --obs 50 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000

python test.py --epi 2 --area-size 4 -n 75 --obs 60 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000

python test.py --epi 2 --area-size 4 -n 100 --obs 50 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000

python test.py --epi 2 --area-size 4 -n 80 --obs 60 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000

python test.py --epi 2 --area-size 4 -n 90 --obs 70 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Desktop/gcbfplus-original/gcbfplus/pretrained/DoubleIntegrator/gcbf+/ --step 1000