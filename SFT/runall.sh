#!/bin/bash
set -e


python test.py --epi 32 --area-size 4 -n 8 --obs 8 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Downloads/1880_FIND_R_Failure_Identifica_SupplementaryMaterials/CoRL_Supplementary/Probed_GCBFPlus/Probed_GCBFplus/pretrained/DoubleIntegrator/gcbf+pretrained --step 1000 --max-step 1024
python test.py --epi 32 --area-size 4 -n 16 --obs 16 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Downloads/1880_FIND_R_Failure_Identifica_SupplementaryMaterials/CoRL_Supplementary/Probed_GCBFPlus/Probed_GCBFplus/pretrained/DoubleIntegrator/gcbf+pretrained --step 1000 --max-step 1024
python test.py --epi 32 --area-size 4 -n 24 --obs 24 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Downloads/1880_FIND_R_Failure_Identifica_SupplementaryMaterials/CoRL_Supplementary/Probed_GCBFPlus/Probed_GCBFplus/pretrained/DoubleIntegrator/gcbf+pretrained --step 1000 --max-step 1024
python test.py --epi 32 --area-size 4 -n 32 --obs 32 --no-video --env DoubleIntegrator --algo gcbf+ --log --cbf 0 --path /home/kunalgarg/Downloads/1880_FIND_R_Failure_Identifica_SupplementaryMaterials/CoRL_Supplementary/Probed_GCBFPlus/Probed_GCBFplus/pretrained/DoubleIntegrator/gcbf+pretrained --step 1000 --max-step 1024