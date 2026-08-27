XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 8 --alpha 0.1
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 16 --alpha 0.1
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 32 --alpha 0.1
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 64 --alpha 0.1
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 8 --alpha 1.0
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 16 --alpha 1.0
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 32 --alpha 1.0
XLA_PYTHON_CLIENT_PREALLOCATE=false python test.py --env DoubleIntegrator --area-size 8 --no-video --epi 32 --log --obs 0 --nojit-rollout --max-step 2048 --algo dec_share_cbf_k -n 64 --alpha 1.0
