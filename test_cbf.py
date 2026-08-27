from model.algo import GCBF, GCBFPlus, make_algo, QuadraticCBF
from model.env import make_env
import jax.random as jr

env = make_env(env_id='DoubleIntegrator', num_agents=2)
algo = make_algo(algo='quadratic_cbf',
                 env=env,
                 node_dim=env.node_dim,
                 edge_dim=env.edge_dim,
                 state_dim=env.state_dim,
                 action_dim=env.action_dim,
                 n_agents=env.num_agents)
test_key = jr.PRNGKey(1234)
graph = env.reset(test_key)
cbf = algo.get_cbf(graph)
action = algo.act(graph)
print(cbf)
aaa = 0
