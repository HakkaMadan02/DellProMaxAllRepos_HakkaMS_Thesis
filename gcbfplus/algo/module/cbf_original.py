import flax.linen as nn
import functools as ft
import jax.numpy as jnp
import numpy as np
import jax

from typing import Type, Optional, Tuple
from ...nn.gnn import GNN
from ...nn.mlp import MLP
from ...nn.utils import default_nn_init
from ...utils.typing import Array, Params
from ...utils.graph import GraphsTuple


class CBFNet(nn.Module):
    gnn_cls: Type[GNN]
    head_cls: Type[nn.Module]

    @nn.compact
    def __call__(self, obs: GraphsTuple, n_agents: int, *args, **kwargs) -> Array:

        gnn_output, gnn_intermediates= self.gnn_cls()(obs, node_type=0, n_type=n_agents, return_intermediates=True)
        #print("GNN output shape", gnn_output.shape)
        x1 = self.head_cls()(gnn_output)
        x = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x1))
        #print("Empty Test 1", x, x1)
        #print("Empty test 2", x, x1)
        #print("cbf shape, intermediate shape", x.shape, x1.shape)
        return x, x1, gnn_intermediates
        
        """ x = self.gnn_cls()(obs, node_type=0, n_type=n_agents)
        x1 = self.head_cls()(x)
        print("Empty Test 1", x, x1)
        x = nn.tanh(nn.Dense(1, kernel_init=default_nn_init())(x1))
        print("Empty test 2", x, x1)
        print("cbf shape, intermediate shape", x.shape, x1.shape)
        return x, x1 """
    



class CBF:

    def __init__(self, node_dim: int, edge_dim: int, n_agents: int, gnn_layers: int):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.n_agents = n_agents

        self.cbf_gnn = ft.partial(
            GNN,
            msg_dim=128,
            hid_size_msg=(256, 256),
            hid_size_aggr=(128, 128),
            hid_size_update=(256, 256),
            out_dim=128,
            n_layers=gnn_layers
        )
        self.cbf_head = ft.partial(
            MLP,
            hid_sizes=(256, 256),
            act=nn.relu,
            act_final=False,
            name='CBFHead'
        )
        self.net = CBFNet(
            gnn_cls=self.cbf_gnn,
            head_cls=self.cbf_head
        )

    #key = jax.random.PRNGKey(0)

    def get_cbf_all(self, params: Params, obs: GraphsTuple) -> Array:
    
        #print(type(params), type(self.n_agents), type(obs))
        #params = self.net.init(self.key, obs, self.n_agents)
        #print(type(params), type(self.n_agents), type(obs))
        return self.net.apply(params, obs, self.n_agents) 
        #return self.net.apply(params, obs, self.n_agents)
    
    '''def get_cbf(self, params: Params, graph: GraphsTuple, agent_id: int) -> Array:
        """Compute CBF value for a specific agent"""
        # Remove the params=None handling since we require explicit params
        h_all = self.net.apply(params, graph, agent_id, graph.n_node[()])
        return h_all[agent_id]
    
    # Convert agent_id to a JAX array if needed
        agent_id_array = jnp.array(agent_id)
    
    # Apply the network
        h = self.net.apply(params, graph, agent_id_array, self.n_agents)
        return h.squeeze()  # Remove any extra dimensions'''

    def get_cbf(self, params, obs):
        return self.get_cbf_all(params, obs)[0]

    def get_cbf_in(self, params, obs):
        return self.get_cbf_all(params, obs)[1] 
    
    # def get_cbf_inner_layer(self, params: Params, obs: GraphsTuple) -> Array:
    #     return self.net.apply(self, params, obs).get_in()
    # return self.net.apply(params, obs, self.n_agents)[1]
    

    # In cbf.py, add these methods to the CBF class:

    ###CODE BLOACK TO GET MLP 1 - 2 - 3 DATA ###

    def get_gnn_msg_intermediate(self, params, obs):
        """Extract message MLP 2nd-last layer"""
        _, _, intermediates = self.net.apply(params, obs, self.n_agents)
        return intermediates.get('msg', None)

    def get_gnn_aggr_intermediate(self, params, obs):
        """Extract aggregation MLP 2nd-last layer"""
        _, _, intermediates = self.net.apply(params, obs, self.n_agents)
        return intermediates.get('aggr', None)

    def get_gnn_update_intermediate(self, params, obs):
        """Extract update MLP 2nd-last layer"""
        _, _, intermediates = self.net.apply(params, obs, self.n_agents)
        return intermediates.get('update', None)

    ###CODE BLOCK TO GET ATTENTION SCORES ###

    def get_attention_weights(self, params, obs):
        """Extract post-softmax attention weights"""
        _, _, intermediates = self.net.apply(params, obs, self.n_agents)
        return intermediates.get('attn', None)
    
    def get_receiver_indices(self, params, obs):
        """Extract receiver indices for attention weights"""
        _, _, intermediates = self.net.apply(params, obs, self.n_agents)
        return intermediates.get('recv_idx', None)

