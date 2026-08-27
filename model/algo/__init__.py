from .base import MultiAgentController
from .dec_share_cbf import DecShareCBF
from .dec_share_cbf_k import DecShareCBFK
from .gcbf import GCBF
from .gcbf_plus import GCBFPlus
from .centralized_cbf import CentralizedCBF
from .dec_opt_cbf import DecOptCBF
from .mappo import MAPPO
from .bc import BC
from .dec_share_cbf_k_u_constrained import DecShareCBFKUConstrained


def make_algo(algo: str, **kwargs) -> MultiAgentController:
    if algo == 'gcbf':
        return GCBF(**kwargs)
    elif algo == 'gcbf+':
        return GCBFPlus(**kwargs)
    elif algo == 'centralized_cbf':
        return CentralizedCBF(**kwargs)
    elif algo == 'dec_opt_cbf':
        return DecOptCBF(**kwargs)
    elif algo == 'dec_share_cbf':
        return DecShareCBF(**kwargs)
    elif algo == 'dec_share_cbf_k':
        return DecShareCBFK(**kwargs)
    elif algo == 'mappo':
        return MAPPO(**kwargs)
    elif algo == 'bc':
        return BC(**kwargs)
    elif algo == 'dec_share_cbf_u_constrained':
        return DecShareCBFKUConstrained(**kwargs)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
