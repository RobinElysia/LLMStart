from .LocalLoadModel import load_model as load_local_model, gen_resp as gen_local_resp
from .RemoteLoadModel import RemoteLoadModel

__all__ = ['load_local_model', 'gen_local_resp', 'RemoteLoadModel']