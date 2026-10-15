"""Fixed input transformation; train=True never mutates policy input statistics."""
import torch

CONTRACT = 'fixed_asinh_raw_value_zero_critic.v1'


class FixedAsinh(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.register_buffer('contract_version', torch.tensor(1, dtype=torch.int64))

    def forward(self, x, *, train=False, inverse=False, **kwargs):
        if x is None: return None
        if inverse: raise ValueError('observation inverse transform is not part of this contract')
        return torch.asinh(x)
