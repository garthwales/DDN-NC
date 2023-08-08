import numpy as np
from datetime import datetime
import torch
import torch.nn as nn

from eigen import EigenDecompositionFcn

class EDNetwork(nn.Module):
    """Example eigen decomposition network comprising a MLP data processing layer followed by a
    differentiable eigen decomposition layer. Input is (B, Z, 1); output is (B, M, M)."""

    def __init__(self, dim_z, m, method='exact', top_k=None, matrix_type='psd'):
        super(EDNetwork, self).__init__()

        self.dim_z = dim_z
        self.m = m
        self.method = method
        self.top_k = None # TODO: use this (or something similar) to test further..
        self.matrix_type = matrix_type

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 20),
            nn.ReLU(),
            nn.Linear(20, m * m)
        )

    def forward(self, z):
        # construct input for declarative node
        z = z.flatten(start_dim=1) # bxnxn -> bxdim_z
        # assert z.shape[1] == self.dim_z
        
        x = self.mlp(z)
        x = torch.reshape(x, (z.shape[0], self.m, self.m))

        if self.matrix_type == 'general':
            pass
        elif self.matrix_type == 'psd':
            x = torch.matmul(x, x.transpose(1, 2)) # positive definite
        elif self.matrix_type == 'rank1':
            u = x
            x = torch.matmul(u[:, :, 0], u[:, :, 0].transpose(1, 2))
        else:
            assert False, "unknown matrix_type"

        try:
            if self.method == 'pytorch':
                x = 0.5 * (x + x.transpose(1, 2))
                v, y = torch.linalg.eigh(x)
            elif self.method == 'exact':
                y = EigenDecompositionFcn().apply(x, self.top_k)
            else:
                assert False
        except:
            date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(x, f'eigh-illconditioned-{date_string}.pth')
            print(f'ill-conditioned input saved to eigh-illconditioned-{date_string}.pth')
            raise
        return y
    
class EDNetwork2(nn.Module):
    # TODO: make this into UNet..
    
    
    """Example eigen decomposition network comprising a UNet data processing layer followed by a
    differentiable eigen decomposition layer. Input is (B, Z, 1); output is (B, M, M)."""

    def __init__(self, dim_z, m, method='exact', top_k=None, matrix_type='psd'):
        super(EDNetwork, self).__init__()

        self.dim_z = dim_z
        self.m = m
        self.method = method
        self.top_k = None # TODO: use this (or something similar) to test further..
        self.matrix_type = matrix_type

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 20),
            nn.ReLU(),
            nn.Linear(20, m * m)
        )

    def forward(self, z):
        # construct input for declarative node
        z = z.flatten(start_dim=1) # bxnxn -> bxdim_z
        # assert z.shape[1] == self.dim_z
        
        x = self.mlp(z)
        x = torch.reshape(x, (z.shape[0], self.m, self.m))

        if self.matrix_type == 'general':
            pass
        elif self.matrix_type == 'psd':
            x = torch.matmul(x, x.transpose(1, 2)) # positive definite
        elif self.matrix_type == 'rank1':
            u = x
            x = torch.matmul(u[:, :, 0], u[:, :, 0].transpose(1, 2))
        else:
            assert False, "unknown matrix_type"

        try:
            if self.method == 'pytorch':
                x = 0.5 * (x + x.transpose(1, 2))
                v, y = torch.linalg.eigh(x)
            elif self.method == 'exact':
                y = EigenDecompositionFcn().apply(x, self.top_k)
            else:
                assert False
        except:
            date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(x, f'eigh-illconditioned-{date_string}.pth')
            print(f'ill-conditioned input saved to eigh-illconditioned-{date_string}.pth')
            raise
        return y