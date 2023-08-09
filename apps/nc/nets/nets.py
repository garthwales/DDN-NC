import torch
import torch.nn as nn

from datetime import datetime
from nets.eigen import EigenDecompositionFcn

class GenericNC(nn.Module):
    def __init__(self, net, n, 
                 net_name = 'unspecified', matrix_type='psd',
                 method='exact'):
        # TODO: could add more params like conditioning... vec flip options...
        super(GenericNC, self).__init__()
        
        # pre-nc network
        self.net = net 

        # based on input image size of n,n 
        # so output of netis n*n,n*n
        # and output after eig is n,n
        self.n = n 
        
        self.net_name = net_name
        self.matrix_type = matrix_type # general, psd
        self.method = method # exact, pytorch
        
        
    def forward(self, z):
        # pre-nc network
        x = self.net(z) # output b, n*n
        # TODO: force a relu here? make it always positive inputs into next?
        
        # make square b,n,n
        x = torch.reshape(x, (z.shape[0], self.n, self.n)) 
        
        # re-format square matrix into specified type
        if self.matrix_type == 'general':
            pass
        elif self.matrix_type == 'psd':
            x = torch.matmul(x, x.transpose(1, 2)) # x = x @ x.T
        else:
            assert False, "unknown matrix_type"
            
        # NOTE: 0.5 * (X + X.transpose(1, 2))
        #       is done before doing either of the eigensolvers no matter what
        #       maybe that should be done with matrix_type?
        try:
            if self.method == 'pytorch':
                x = 0.5 * (x + x.transpose(1, 2))
                v, y = torch.linalg.eigh(x)
            elif self.method == 'exact':
                y = EigenDecompositionFcn().apply(x)
            else:
                assert False
        except Exception as err:
            date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
            torch.save(x, f'{date_string}-{type(err).__name__}.pth')
            print(f'{date_string}-{err}.pth')
            raise
        
        return y

class BasicMLP(nn.Module):
    """ Multi-layer perceptron to pass before NC. """

    def __init__(self, dim_z, m, k=1000, j=20):
        """_summary_

        Args:
            dim_z (int): input size (e.g. b,c,n,n -> dim_z = c*n*n)
            m (int): output is a m*m weight matrix
            k (int, optional): linear layers size. Defaults to 1000.
            j (int, optional): last linear layers size to avoid too many parameters. Defaults to 20.
        """
        super(BasicMLP, self).__init__()

        self.dim_z = dim_z
        self.m = m
        
        self.mlp = nn.Sequential(
            nn.Linear(dim_z, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.ReLU(),
            nn.Linear(k, j),
            nn.ReLU(),
            nn.Linear(j, m * m)
        )

    def forward(self, z):
        """ Assumes input is b,n,n """
        z = z.flatten(start_dim=1)
        return self.mlp(z)