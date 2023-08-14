import torch
import torch.nn as nn

from datetime import datetime
from nets.eigen import EigenDecompositionFcn
from utils.utils import save_plot_imgs

class GenericNC(nn.Module):
    def __init__(self, net, n, 
                 net_name = 'unspecified', matrix_type='psd',
                 method='exact', width=-1, laplace=None):
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
        self.lapalce = laplace # None, basic, symm, symm2
        
        self.forward_calls = 0
        self.width = width
        self.w = width if width != -1 else n*n
        
    def forward(self, z):
        # pre-nc network
        x = self.net(z) # output b, n*n
        # TODO: force a relu here? make it always positive inputs into next?
        
        # make square b,m,m
        x = torch.reshape(x, (z.shape[0], self.w, self.n*self.n)) 
        
        # if it is a smaller width (e.g. 100x1024 instead of 1024x1024, reshape into full matrix with 100 on diagonals)
        if self.width != -1:
            # move this square slice into diagonal
            reconst = torch.zeros((z.shape[0],self.w,self.N), device=x.device)
            # diags = r + 1 # include the main diagonal of ones
            for b in range(z.shape[0]):
                for i  in range(0, self.w):
                    # if i == 0: # add the main diagonal (of all ones)
                        # reconst[b] = torch.add(reconst[b], torch.eye(N, device=out.device))
                    # else: # add the symmetric non-main diagonals
                    diagonal = x[b][i-1]
                    temp = torch.diag(diagonal[:self.n*self.n-i], i).to(x.device) # [:N-i] trims to fit the index'th diag size, places into index'th place
                    reconst[b] = torch.add(reconst[b], temp) # add the upper diagonal (or middle if 0)
                    if i != 0: # only need two when not the middle
                        temp = torch.diag(diagonal[:self.n*self.n-i], -i).to(x.device)
                        reconst[b] = torch.add(reconst[b], temp) # add the lower diagonal (symmetric)
            x = reconst
        
        if self.forward_calls % 10 == 0:
            save_plot_imgs(x.detach().cpu().numpy(), output_name=f'weights-{self.net_name}-{self.forward_calls}', output_path='figures/')
        self.forward_calls += 1
        
        # re-format square matrix into specified type
        if self.width == -1:
            if self.matrix_type == 'general':
                pass
            elif self.matrix_type == 'psd':
                x = torch.matmul(x, x.transpose(1, 2)) # x = x @ x.T
            else:
                assert False, "unknown matrix_type"

        if self.laplace is not None:
            x = get_laplace(x, self.laplace)
        
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

def get_laplace(x, laplace):
    d = x.sum(1)
    D = torch.diag(d)
    L = D-x
    if laplace == 'basic':
        return L
    if laplace == 'symm':
        D_inv_sqrt = torch.diag_embed(torch.where(d>0, d.pow(-0.5), 0))
        return torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...jk->...ik', D_inv_sqrt , L) , D_inv_sqrt)
    if laplace == 'symm2':
        D_inv_sqrt = torch.diag_embed(torch.where(d>0, d.pow(-0.5), 0))
        return 1 - torch.einsum('...ij,...jk->...ik', torch.einsum('...ij,...jk->...ik', D_inv_sqrt , x) , D_inv_sqrt) 
    assert 'incorrect laplace provided'

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
    
class BasicCNN(nn.Module):
    def __init__(self, n, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(3 * n * n, 1 * n * n)
        self.fc2 = nn.Linear(1 * n * n, out_size)
        
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(self.conv2(x))
        x = nn.ReLU(self.conv3(x))
        x = nn.ReLU(self.conv4(x))
        x = nn.ReLU(self.conv5(x))
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc1(x)
        x = self.fc2(x)
        return x