
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from matrix import *

def extract_3x3_patch(image, x, y):
    # Ensure the input image is a torch tensor
    if not torch.is_tensor(image):
        image = torch.tensor(image)

    # Check shape and transpose if needed
    # if len(image.shape) == 3 and image.shape[0] <= 3:
    #     image = image.permute(1, 2, 0)  # equivalent of np.transpose for torch tensor
    
    H, W = image.shape

    # Calculate padding
    left_pad = max(0, 1 - x)
    right_pad = max(0, x + 2 - W)
    top_pad = max(0, 1 - y)
    bottom_pad = max(0, y + 2 - H)

    # Pad the image
    padded_image = torch.nn.functional.pad(image, (left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)

    # Extract the 3x3 patch
    patch = padded_image[y - 1 + top_pad:y + 2 + top_pad, x - 1 + left_pad:x + 2 + left_pad]
    
    return patch.float().unsqueeze(0).unsqueeze(0)

class LeNetVariant(nn.Module):
    def __init__(self):
        super(LeNetVariant, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 2, 3)  # Using a 3x3 kernel, output: 6x3x3
        # self.conv2 = nn.Conv2d(32, 16, 3)  # 1x1 spatial dimension left after this
        
        self.fc1 = nn.Linear(2, 9)  # The spatial dimension is 1x1 after the convolutions
        self.fc2 = nn.Linear(9, 9)
        self.fc3 = nn.Linear(9, 1)  # This can be adjusted based on your final desired output size

    def forward(self, x):
        x = (self.conv1(x))
        # x = F.relu(self.conv2(x))
        
        x = x.view(-1, 2)  # Flatten
        x = (self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

class SiameseNetworkLeNet(nn.Module):
    def __init__(self):
        super(SiameseNetworkLeNet, self).__init__()
        self.subnetwork = LeNetVariant()

    def forward(self, input1, input2):
        output1 = self.subnetwork(input1)
        output2 = self.subnetwork(input2)
        
        scalar = output1 * output2
        dissimilarity_score = torch.sigmoid(scalar)
                
        return dissimilarity_score

class EigenDecompositionFcn(torch.autograd.Function):
    """PyTorch autograd function for eigen decomposition of real symmetric matrices. Returns all eigenvectors
    or just eigenvectors associated with the top-k eigenvalues. The input matrix is made symmetric within the
    forward evaluation function."""

    eps = 1.0e-9 # tolerance to consider two eigenvalues equal

    @staticmethod
    def forward(ctx, X, top_k=None):
        B, M, N = X.shape
        assert N == M
        assert (top_k is None) or (1 <= top_k <= M)

        with torch.no_grad():
            lmd, Y = torch.linalg.eigh(X,UPLO='U')

        ctx.save_for_backward(lmd, Y)
        return Y if top_k is None else Y[:, :, -top_k:] # TODO: only return the second smallest for NC

    @staticmethod
    def backward(ctx, dJdY):
        lmd, Y = ctx.saved_tensors
        B, M, K = dJdY.shape

        zero = torch.zeros(1, dtype=lmd.dtype, device=lmd.device)
        L = lmd[:, -K:].view(B, 1, K) - lmd.view(B, M, 1)
        L = torch.where(torch.abs(L) < EigenDecompositionFcn.eps, zero, 1.0 / L)
        dJdX = torch.bmm(torch.bmm(Y, L * torch.bmm(Y.transpose(1, 2), dJdY)), Y[:, :, -K:].transpose(1, 2))
        
        return dJdX, None

class MatrixNet(nn.Module):
    """
    For one image, compare 3x3 patches within to produce a weights matrix
    """
    def __init__(self, shape, d=1, device='cpu'):
        super(MatrixNet, self).__init__()
        self.d = d
        self.n = shape[2] # C,W,H or W,H,C work with [2] but should be W,H,C within
        distance_matrix = distance_manhattan(self.n) 
        self.pairs = get_pairs_within_threshold(distance_matrix, self.d)
        self.net = SiameseNetworkLeNet()
        self.nc = EigenDecompositionFcn()
        self.device = device
        
    def forward(self, image):
        x = torch.zeros((self.n*self.n, self.n*self.n), device=self.device)
        for num,pair in enumerate(self.pairs):
            i,j = pair
            patch1 = extract_3x3_patch(image, i//self.n, i % self.n) # TODO: verify if this part is correct.. for now ignore
            patch2 = extract_3x3_patch(image, i//self.n, j % self.n)
            x[i][j] = self.net(patch1, patch2)
            
            if num % 500 == 0:
                    plt.imshow(patch1.detach().cpu().numpy()[0][0])
                    plt.savefig(f'outputs/{num}-patch1.pdf', dpi=300, bbox_inches='tight')
                    plt.imshow(patch2.detach().cpu().numpy()[0][0])
                    plt.savefig(f'outputs/{num}-patch2-{x[i][j]}.pdf', dpi=300, bbox_inches='tight')
        return x
    
class NC(nn.Module):
    """
    Take a weights matrix and compute eigenvector
    """
    
    def forward(self, x):
        y = EigenDecompositionFcn().apply(x)[:, 1] # second smallest eigenvector.. no batch dim
        return y
        
d = 1
lr = 1e-1
epochs = 10
    
rand_inputs = torch.rand(10,20,20)
rand_inputs = torch.where(rand_inputs > 0.5, 1.0, 0.0)
rand_targets = []
distance_matrix = distance_manhattan(rand_inputs.shape[2]) 
for i, values in enumerate(rand_inputs):
    x = dissimilarity_torch(values)
    x = torch.triu(x, 1)
    
    pairs = get_pairs_outside_threshold(distance_matrix, 1)
    for i,j in pairs:
        x[i][j] = 0
    
    rand_targets.append(x)
rand_targets = torch.stack(rand_targets)

criterion = nn.MSELoss(reduction='mean')
device = 'cuda:1'


# loop through batches
i = 0
trials = 1
learning_curves = [[] for i in range(trials)]
for trial in range(trials):
    torch.manual_seed(22 + trial)
    model = MatrixNet(rand_inputs.shape, d=d, device=device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    
    for epoch in range(epochs):
        print(f'Epoch {epoch}')
        for (inputs, labels) in zip(rand_inputs, rand_targets):
            i += 1
            inputs = inputs.to(device)
            # inputs = inputs.unsqueeze(-1) # add C dimension of 1 to the B/W images.
            inputs.requires_grad_(True)
            labels = labels.to(device)

            # passes and weights update
            with torch.set_grad_enabled(True):
                
                # forward pass 
                preds = model(inputs)
                loss  = criterion(preds, labels)
                
                print(f'{i} Loss: {loss.item()}')
                if i % 10 == 0:
                    plt.imshow(inputs.detach().cpu().numpy())
                    plt.savefig(f'outputs/{i}-input.pdf', dpi=300, bbox_inches='tight')
                    
                    plt.imshow(labels.detach().cpu().numpy())
                    plt.savefig(f'outputs/{i}-labels.pdf', dpi=300, bbox_inches='tight')
                    
                    plt.imshow(preds.detach().cpu().numpy())
                    plt.savefig(f'outputs/{i}-preds.pdf', dpi=300, bbox_inches='tight')

                # backward pass
                loss.backward() 

                # weights update
                optimizer.step()
                optimizer.zero_grad()
                
                learning_curves[trial].append(float(loss.item()))
                
for trial in range(len(learning_curves)):
    plt.plot(learning_curves[trial], 'b', alpha=0.1)
plt.xlabel('iter.'); plt.ylabel('loss')
plt.savefig(f'outputs/trials.pdf', dpi=300, bbox_inches='tight')