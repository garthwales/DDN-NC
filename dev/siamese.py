
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

def extract_3x3_patch(image, x, y):
    # Ensure the input image is a torch tensor
    if not torch.is_tensor(image):
        image = torch.tensor(image)

    # Check shape and transpose if needed
    if len(image.shape) == 3 and image.shape[0] <= 3:
        image = image.permute(1, 2, 0)  # equivalent of np.transpose for torch tensor
    
    H, W, C = image.shape

    # Calculate padding
    left_pad = max(0, 1 - x)
    right_pad = max(0, x + 2 - W)
    top_pad = max(0, 1 - y)
    bottom_pad = max(0, y + 2 - H)

    # Pad the image
    padded_image = torch.nn.functional.pad(image, (0, 0, left_pad, right_pad, top_pad, bottom_pad), mode='constant', value=0)

    # Extract the 3x3 patch
    patch = padded_image[y - 1 + top_pad:y + 2 + top_pad, x - 1 + left_pad:x + 2 + left_pad, :]
    
    return patch.float()

class LeNetVariant(nn.Module):
    def __init__(self):
        super(LeNetVariant, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3)  # Using a 3x3 kernel, output: 6x3x3
        # self.conv2 = nn.Conv2d(32, 16, 3)  # 1x1 spatial dimension left after this
        
        self.fc1 = nn.Linear(32, 256)  # The spatial dimension is 1x1 after the convolutions
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)  # This can be adjusted based on your final desired output size

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        
        x = x.view(-1, 32)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

class SiameseNetworkLeNet(nn.Module):
    def __init__(self):
        super(SiameseNetworkLeNet, self).__init__()
        self.subnetwork = LeNetVariant()

    def forward(self, input1, input2):
        output1 = self.subnetwork(input1)
        output2 = self.subnetwork(input2)
        
        # Compute the L2 distance between the two outputs
        euclidean_distance = output1 * output2
        
        # Convert the distance to a similarity score between 0 and 1
        dissimilarity_score = torch.sigmoid(euclidean_distance)
        
        return dissimilarity_score
    
def distance_manhattan(N):
    distances = np.zeros((N * N, N * N), dtype=int)

    for u in range(N * N):
        for v in range(N * N):
            coord1 = np.array(divmod(u, N))
            coord2 = np.array(divmod(v, N))
            distances[u, v] = np.sum(np.abs(coord1 - coord2))
    return distances

def get_pairs_within_threshold(distance_matrix, threshold):
    # Get the upper triangle indices
    i, j = np.triu_indices(distance_matrix.shape[0], k=1)
    
    # Set the diagonal to values greater than the threshold
    np.fill_diagonal(distance_matrix, threshold + 1)
    
    # Filter the pairs based on the threshold
    mask = distance_matrix[i, j] < threshold
    
    # Get the pairs where the condition is met
    pairs = list(zip(i[mask], j[mask]))

    return pairs

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
    def __init__(self, shape, d=1):
        super(MatrixNet, self).__init__()
        self.d = d
        self.n = shape[2] # C,W,H or W,H,C work with [2] but should be W,H,C within
        distance_matrix = distance_manhattan(self.n) 
        self.pairs = get_pairs_within_threshold(distance_matrix, self.d)
        self.net = SiameseNetworkLeNet()
        self.nc = EigenDecompositionFcn()
        
    def forward(self, image):
        x = torch.zeros((self.n*self.n, self.n*self.n))
        for i,j in self.pairs:
            patch1 = extract_3x3_patch(image, i//self.n, i % self.n) # TODO: verify if this part is correct.. for now ignore
            patch2 = extract_3x3_patch(image, i//self.n, j % self.n)
            x[i][j] = self.net(patch1, patch2)
        return x
    
class NC(nn.Module):
    """
    Take a weights matrix and compute eigenvector
    """
    
    def forward(self, x):
        y = EigenDecompositionFcn().apply(x)[:, 1] # second smallest eigenvector.. no batch dim
        return y
    
rand_inputs = torch.randn(100,10,10,1)
model = MatrixNet(rand_inputs.shape, d=1)
# Learn matricies of b/w images for now


# loop through batches
for (inputs, labels) in data_loader:

    # extract inputs and labels
    inputs = inputs.to(device)
    labels = labels.to(device)

    # passes and weights update
    with torch.set_grad_enabled(True):
        
        # forward pass 
        preds = model(inputs)
        loss  = criterion(preds, labels)

        # backward pass
        loss.backward() 

        # weights update
        optimizer.step()
        optimizer.zero_grad()