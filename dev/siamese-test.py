import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2

def PlotResults(exact_curves, fcn=plt.semilogy):
    """plot results of experiments."""

    exact_mean = np.mean(exact_curves, axis=0)

    fcn(exact_mean, 'b')
    for trial in range(len(exact_curves)):
        fcn(exact_curves[trial], 'b', alpha=0.1)
    # fcn(exact_mean, 'b')
    plt.xlabel('iter.'); plt.ylabel('loss')
    plt.legend(('exact'))
    

def extract_5x5_patch(image, x, y):
    if len(image.shape) == 3 and image.shape[0] <= 3:
        image = np.transpose(image, (1, 2, 0))
    
    H, W, C = image.shape
    
    # Calculate padding
    left_pad = max(0, 2 - x)
    right_pad = max(0, x + 3 - W)
    top_pad = max(0, 2 - y)
    bottom_pad = max(0, y + 3 - H)
    
    # Pad the image
    padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    
    # Extract the 5x5 patch
    patch = padded_image[y - 2 + top_pad:y + 3 + top_pad, x - 2 + left_pad:x + 3 + left_pad, :]
    
    # Transpose the patch to (C, H, W) format
    patch = np.transpose(patch, (2, 0, 1))
    
    return torch.tensor(patch).float()

# def extract_3x3_patch(image, x, y):
#     if len(image.shape) == 3 and image.shape[0] <= 3:
#         image = np.transpose(image, (1, 2, 0))
    
#     H, W, C = image.shape
    
#     # Calculate padding
#     left_pad = max(0, 1 - x)
#     right_pad = max(0, x + 2 - W)
#     top_pad = max(0, 1 - y)
#     bottom_pad = max(0, y + 2 - H)
    
#     # Pad the image
#     padded_image = np.pad(image, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant')
    
#     # Extract the 3x3 patch
#     patch = padded_image[y - 1 + top_pad:y + 2 + top_pad, x - 1 + left_pad:x + 2 + left_pad, :]
    
#     return torch.tensor(patch).float()

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

def print_memory_usage():
    print("Memory Allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
    print("Cached Memory:", torch.cuda.memory_reserved() / 1e6, "MB")
        
def compute_difference_matrix(img):
    # Flatten the image to have each pixel as a row vector
    flattened = img.reshape(-1, 3).detach().numpy()

    # Calculate pairwise squared differences between pixels using broadcasting
    diff = np.sum((flattened[:, np.newaxis] - flattened[np.newaxis, :])**2, axis=-1)

    # A pixel is considered to be of the "same color" if its squared difference is 0
    # You can adjust the threshold if you want some tolerance
    threshold = 1e-6
    binary_diff = (diff > threshold).astype(int)

    return binary_diff

class MatrixNet(nn.Module):
    def __init__(self):
        super(MatrixNet, self).__init__()
        self.n = 10
        self.d = 3
        distance_matrix = distance_manhattan(self.n)
        self.pairs = get_pairs_within_threshold(distance_matrix, self.d)
        self.net = SiameseNetworkLeNet()
        
    def forward(self, image):
        x = torch.zeros((self.n*self.n, self.n*self.n))
        for i,j in self.pairs:
            patch1 = extract_3x3_patch(image, i//self.n, i % self.n)
            patch2 = extract_3x3_patch(image, i//self.n, j % self.n)
            x[i][j] = self.net(patch1, patch2)
        return x
    
print_memory_usage()
img = torch.tensor(cv2.imread('green-white-10x10.png')/255, requires_grad=True)
test = MatrixNet()

output = test(img)
print_memory_usage()


plt.imshow(compute_difference_matrix(img))
plt.savefig('target.pdf', dpi=300, bbox_inches='tight')

plt.imshow(output.detach().numpy())
plt.savefig('random.pdf', dpi=300, bbox_inches='tight')

plt.imshow(img.detach().numpy())
plt.savefig('input.pdf', dpi=300, bbox_inches='tight')

# Training loop parameters
if False:
    # Create dataset of green and red patches
    green = torch.zeros(1, 3, 3, 3)
    green[:, 1, :, :] = 1

    red = torch.zeros(1, 3, 3, 3)
    red[:, 0, :, :] = 1
    
    epochs = 1000
    lr = 1e-1
    trials = 5
    learning_curves = [[] for i in range(trials)]

    for trial in range(trials):
        # Initialize network, criterion and optimizer
        torch.manual_seed(22 + trial)
        model = SiameseNetworkLeNet()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            
            optimizer.zero_grad()
            
            # Similar pairs
            output1 = model(red, red)
            output2 = model(green, green)
            loss1 = criterion(output1, torch.tensor([1.0]))
            loss2 = criterion(output2, torch.tensor([1.0]))
            
            # # Dissimilar pairs
            output3 = model(red, green)
            loss3 = criterion(output3, torch.tensor([0.0]))
            output4 = model(green, red)
            loss4 = criterion(output4, torch.tensor([0.0]))
            
            total_loss = (loss1 + loss2 + loss3 + loss4) / 4
            # total_loss = loss3
            
            total_loss.backward()
            
            
            optimizer.step()
            
            learning_curves[trial].append(float(total_loss.item()))
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}\n{output3.item()}")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4f}\n{output1.item()} {output2.item()} {output3.item()} {output4.item()}")

    print("Training completed.")

    PlotResults(learning_curves, plt.plot)
    plt.savefig('test.pdf', dpi=300, bbox_inches='tight')

    model.eval()

    # For similar pairs:
    output_red_red = model(red, red).item()
    output_green_green = model(green, green).item()

    # For dissimilar pairs:
    output_red_green = model(red, green).item()

    print(f"Output for red-red pair: {output_red_red:.4f}")
    print(f"Output for green-green pair: {output_green_green:.4f}")
    print(f"Output for red-green pair: {output_red_green:.4f}")

    # Interpretation
    if output_red_red > 0.5:
        print("Red-Red pair is considered similar.")
    else:
        print("Red-Red pair is considered dissimilar.")

    if output_green_green > 0.5:
        print("Green-Green pair is considered similar.")
    else:
        print("Green-Green pair is considered dissimilar.")

    if output_red_green < 0.5:
        print("Red-Green pair is considered dissimilar.")
    else:
        print("Red-Green pair is considered similar.")
