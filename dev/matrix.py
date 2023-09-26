import numpy as np

def dissimilarity(values):
    sigma_value = 0.1
    sigma_spatial = 4
    
    N,N = values.shape # assumes square
    weights = np.zeros((N * N, N * N))

    values = values.flatten()
    for u in range(N * N):
        for v in range(N * N):
            x_u, y_u = divmod(u, N)
            x_v, y_v = divmod(v, N)
            
            w_value = np.exp(-((values[u] - values[v]) ** 2) / (2 * sigma_value ** 2))
            w_spatial = np.exp(-((x_u - x_v) ** 2 + (y_u - y_v) ** 2) / (2 * sigma_spatial ** 2))

            weights[u][v] = w_value * w_spatial
    return weights

def dissimilarity_torch(values):
    import torch
    
    sigma_value = 0.1
    sigma_spatial = 4.0
    
    N, _ = values.shape
    device = values.device

    # Create meshgrid for spatial coordinates
    x = torch.arange(N, device=device)
    y = torch.arange(N, device=device)
    x_grid, y_grid = torch.meshgrid(x, y, indexing='ij') # indexing to avoid warning

    # Flatten and repeat to prepare for pairwise computations
    x_u = x_grid.reshape(-1, 1).repeat(1, N*N)
    y_u = y_grid.reshape(-1, 1).repeat(1, N*N)
    x_v = x_grid.reshape(1, -1).repeat(N*N, 1)
    y_v = y_grid.reshape(1, -1).repeat(N*N, 1)

    # Compute spatial weights
    w_spatial = torch.exp(-((x_u - x_v)**2 + (y_u - y_v)**2) / (2 * sigma_spatial**2))
    
    # Flatten values and compute value weights
    values_flat = values.reshape(-1, 1).repeat(1, N*N)
    values_comp = values.reshape(1, -1).repeat(N*N, 1)
    w_value = torch.exp(-((values_flat - values_comp)**2) / (2 * sigma_value**2))

    # Compute combined weights
    weights = w_value * w_spatial

    return weights
            
def distance_wrapped(N):
    distances = np.zeros((N*N, N*N), dtype=int)
    for u in range(N * N):
        for v in range(N * N):
            row1, col1 = divmod(u, N)
            row2, col2 = divmod(v, N)
            
            dx = abs(col1 - col2)
            dy = abs(row1 - row2)
            
            # Calculate the Manhattan distance, including diagonals
            manhattan_distance = min(dx, dy) + abs(dx - dy)
            
            distances[u, v] = manhattan_distance
    return distances
            
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

def get_pairs_outside_threshold(distance_matrix, threshold):
    # Get the upper triangle indices
    i, j = np.triu_indices(distance_matrix.shape[0], k=1)
    
    # Set the diagonal to values less than the threshold
    np.fill_diagonal(distance_matrix, threshold - 1)
    
    # Filter the pairs based on the threshold
    mask = distance_matrix[i, j] > threshold
    
    # Get the pairs where the condition is met
    pairs = list(zip(i[mask], j[mask]))

    return pairs