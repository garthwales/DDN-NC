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