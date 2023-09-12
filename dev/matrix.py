import numpy as np

def dissimilarity(values):
    X,Y = values.shape
    distances = np.zeros((X*Y, X*Y), dtype=int)
    for i in range(X*Y):
        for j in range(X*Y):
            distances[i, j] = abs(values[i] - values[j])
    return distances
            
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
            coord1 = np.array([u // N, u % N])
            coord2 = np.array([v // N, v % N])
            distances[u, v] = np.sum(np.abs(coord1 - coord2))
    return distances