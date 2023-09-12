import numpy as np
import matplotlib.pyplot as plt


def dissimilarity(values):
    X,Y = values.shape
    distances = np.zeros((X*Y, X*Y), dtype=int)
    for i in range(X*Y):
        for j in range(X*Y):
            distances[i, j] = abs(values[i] - values[j])
            
def distance_wrapped(N):
    distances = np.zeros((N*N, N*N), dtype=int)
    for u in range(N * N):
        for v in range(N * N):
            # coordinates from x//y, x%y
            x1, y1 = divmod(u, N) 
            x2, y2 = divmod(v, N)
            # abs difference in both directions
            dx = abs(y1 - y2)
            dy = abs(x1 - x2)
            # handle wrap around by considering minimum in both directions
            wrap_dx = min(dx, N - dx)
            wrap_dy = min(dy, N - dy)
            # manhattan distance as sum of wrapped distances
            distances[u, v] = wrap_dx + wrap_dy
    return distances
            
def distance_manhattan(N):
    distances = np.zeros((N * N, N * N), dtype=int)

    for u in range(N * N):
        for v in range(N * N):
            coord1 = np.array([u // N, u % N])
            coord2 = np.array([v // N, v % N])
            distances[u, v] = np.sum(np.abs(coord1 - coord2))
    return distances

man = distance_manhattan(5)
wrap = distance_wrapped(5)
filter_man = np.where(man > 2, 0, man)
filter_wrap = np.where(wrap > 2, 0, wrap)
print(filter_man)
print()
print(filter_wrap)



plt.imshow(filter_man, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Filtered Array")
plt.show()

plt.imshow(filter_wrap, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.title("Filtered Array")
plt.show()