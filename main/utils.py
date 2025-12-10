import math
import numpy as np

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def build_distance_matrix(coords):
    n = len(coords)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(coords[i], coords[j])
            D[i][j] = d
            D[j][i] = d
    return D

def route_cost(route, dist_matrix):
    n = len(route)
    return sum(dist_matrix[route[i]][route[(i+1) % n]] for i in range(n))
