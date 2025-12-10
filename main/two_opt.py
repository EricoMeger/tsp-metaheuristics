def two_opt(route, dist_matrix):
    improved = True
    n = len(route)

    while improved:
        improved = False
        best_i, best_j = None, None
        best_gain = 0

        for i in range(n - 1):
            for j in range(i + 2, n):
                a, b = route[i], route[(i + 1) % n]
                c, d = route[j], route[(j + 1) % n]

                before = dist_matrix[a][b] + dist_matrix[c][d]
                after = dist_matrix[a][c] + dist_matrix[b][d]

                if after < before:
                    gain = before - after
                    if gain > best_gain:
                        best_gain = gain
                        best_i, best_j = i + 1, j
                        improved = True

        if improved:
            route[best_i:best_j + 1] = reversed(route[best_i:best_j + 1])

    return route
