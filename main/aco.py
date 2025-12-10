import numpy as np
import random
from utils import route_cost

class ACO:
    def __init__(self, dist_matrix, n_ants=30, alpha=1, beta=3, evaporation=0.2):
        self.dist = dist_matrix
        self.n = len(dist_matrix)
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = evaporation

        # feromônios
        self.pher = np.ones((self.n, self.n))

    def _probabilities(self, current, visited):
        probs = np.zeros(self.n)
        remaining = [i for i in range(self.n) if i not in visited]

        denom = sum(
            (self.pher[current][j] ** self.alpha) *
            ((1 / self.dist[current][j]) ** self.beta)
            for j in remaining
        )

        for j in remaining:
            probs[j] = (
                (self.pher[current][j] ** self.alpha) *
                ((1 / self.dist[current][j]) ** self.beta)
            ) / denom

        return probs

    def _build_route(self):
        route = []
        start = random.randrange(self.n)
        route.append(start)

        while len(route) < self.n:
            cur = route[-1]
            probs = self._probabilities(cur, set(route))
            nxt = random.choices(range(self.n), weights=probs)[0]
            route.append(nxt)

        return route

    def run(self, iterations=100):
        best_route = None
        best_cost = float("inf")

        for it in range(iterations):
            routes = []
            costs = []

            for _ in range(self.n_ants):
                r = self._build_route()
                c = route_cost(r, self.dist)
                routes.append(r)
                costs.append(c)

            idx = np.argmin(costs)
            if costs[idx] < best_cost:
                best_route = routes[idx].copy()
                best_cost = costs[idx]
                print(f"[Iter {it}] Melhor: {best_cost:.2f}")

            # evaporação
            self.pher *= (1 - self.rho)

            # deposição
            for r, c in zip(routes, costs):
                delta = 1 / c
                for i in range(self.n):
                    a, b = r[i], r[(i + 1) % self.n]
                    self.pher[a][b] += delta
                    self.pher[b][a] += delta

        return best_route, best_cost
