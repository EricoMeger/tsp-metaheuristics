import random


class TSPUtils:

    @staticmethod
    def tour_cost(tour, dist):
        n = len(tour)
        if n == 0:
            return 0.0
        cost = 0.0
        for i in range(n):
            cost += dist[tour[i]][tour[(i + 1) % n]]
        return cost

    @staticmethod
    def cheapest_insertion_with_noise(remainder, partial_tour, dist, top_k, noise_prob):
        tour = partial_tour[:]
        cities = remainder[:]
        random.shuffle(cities)
        for city in cities:
            if len(tour) == 0:
                tour.append(city)
                continue
            candidates = []
            for pos in range(len(tour)):
                a = tour[pos]
                b = tour[(pos + 1) % len(tour)]
                delta = dist[a][city] + dist[city][b] - dist[a][b]
                candidates.append((delta, pos + 1))
            
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    if candidates[j][0] < candidates[i][0]:
                        candidates[i], candidates[j] = candidates[j], candidates[i]
            
            k = min(top_k, len(candidates))
            if random.random() < noise_prob:
                choice = random.randrange(k)
            else:
                choice = 0
            tour.insert(candidates[choice][1], city)
        return tour

    @staticmethod
    def nearest_neighbor_construction(nodes, dist):
        if not nodes:
            return []
        nodes_set = set(nodes)
        start = random.choice(nodes)
        tour = [start]
        nodes_set.remove(start)
        while nodes_set:
            last = tour[-1]
            
            nearest_node = None
            nearest_distance = float("inf")
            for node in nodes_set:
                distance = dist[last][node]
                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_node = node
            
            tour.append(nearest_node)
            nodes_set.remove(nearest_node)
        return tour

    @staticmethod
    def two_opt_pass(tour, dist):
        n = len(tour)
        if n < 4:
            return tour, False
        
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                a, b = tour[i], tour[i + 1]
                c, d = tour[j], tour[(j + 1) % n]
                
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                
                if delta < -1e-9:
                    tour[i+1:j+1] = reversed(tour[i+1:j+1])
                    return tour, True
        return tour, False

    @staticmethod
    def two_opt(tour, dist, max_iter):
        tour = tour[:]
        for _ in range(max_iter):
            tour, improved = TSPUtils.two_opt_pass(tour, dist)
            if not improved:
                break
        return tour

    @staticmethod
    def two_opt_limited(tour, dist, max_no_improve):
        n = len(tour)
        if n < 4:
            return tour
        
        tour = tour[:]
        no_improve = 0
        
        while no_improve < max_no_improve:
            improved = False
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    
                    delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]
                    
                    if delta < -1e-9:
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        improved = True
                        no_improve = 0
                        break
                if improved:
                    break
            if not improved:
                no_improve += 1
        return tour
