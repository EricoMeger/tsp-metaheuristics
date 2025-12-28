import random


class DestroyOperators:

    @staticmethod
    def block_removal(tour, remove_size):
        n = len(tour)
        if remove_size <= 0 or remove_size >= n:
            return [], tour[:]
        start = random.randrange(0, n)
        removed = []
        remaining = []
        for i in range(n):
            idx = (start + i) % n
            if i < remove_size:
                removed.append(tour[idx])
            else:
                remaining.append(tour[idx])
        return removed, remaining

    @staticmethod
    def random_removal(tour, k):
        n = len(tour)
        k = min(max(1, int(k)), n - 1)
        indices = set(random.sample(range(n), k))
        removed = [tour[i] for i in indices]
        remaining = [tour[i] for i in range(n) if i not in indices]
        return removed, remaining

    @staticmethod
    def double_bridge(tour):
        n = len(tour)
        if n < 8:
            return DestroyOperators.random_removal(tour, max(2, n // 4))
        
        positions = sorted(random.sample(range(1, n), 4))
        a, b, c, d = positions
        
        new_tour = tour[:a] + tour[c:d] + tour[b:c] + tour[a:b] + tour[d:]
        return [], new_tour

    @staticmethod
    def worst_edges_removal(tour, dist, k):
        n = len(tour)
        edges = []
        for i in range(n):
            cost = dist[tour[i]][tour[(i + 1) % n]]
            edges.append((cost, i))
        edges.sort(reverse=True)
        
        remove_indices = set()
        for _, pos in edges:
            if len(remove_indices) >= k:
                break
            remove_indices.add(pos)
        
        removed = [tour[i] for i in remove_indices]
        remaining = [tour[i] for i in range(n) if i not in remove_indices]
        return removed, remaining

    @staticmethod
    def related_removal(tour, dist, k, start_node):
        n = len(tour)
        k = min(k, n - 1)
        
        if start_node is None:
            start_node = random.choice(tour)
        
        removed = [start_node]
        candidates = set(tour) - {start_node}
        
        while len(removed) < k and candidates:
            best_node = None
            best_dist = float("inf")
            for node in candidates:
                min_d = min(dist[node][r] for r in removed)
                if min_d < best_dist:
                    best_dist = min_d
                    best_node = node
            removed.append(best_node)
            candidates.remove(best_node)
        
        remaining = [c for c in tour if c not in set(removed)]
        return removed, remaining

    @staticmethod
    def segment_shuffle(tour, num_segments):
        n = len(tour)
        if n < num_segments * 2:
            return [], tour[:]
        
        seg_size = n // num_segments
        segments = []
        for i in range(num_segments):
            start = i * seg_size
            end = start + seg_size if i < num_segments - 1 else n
            segments.append(tour[start:end])
        
        random.shuffle(segments)
        new_tour = []
        for seg in segments:
            new_tour.extend(seg)
        
        return [], new_tour
