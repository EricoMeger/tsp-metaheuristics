import random
import math
import time

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
    def stochastic_greedy_insertion(remainder, partial_tour, dist, top_k=3, noise_prob=0.15):
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
            candidates.sort(key=lambda x: x[0])
            k = min(top_k, len(candidates))
            if random.random() < noise_prob:
                choice = random.randrange(k)
            else:
                choice = 0
            tour.insert(candidates[choice][1], city)
        return tour

    @staticmethod
    def full_greedy_from_scratch(nodes, dist):
        if not nodes:
            return []
        nodes_set = set(nodes)
        start = random.choice(nodes)
        tour = [start]
        nodes_set.remove(start)
        while nodes_set:
            last = tour[-1]
            nxt = min(nodes_set, key=lambda x: dist[last][x])
            tour.append(nxt)
            nodes_set.remove(nxt)
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
    def two_opt(tour, dist, max_iter=1000):
        tour = tour[:]
        for _ in range(max_iter):
            tour, improved = TSPUtils.two_opt_pass(tour, dist)
            if not improved:
                break
        return tour

    @staticmethod
    def two_opt_limited(tour, dist, max_no_improve=50):
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
    def related_removal(tour, dist, k, start_node=None):
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
    def segment_shuffle(tour, num_segments=4):
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

class BanditArm:
    __slots__ = ['action_idx', 'successes', 'failures', 'visits']
    
    def __init__(self, action_idx):
        self.action_idx = action_idx
        self.successes = 1.0
        self.failures = 1.0
        self.visits = 0

    def sample_thompson(self):
        return random.betavariate(self.successes, self.failures)

class MAB_LNS:

    def __init__(self,
                 distance_matrix,
                 n_nodes,
                 time_limit=300,
                 R_repairs=10,
                 top_k_insert=3,
                 noise_prob=0.15,
                 remove_frac=0.20):
        self.dist = distance_matrix
        self.n = n_nodes
        self.time_limit = time_limit
        self.R = R_repairs
        self.top_k_insert = top_k_insert
        self.noise_prob = noise_prob
        self.remove_frac = remove_frac

        self.actions = [
            ("block", self._destroy_block),
            ("random", self._destroy_random),
            ("double_bridge", self._destroy_double_bridge),
            ("worst_edges", self._destroy_worst),
            ("related", self._destroy_related),
            ("segment", self._destroy_segment),
        ]
        
        self.arms = [BanditArm(i) for i in range(len(self.actions))]

    def _destroy_block(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.block_removal(tour, k)

    def _destroy_random(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.random_removal(tour, k)

    def _destroy_double_bridge(self, tour):
        return DestroyOperators.double_bridge(tour)

    def _destroy_worst(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.worst_edges_removal(tour, self.dist, k)

    def _destroy_related(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.related_removal(tour, self.dist, k)

    def _destroy_segment(self, tour):
        return DestroyOperators.segment_shuffle(tour, num_segments=4)

    def run(self, initial_tour=None):
        start_time = time.time()
        
        if initial_tour is None:
            best_initial = None
            best_initial_cost = float("inf")
            
            for _ in range(5):
                nodes = list(range(self.n))
                tour = TSPUtils.full_greedy_from_scratch(nodes, self.dist)
                tour = TSPUtils.two_opt(tour, self.dist, max_iter=2000)
                cost = TSPUtils.tour_cost(tour, self.dist)
                if cost < best_initial_cost:
                    best_initial_cost = cost
                    best_initial = tour
            
            initial_tour = best_initial
            print(f"initial solution: {best_initial_cost:.2f}")
        
        current = initial_tour[:]
        current_cost = TSPUtils.tour_cost(current, self.dist)
        best = current[:]
        best_cost = current_cost

        temperature = best_cost * 0.03
        cooling_rate = 0.9998

        iter_count = 0
        no_improve = 0
        last_print = start_time

        while time.time() - start_time < self.time_limit:
            iter_count += 1

            arm = self._select_thompson()
            action_idx = arm.action_idx

            reward, sol_candidate, cand_cost = self._apply_operator(current, current_cost, action_idx)

            self._update_arm(arm, reward)

            delta = cand_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
                current = sol_candidate
                current_cost = cand_cost
                no_improve = 0
            else:
                no_improve += 1

            if cand_cost < best_cost - 1e-9:
                best = sol_candidate[:]
                best_cost = cand_cost
                print(f"[iter {iter_count}] new best: {best_cost:.2f}")

            temperature *= cooling_rate

            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.0f}s] iter={iter_count}, curr={current_cost:.2f}, best={best_cost:.2f}, T={temperature:.2f}")
                last_print = time.time()

            if no_improve > 200:
                current = best[:]
                current_cost = best_cost
                temperature = best_cost * 0.02
                no_improve = 0

        print(f"MAB-LNS done. iterations: {iter_count}, best: {best_cost:.2f}")
        self._print_arm_stats()
        return best, best_cost

    def _select_thompson(self):
        samples = [(arm.sample_thompson(), arm) for arm in self.arms]
        return max(samples, key=lambda x: x[0])[1]

    def _apply_operator(self, current_tour, current_cost, action_idx):
        _, action_func = self.actions[action_idx]
        
        removed, remaining = action_func(current_tour)
        
        if len(removed) == 0 and len(remaining) == len(current_tour):
            candidate = TSPUtils.two_opt_limited(remaining, self.dist, max_no_improve=30)
            cost = TSPUtils.tour_cost(candidate, self.dist)
            improvement = current_cost - cost
            reward = improvement / max(current_cost, 1e-10)
            return reward, candidate, cost

        best_sol = None
        best_cost = float("inf")
        
        for r in range(self.R):
            cand = TSPUtils.stochastic_greedy_insertion(
                removed, remaining, self.dist,
                top_k=self.top_k_insert, noise_prob=self.noise_prob
            )

            if r < 3 or random.random() < 0.3:
                cand = TSPUtils.two_opt_limited(cand, self.dist, max_no_improve=20)
            
            cost = TSPUtils.tour_cost(cand, self.dist)
            
            if cost < best_cost:
                best_cost = cost
                best_sol = cand

        improvement = current_cost - best_cost
        reward = improvement / max(current_cost, 1e-10)
        return reward, best_sol, best_cost

    def _update_arm(self, arm, reward):
        arm.visits += 1
        
        if reward > 0.0005:
            arm.successes += 1
        else:
            arm.failures += 1

    def _print_arm_stats(self):
        print("\nArm Statistics:")
        for arm in self.arms:
            name = self.actions[arm.action_idx][0]
            thompson_mean = arm.successes / (arm.successes + arm.failures)
            print(f"  {name:15s}: visits={arm.visits:5d}, thompson_mean={thompson_mean:.3f}")

if __name__ == "__main__":
    from parser import Parser
    
    data = Parser.parse("input/300_tsp.txt")
    n = data["n_clients"]
    dist = data["distance_matrix"]

    print(f"Loaded instance with {n} nodes")

    solver = MAB_LNS(
        distance_matrix=dist,
        n_nodes=n,
        time_limit=300,
        R_repairs=20,
        remove_frac=0.20
    )
    
    best_tour, best_cost = solver.run()
    print(f"\nBest cost: {best_cost:.2f}")
    print(f"Tour: {best_tour[:20]}...")