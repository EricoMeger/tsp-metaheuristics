import random
import math
import time

from tsp_utils import TSPUtils
from destroy_operators import DestroyOperators
from bandit_arm import BanditArm


class MAB_LNS:
    def __init__(self,
                 distance_matrix,
                 n_nodes,
                 time_limit,
                 R_repairs,
                 top_k_insert,
                 noise_prob,
                 remove_frac,
                 two_opt_max_iter,
                 two_opt_max_no_improve,
                 two_opt_repair_max_no_improve,
                 segment_num_segments,
                 initial_temperature_factor,
                 cooling_rate,
                 restart_threshold,
                 restart_temperature_factor,
                 reward_threshold):
        self.dist = distance_matrix
        self.n = n_nodes
        self.time_limit = time_limit
        self.R = R_repairs
        self.top_k_insert = top_k_insert
        self.noise_prob = noise_prob
        self.remove_frac = remove_frac
        self.two_opt_max_iter = two_opt_max_iter
        self.two_opt_max_no_improve = two_opt_max_no_improve
        self.two_opt_repair_max_no_improve = two_opt_repair_max_no_improve
        self.segment_num_segments = segment_num_segments
        self.initial_temperature_factor = initial_temperature_factor
        self.cooling_rate = cooling_rate
        self.restart_threshold = restart_threshold
        self.restart_temperature_factor = restart_temperature_factor
        self.reward_threshold = reward_threshold

        self.actions = [
            ("block", self.destroy_block),
            ("random", self.destroy_random),
            ("double_bridge", self.destroy_double_bridge),
            ("worst_edges", self.destroy_worst),
            ("related", self.destroy_related),
            ("segment", self.destroy_segment),
        ]
        
        self.arms = [BanditArm(i) for i in range(len(self.actions))]

    def destroy_block(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.block_removal(tour, k)

    def destroy_random(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.random_removal(tour, k)

    def destroy_double_bridge(self, tour):
        return DestroyOperators.double_bridge(tour)

    def destroy_worst(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        return DestroyOperators.worst_edges_removal(tour, self.dist, k)

    def destroy_related(self, tour):
        k = max(2, int(self.remove_frac * self.n))
        start_node = random.choice(tour)
        return DestroyOperators.related_removal(tour, self.dist, k, start_node)

    def destroy_segment(self, tour):
        return DestroyOperators.segment_shuffle(tour, self.segment_num_segments)

    def run(self, initial_tour):
        start_time = time.time()
        
        if initial_tour is None:
            best_initial = None
            best_initial_cost = float("inf")
            
            for _ in range(5):
                nodes = list(range(self.n))
                tour = TSPUtils.nearest_neighbor_construction(nodes, self.dist)
                tour = TSPUtils.two_opt(tour, self.dist, self.two_opt_max_iter)
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

        temperature = best_cost * self.initial_temperature_factor

        iter_count = 0
        no_improve = 0
        last_print = start_time

        while time.time() - start_time < self.time_limit:
            iter_count += 1

            arm = self.select_thompson()
            action_idx = arm.action_idx

            reward, sol_candidate, cand_cost = self.apply_operator(current, current_cost, action_idx)

            self.update_arm(arm, reward)

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

            temperature *= self.cooling_rate

            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                print(f"  [{elapsed:.0f}s] iter={iter_count}, curr={current_cost:.2f}, best={best_cost:.2f}, T={temperature:.2f}")
                last_print = time.time()

            if no_improve > self.restart_threshold:
                current = best[:]
                current_cost = best_cost
                temperature = best_cost * self.restart_temperature_factor
                no_improve = 0

        print(f"done. iterations: {iter_count}, best: {best_cost:.2f}")
        self.print_arm_stats()
        return best, best_cost

    def select_thompson(self):
        best_sample = -1
        best_arm = None
        for arm in self.arms:
            sample = arm.sample_thompson()
            if sample > best_sample:
                best_sample = sample
                best_arm = arm
        return best_arm

    def apply_operator(self, current_tour, current_cost, action_idx):
        action_name, action_func = self.actions[action_idx]
        
        removed, remaining = action_func(current_tour)
        
        if len(removed) == 0 and len(remaining) == len(current_tour):
            candidate = TSPUtils.two_opt_limited(remaining, self.dist, self.two_opt_max_no_improve)
            cost = TSPUtils.tour_cost(candidate, self.dist)
            improvement = current_cost - cost
            reward = improvement / max(current_cost, 1e-10)
            return reward, candidate, cost

        best_sol = None
        best_cost = float("inf")
        
        for r in range(self.R):
            cand = TSPUtils.cheapest_insertion_with_noise(
                removed, remaining, self.dist,
                self.top_k_insert, self.noise_prob
            )

            if r < 3 or random.random() < 0.3:
                cand = TSPUtils.two_opt_limited(cand, self.dist, self.two_opt_repair_max_no_improve)
            
            cost = TSPUtils.tour_cost(cand, self.dist)
            
            if cost < best_cost:
                best_cost = cost
                best_sol = cand

        improvement = current_cost - best_cost
        reward = improvement / max(current_cost, 1e-10)
        return reward, best_sol, best_cost

    def update_arm(self, arm, reward):
        arm.visits += 1
        if reward > self.reward_threshold:
            arm.successes += 1.0
        else:
            arm.failures += 1.0

    def print_arm_stats(self):
        print("\n=== Arm Statistics ===")
        for arm in self.arms:
            action_name = self.actions[arm.action_idx][0]
            arm.display_info(action_name)
