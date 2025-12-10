from parser import Parser
from mab_lns import MAB_LNS
from plotter import Plotter


def main():
    data = Parser.parse("input/300_tsp.txt")
    n = data["n_clients"]
    dist = data["distance_matrix"]

    print(f"Loaded instance with {n} nodes")

    solver = MAB_LNS(
        distance_matrix=dist,
        n_nodes=n,
        time_limit=300,
        #number of repairs after a solution is destroyed (highe: increase exploration, but also runtime)
        R_repairs=20,
        #frac of nods removed during each iteration (higher: increase diversification, but also runtime)
        remove_frac=0.20,
        #at a node reinsert, consider the k best positions and choose one randomly. (higher: more diversity less greedy)
        top_k_insert=3,
        #probability of adding noise during insertion (higher: more randomness, helps escaping local optima)
        noise_prob=0.15,
        
        # -- Two opt params --
        two_opt_max_iter=2000, #max iterations at the initial solution
        two_opt_max_no_improve=30, #max iterations without improvement (any solution)
        two_opt_repair_max_no_improve=20, #max iterations without improvement (repair phase)
        
        #for segment_removal: number of segments to divide the tour into
        segment_num_segments=4,
        
        # -- SA params --
        initial_temperature_factor=0.03, #accepts worse solutions at the start
        cooling_rate=0.9998, #cooling rate 
        restart_threshold=200, #iterations before restart
        restart_temperature_factor=0.02, #temperature factor for restart, tries to reheat to escape local optima (n * best_cost)
        
        # minimum improvement to consider a signficative reward for the destruction operator (MAB)
        reward_threshold=0.0005
    )
    
    best_tour, best_cost = solver.run(initial_tour=None)
    print(f"\nBest cost: {best_cost:.2f}")
    print(f"Tour: {best_tour}...")
    
    Plotter.plot_tour(
        coordinates=data["coordinates"],
        tour=best_tour,
        cost=best_cost,
        title="Melhor rota encontrada"
    )



main()