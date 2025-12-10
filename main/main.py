from parser import parse_instance
from utils import build_distance_matrix, route_cost
from aco import ACO
from two_opt import two_opt
from plot import plot_route

def main():
    path = "300_tsp.txt"
    coords = parse_instance(path)
    dist_matrix = build_distance_matrix(coords)

    # ACO
    aco = ACO(dist_matrix, n_ants=40, alpha=1, beta=3, evaporation=0.2)
    route, cost = aco.run(iterations=150)

    print(f"\nCusto após ACO: {cost:.2f}")

    # 2-opt
    print("\nAplicando 2-opt...")
    final_route = two_opt(route, dist_matrix)
    final_cost = route_cost(final_route, dist_matrix)
    print(f"Custo final: {final_cost:.2f}")

    # plot
    plot_route(coords, final_route, final_cost)

if __name__ == "__main__":
    main()
