import matplotlib.pyplot as plt
import os

class Plotter:
    @staticmethod
    def plot_tour(coordinates, tour, cost=None, title="TSP Solution"):
        """
        coordinates : list of [x, y]
        tour        : list of node indices (order of visit)
        cost        : optional, total route cost
        """

        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]

        xs = [coordinates[i][0] for i in tour]
        ys = [coordinates[i][1] for i in tour]

        plt.figure(figsize=(10, 10))

        plt.plot(xs, ys, 'b-', linewidth=1.2, zorder=1)

        for i, (x, y) in enumerate(coordinates):
            plt.scatter(x, y, c='red', s=25, zorder=2)
            plt.text(x + 0.4, y + 0.4, str(i), fontsize=8)

        if cost is not None:
            plt.title(f"{title} : {round(cost, 2)}")
        else:
            plt.title(title)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')

        plt.tight_layout()

        results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, f"solution_cost_{cost:.2f}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"image saved in: {filepath}")
        
        plt.show()
