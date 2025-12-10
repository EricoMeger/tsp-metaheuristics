import matplotlib.pyplot as plt

class Plotter:
    @staticmethod
    def plot_tour(coordinates, tour, cost=None, title="TSP Solution"):
        """
        coordinates : list of [x, y]
        tour        : list of node indices (order of visit)
        cost        : optional, total route cost
        """

        # Garantir que o tour é fechado
        if tour[0] != tour[-1]:
            tour = tour + [tour[0]]

        # Extrair coordenadas da rota
        xs = [coordinates[i][0] for i in tour]
        ys = [coordinates[i][1] for i in tour]

        plt.figure(figsize=(10, 10))

        # Plotar rota
        plt.plot(xs, ys, 'b-', linewidth=1.2, zorder=1)

        # Plotar nós
        for i, (x, y) in enumerate(coordinates):
            plt.scatter(x, y, c='red', s=25, zorder=2)
            plt.text(x + 0.4, y + 0.4, str(i), fontsize=8)

        # Título
        if cost is not None:
            plt.title(f"{title} : {round(cost, 2)}")
        else:
            plt.title(title)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()
