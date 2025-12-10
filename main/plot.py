import matplotlib.pyplot as plt

def plot_route(coords, route, cost):
    xs = [coords[i][0] for i in route] + [coords[route[0]][0]]
    ys = [coords[i][1] for i in route] + [coords[route[0]][1]]

    plt.figure(figsize=(10, 8))
    plt.plot(xs, ys, '-o', markersize=4)
    plt.title(f"Melhor rota encontrada: {cost:.2f}")

    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=6)

    plt.grid(True)
    plt.show()
