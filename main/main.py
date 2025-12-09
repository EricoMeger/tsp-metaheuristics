from parser import Parser

data = Parser.parse("input/200_tsp.txt")

print(f"clients: {data['n_clients']}")
print(f"coords: {data['coordinates']}")
print(f"matrix: {data['distance_matrix']}")