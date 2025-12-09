class Parser:
    @staticmethod
    def parse(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        lines = [line.strip() for line in lines]
        
        coord_idx = None
        for i, line in enumerate(lines):
            if 'Coordenadas:' in line:
                coord_idx = i + 1
                break
        
        coordinates = []
        i = coord_idx
        while i < len(lines):
            if 'Matriz de Distâncias:' in lines[i]:
                break

            if lines[i] and ':' in lines[i]:
                parts = lines[i].split(':')
                coords = parts[1].split(',')
                x = float(coords[0].strip())
                y = float(coords[1].strip())
                coordinates.append([x, y])

            i += 1
        
        n_clients = len(coordinates)
        
        matrix_idx = None
        for j in range(i, len(lines)):
            if 'Matriz de Distâncias:' in lines[j]:
                matrix_idx = j + 1
                break
        
        distance_matrix = []
        for k in range(matrix_idx, min(matrix_idx + n_clients, len(lines))):
            if lines[k]:
                distances = list(map(float, lines[k].split()))
                distance_matrix.append(distances)
        
        return {
            'coordinates': coordinates,
            'distance_matrix': distance_matrix,
            'n_clients': n_clients
        }


