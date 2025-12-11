def parse_instance(path):
    coords = []

    with open(path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        # Ignorar linhas vazias
        if not line:
            continue

        # Ignorar linhas com texto antes das coordenadas
        # Ex: "Seed utilizada:", "Coordenadas:", "0: 2.0000, 27.0000"
        # No arquivo cada linha de coordenada está no formato: 0: X, Y
        if ":" in line:
            try:
                # Ex: "0: 2.0000, 27.0000"
                idx, rest = line.split(":")
                x_str, y_str = rest.split(",")
                x = float(x_str)
                y = float(y_str)
                coords.append((x, y))
            except:
                # Linha tem ":" mas não está no formato esperado → ignorar
                continue

        else:
            # Tentar interpretar como "X Y"
            parts = line.split()
            if len(parts) == 2:
                try:
                    x, y = map(float, parts)
                    coords.append((x, y))
                except:
                    pass  # Ignorar linhas inválidas

    return coords
