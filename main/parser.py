def parse_instance(path):
    coords = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = map(float, line.split())
            coords.append((x, y))
    return coords
