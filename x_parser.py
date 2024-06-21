import numpy as np


def parse_x_model(filepath: str) -> (np.array, np.array):
    vertices = []
    triangles = []
    transform_matrix = parse_transform_matrix(filepath)

    with open(filepath, 'r') as file:
        lines = file.readlines()
        section = False

        for line in lines:
            line = line.strip()

            if line.startswith('Mesh '):
                section = True
                section_name = 'Mesh'
                continue

            elif line.startswith('Triangles '):
                section = True
                section_name = 'Triangles'
                continue

            if section:
                if line.startswith(';;'):
                    section = False  # конец блока
                if line.endswith(';,'):
                    if section_name == 'Mesh':
                        parts = line[:-2].split(';')
                        if len(parts) == 3:
                            x, y, z = map(float, parts)
                            vertex = np.array([x, y, z, 1.0])
                            vertices.append(vertex)
                    elif section_name == 'Triangles':
                        parts = line[2:-2].split(',')
                        if len(parts) == 3:
                            x, y, z = map(int, parts)
                            triangles.append((x, y, z))

    transformed_vertices = np.array(list(map(lambda vertex: np.dot(transform_matrix, vertex)[:3], vertices)))

    return np.array(transformed_vertices), np.array(triangles)


def parse_transform_matrix(filepath: str) -> np.array:
    with open(filepath, 'r') as file:
        lines = file.readlines()
        section = False
        for line in lines:
            line = line.strip()
            if line.startswith('FrameTransformMatrix '):
                section = True
                continue
            if section:
                if line.startswith('}'):
                    section = False  # конец блока
            if section:
                parts = line.split(';')[0].split(',')
                numbers = [float(part) for part in parts]
                transform_matrix = np.array(numbers).reshape((4, 4))
    transform_matrix = transform_matrix.T
    return transform_matrix
