import numpy as np
from skimage import measure
import pyvista as pv
from x_parser import parse_x_model
from shapely.geometry import Polygon


def voxelization(vertices, triangles, grid_size):
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    min_bounds = vertices.min(axis=0)
    max_bounds = vertices.max(axis=0)
    scale = (grid_size - 1) / (max_bounds - min_bounds)

    # Перебираем треугольники и заполняем сетку
    for tri in triangles:
        v0, v1, v2 = vertices[tri]

        v0_scaled = ((v0 - min_bounds) * scale).astype(int)
        v1_scaled = ((v1 - min_bounds) * scale).astype(int)
        v2_scaled = ((v2 - min_bounds) * scale).astype(int)

        x_coords = [v0_scaled[0], v1_scaled[0], v2_scaled[0]]
        y_coords = [v0_scaled[1], v1_scaled[1], v2_scaled[1]]
        z_coords = [v0_scaled[2], v1_scaled[2], v2_scaled[2]]

        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        min_z = min(z_coords)
        max_z = max(z_coords)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    pt = np.array([x, y, z])
                    b1 = Polygon([pt, v0_scaled, v1_scaled]).area < 0.0
                    b2 = Polygon([pt, v0_scaled, v1_scaled]).area < 0.0
                    b3 = Polygon([pt, v0_scaled, v1_scaled]).area < 0.0

                    if (b1 == b2) and (b2 == b3):
                        voxel_grid[x, y, z] = True

    return voxel_grid


def create_mesh_from_voxels(voxel_grid):
    vertices, faces, normals, _ = measure.marching_cubes(voxel_grid, 0)
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)

    return vertices, faces


def save_model(vertices, faces, filepath, scale_factor=1.9):
    scaled_vertices = vertices.copy()
    scaled_vertices[:, 0] *= scale_factor

    lines = []
    for v in scaled_vertices:
        lines.append(f"v {v[0]} {v[1]} {v[2]}\n")
    for face in faces:
        lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    with open(filepath, 'w') as f:
        f.writelines(lines)


def visualize_model(filepath):
    mesh = pv.read(filepath)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='white')

    plotter.show()


if __name__ == '__main__':
    GRID_SIZE = 128

    vertices1, triangles1 = parse_x_model('data/teapot_1.x')
    vertices2, triangles2 = parse_x_model('data/teapot_2.x')

    vertices = np.concatenate((vertices1, vertices2))
    triangles = np.concatenate((triangles1, triangles2 + len(vertices1)))

    voxel_grid = voxelization(vertices, triangles, grid_size=GRID_SIZE)

    vertices, faces = create_mesh_from_voxels(voxel_grid)

    save_model(vertices, faces, 'result_teapot.obj')
    visualize_model('result_teapot.obj')
