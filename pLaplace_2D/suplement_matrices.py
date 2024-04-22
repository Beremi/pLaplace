import numpy as np
from scipy.sparse import csr_matrix
from .mesh import Mesh2D


def mass_matrix(mesh: Mesh2D):
    areas = mesh.areas
    local_mass_matrix = 1 / 12 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])

    row_idx = []
    col_idx = []
    values = []
    for i in range(3):
        for j in range(3):
            row_idx.append(mesh.elements[:, i])
            col_idx.append(mesh.elements[:, j])
            values.append(local_mass_matrix[i, j] * areas)

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    values = np.concatenate(values)

    res = csr_matrix((values, (row_idx, col_idx)), shape=(mesh.nodes.shape[0], mesh.nodes.shape[0]))
    res.sum_duplicates()
    return res


def stiffness_matrix(mesh: Mesh2D):
    areas = mesh.areas

    row_idx = []
    col_idx = []
    values = []
    for i in range(3):
        for j in range(3):
            row_idx.append(mesh.elements[:, i])
            col_idx.append(mesh.elements[:, j])
            values.append(
                (mesh.gradx[:, i] * mesh.gradx[:, j] + mesh.grady[:, i] * mesh.grady[:, j]) * areas)

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    values = np.concatenate(values)

    res = csr_matrix((values, (row_idx, col_idx)), shape=(mesh.nodes.shape[0], mesh.nodes.shape[0]))
    res.sum_duplicates()
    return res


def incidence_matrix(mesh: Mesh2D):
    row_idx = []
    col_idx = []
    values = []
    for i in range(3):
        for j in range(3):
            row_idx.append(mesh.elements[:, i])
            col_idx.append(mesh.elements[:, j])
            values.append(np.ones_like(mesh.elements[:, i], dtype=np.bool_))

    row_idx = np.concatenate(row_idx)
    col_idx = np.concatenate(col_idx)
    values = np.concatenate(values)

    res = csr_matrix((values, (row_idx, col_idx)), shape=(mesh.nodes.shape[0], mesh.nodes.shape[0]))
    res.sum_duplicates()
    return res
