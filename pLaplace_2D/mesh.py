import numpy as np
import scipy.io as sio
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from typing import Literal


class Mesh2D:
    def __init__(self) -> None:
        self.nodes = np.array([])
        self.elements = np.array([])
        self.free_nodes = np.array([])
        self.gradx = np.array([])
        self.grady = np.array([])

    def load_from_matfile(self, filename: str) -> None:
        data = sio.loadmat(filename)
        keys = list(data.keys())
        keys = [key for key in keys if not key.startswith('__')]
        if keys == ['mesh']:
            # handle 'mesh' case
            self.nodes = data['mesh'][0][0][1]
            self.elements = data['mesh'][0][0][3] - 1
            self.free_nodes = data['mesh'][0][0][5].ravel() - 1
            self.gradx = data['mesh'][0][0][7]
            self.grady = data['mesh'][0][0][8]
        elif keys == ['coordinates', 'elements']:
            # handle 'coordinates', 'elements' case
            self.nodes = data["coordinates"]
            self.elements = data["elements"] - 1
            self.compute_gradients()
            self.identify_inner_nodes()
        elif keys == ['coords', 'elems']:
            # handle 'coords', 'elems' case
            print('Handling coords and elems')
            raise NotImplementedError("This functionality is not implemented yet")
        else:
            raise ValueError(f"Unknown keys: {keys}")
        self.get_areas()

    def get_areas(self):
        X = self.nodes[self.elements]
        edge0 = X[:, 1] - X[:, 0]
        edge1 = X[:, 2] - X[:, 0]
        self.areas = 0.5 * np.abs(edge0[:, 0] * edge1[:, 1] - edge0[:, 1] * edge1[:, 0])

    def compute_gradients(self) -> None:
        # Define the variables
        a, b, c = sp.symbols('a b c')
        Ax, Ay, Bx, By, Cx, Cy = sp.symbols('Ax Ay Bx By Cx Cy')
        # Define the equations based on the given conditions
        eq1 = sp.Eq(a * Ax + b * Ay + c, 1)
        eq2 = sp.Eq(a * Bx + b * By + c, 0)
        eq3 = sp.Eq(a * Cx + b * Cy + c, 0)
        # Solve the system of equations for a, b, and c
        solution = sp.solve((eq1, eq2, eq3), (a, b, c))
        lin_coeff = sp.lambdify((Ax, Ay, Bx, By, Cx, Cy), list(solution.values()), 'numpy')
        # initialize the gradient arrays
        gradx = np.zeros((self.elements.shape[0], 3))
        grady = np.zeros((self.elements.shape[0], 3))
        # compute the gradients for first vertex
        a, b, _ = lin_coeff(self.nodes[self.elements[:, 0], 0], self.nodes[self.elements[:, 0], 1],
                            self.nodes[self.elements[:, 1], 0], self.nodes[self.elements[:, 1], 1],
                            self.nodes[self.elements[:, 2], 0], self.nodes[self.elements[:, 2], 1])
        gradx[:, 0] = a
        grady[:, 0] = b
        # compute the gradients for second vertex
        a, b, _ = lin_coeff(self.nodes[self.elements[:, 1], 0], self.nodes[self.elements[:, 1], 1],
                            self.nodes[self.elements[:, 2], 0], self.nodes[self.elements[:, 2], 1],
                            self.nodes[self.elements[:, 0], 0], self.nodes[self.elements[:, 0], 1])
        gradx[:, 1] = a
        grady[:, 1] = b
        # compute the gradients for third vertex
        a, b, _ = lin_coeff(self.nodes[self.elements[:, 2], 0], self.nodes[self.elements[:, 2], 1],
                            self.nodes[self.elements[:, 0], 0], self.nodes[self.elements[:, 0], 1],
                            self.nodes[self.elements[:, 1], 0], self.nodes[self.elements[:, 1], 1])
        gradx[:, 2] = a
        grady[:, 2] = b
        # store the gradients
        self.gradx = gradx
        self.grady = grady

    def plot_triangulation(self):
        mesh_coordinates = self.nodes
        mesh_elements = self.elements
        plt.figure(figsize=(10, 10))
        plt.triplot(mesh_coordinates[:, 0], mesh_coordinates[:, 1], mesh_elements)
        # plot nodes numbers
        for i, node in enumerate(mesh_coordinates):
            plt.text(node[0], node[1], str(i))
        # compute baricenters
        barycentres = np.zeros((mesh_elements.shape[0], 2))
        barycentres[:, 0] = np.sum(mesh_coordinates[mesh_elements, 0], axis=1) / 3
        barycentres[:, 1] = np.sum(mesh_coordinates[mesh_elements, 1], axis=1) / 3
        # plot baricenters numbers
        for i, barycentre in enumerate(barycentres):
            plt.text(barycentre[0], barycentre[1], str(i), color='red')
        plt.show()

    def identify_inner_nodes(self):
        # Create a sparse matrix of edges
        edges_orig = np.vstack((self.elements[:, [0, 1]], self.elements[:, [1, 2]], self.elements[:, [2, 0]]))
        # Sort each row
        sorted_matrix = np.sort(edges_orig, axis=1)
        # Find unique rows
        _, indexes, counts = np.unique(sorted_matrix, axis=0, return_index=True, return_counts=True)
        boundary_nodes = np.unique(edges_orig[indexes[counts == 1], :].ravel())
        self.free_nodes = np.setdiff1d(np.arange(self.nodes.shape[0]), boundary_nodes)

    def plot_solution(self, u_res, type: Literal["contour", "triplot"] = 'contour', levels=10, plot_grid=False):
        # Create the triangulation
        triangulation = tri.Triangulation(self.nodes[:, 0], self.nodes[:, 1], self.elements)
        plt.figure(figsize=(8, 6))
        plt.gca().set_aspect('equal')
        # set the plot type
        if type == 'contour':
            # Create a pseudocolor plot
            epsilon = 1e-16
            contour_levels = np.linspace(u_res.min() - epsilon, u_res.max() + epsilon, levels)
            plot_handle = plt.tricontourf(triangulation, u_res, cmap='jet', levels=contour_levels)
            if plot_grid:
                plt.triplot(triangulation, 'ko-', linewidth=0.5, alpha=0.5)
            plt.title('Contour plot of the solution')
        elif type == 'triplot':
            plot_handle = plt.tripcolor(triangulation, u_res, shading='gouraud', cmap='jet')
            if plot_grid:
                plt.triplot(triangulation, 'ko-', linewidth=0.5, alpha=0.5)
            plt.title('Triplot plot of the solution')
        else:
            raise ValueError(f"Unknown type: {type}")
        # Add a colorbar
        plt.colorbar(plot_handle)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def print_statistics(self) -> None:
        n_nodes = self.nodes.shape[0]
        n_elements = self.elements.shape[0]
        s_free_nodes = self.free_nodes.shape[0]
        print(f"Number of nodes: {n_nodes}")
        print(f"Number of elements: {n_elements}")
        print(f"Number of free nodes: {s_free_nodes}")
