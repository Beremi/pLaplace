# Marta Matice Valdman 1 - solution
# here is the original file in OneDrive
import scipy.io as io
import numpy as np


# from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import matplotlib.pyplot as plt
from datetime import datetime, date, time
import pandas as pd

import numpy.linalg as la
import scipy.linalg as spla

import time
from multiprocessing import Pool, cpu_count
# from functools import partial
import collections
import os
import math

n_processors = cpu_count()

# read file
file = 'mesh_3D_level5.mat'
coordinates = 'coords'
elements = 'elems'

mesh = mesh_coordinates = io.loadmat(file)
mesh_coordinates = mesh['coords']
mesh_elements = mesh['elems']
n_n = mesh_coordinates.shape[0]        # number of nodes
n_e = mesh_elements.shape[0]           # number of element
dim = mesh_coordinates.shape[1]        # dimension
print(f'no.nodes: {n_n}, no.elements: {n_e}, dim: {dim}')

# print(mesh_coordinates.shape)
# print(mesh_elements.shape)

# coords 3D matice
# single process

start_time = time.time()

coords3D = np.zeros((dim, dim, n_e))
det_coords3D = np.zeros(n_e)
inv_coords3D = np.zeros((dim, dim, n_e))

for el in range(0, n_e):
    nodes = mesh_elements[el]
    coordinates = mesh_coordinates[nodes - 1, :]
    for comp in range(dim):
        coords3D[comp, :, el] = coordinates[comp + 1, :] - coordinates[0, :]

    det_coords3D[el] = np.linalg.det(coords3D[:, :, el])
    inv_coords3D[:, :, el] = np.linalg.inv(coords3D[:, :, el])

print(det_coords3D)
print(sum(det_coords3D) / 6)
print("--single process - %s seconds ---" % (time.time() - start_time))

# multiprocessing
# multi process


def create_3D_set(i):
    coords3D_1 = (mesh_coordinates[mesh_elements[:, i]] - mesh_coordinates[mesh_elements[:, 0]]).T
    # print(i,coords3D_1.shape)
    return coords3D_1


start = time.time()
pool = Pool(processes=8)

# coords3D = np.array(pool.map(create_3D_set,coords3D))
coords3D = np.array(pool.map(create_3D_set, [i for i in range(1, dim + 1)]))

det_coords3D = np.linalg.det(coords3D.T)
inv_coords3D = np.linalg.inv(coords3D.T)

end = time.time()

print("--mutliprocess - %s seconds ---" % (time.time() - start_time))
