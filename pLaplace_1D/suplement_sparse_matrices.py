from scipy.sparse._construct import diags
import numpy as np
import scipy.sparse as sp


def sparse_tridiagonal_scipy(n):
    """Create a three-diagonal CSR matrix with ones on its diagonals.

    Args:
    n: The number of rows (or columns) in the matrix.

    Returns:
    The created matrix as a CSR matrix.
    """
    # Create the diagonals
    diagonals = np.ones((3, n))
    # The main diagonal has offset 0, the diagonals above and below it have offsets -1 and 1, respectively
    offsets = [-1, 0, 1]
    # Use the `diags` function to create a sparse matrix with these diagonals
    matrix = diags(diagonals, offsets, shape=(n, n), format='csr')  # type: ignore
    return matrix


def laplace_1D(x):
    # Calculate distances between nodes
    h = np.diff(x)

    # Initialize diagonals
    main_diag = np.hstack((1, 1 / h[:-1] + 1 / h[1:], 1))
    lower_diag = np.hstack((0, -1 / h[1:]))
    upper_diag = np.hstack((-1 / h[:-1], 0))

    # Create the sparse matrix
    laplace_matrix = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csr')  # type: ignore

    return laplace_matrix[1:-1, 1:-1]  # type: ignore


def ff_mass_construct(ff_full, n, h):

    # Length of diagonal elements is n+2
    # Length of sub-diagonal and super-diagonal elements is n+1
    diagonal_elems = np.arange(n + 1)
    h_single = h[0]
    # Rows and columns indices for the diagonals
    row_diag = np.concatenate([diagonal_elems, diagonal_elems[1:], diagonal_elems[:-1]])
    col_diag = np.concatenate([diagonal_elems, diagonal_elems[:-1], diagonal_elems[1:]])

    # Data for the diagonals
    # [[h/3 2*h*ones(1,n)/3 h/3] h*ones(1,2*n+2)/6]
    data_diag = np.concatenate([np.array([h_single / 3]), 2 * h_single / 3 * np.ones(n - 1),
                                np.array([h_single / 3]), h_single / 6 * np.ones(2 * n)])

    Mass = sp.coo_matrix((data_diag, (row_diag, col_diag)), shape=(n + 1, n + 1))  # type: ignore

    ff_mass = ff_full @ Mass

    return ff_mass
