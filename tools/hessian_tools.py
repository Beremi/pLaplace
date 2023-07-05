from scipy.sparse.csgraph import reverse_cuthill_mckee  # type: ignore
import numpy as np
import scipy.sparse as sp  # type: ignore


def color(J, p=None):
    m, n = J.shape
    if p is None:
        p = reverse_cuthill_mckee(J)

    J = J[:, p]
    group = np.zeros(n, dtype=int)
    ncol = 0

    while np.any(group == 0):
        ncol += 1
        rows = np.zeros(m)
        index = np.where(group == 0)[0]
        lenindex = len(index)
        for i in range(lenindex):
            k = index[i]
            inner = np.inner(J[:, k].todense().ravel(), rows)
            if inner == 0:
                group[k] = ncol
                rows += J[:, k].toarray().flatten()

    group[p] = group.copy()
    return group


def sfd(x, grad, Hstr, group, dx):
    xcurr = x.flatten()
    m, n = Hstr.shape
    v = np.zeros(n)
    ncol = np.max(group)

    alpha = 1e-7 * np.ones(ncol)

    # H = lil_matrix(Hstr != 0, dtype=float)  # Equivalent of MATLAB's spones
    rowH, colH = Hstr.nonzero()
    all_new_rows = []
    all_new_cols = []
    all_new_data = []
    for k in range(ncol):
        d = (group == k + 1)

        xnrm = max(np.linalg.norm(xcurr[d]), 1)  # type: ignore
        alpha[k] *= xnrm
        y = xcurr + alpha[k] * d

        v = dx(y)
        w = (v - grad) / alpha[k]
        mask_H_cols = d[colH]

        new_rows = rowH[mask_H_cols]
        new_cols = colH[mask_H_cols]
        new_data = w[new_rows]
        all_new_rows.append(new_rows)
        all_new_cols.append(new_cols)
        all_new_data.append(new_data)

    new_rows_singe_array = np.concatenate(all_new_rows + all_new_cols)
    new_cols_singe_array = np.concatenate(all_new_cols + all_new_rows)
    new_data_singe_array = np.concatenate(all_new_data + all_new_data) / 2
    H = sp.csc_matrix((new_data_singe_array, (new_rows_singe_array, new_cols_singe_array)),  # type: ignore
                      shape=(m, n))
    return H
