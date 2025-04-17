from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import numpy as np


def solve_system_homogeneous_bcs(K, f, dofs_bcs=[]):
    n_dofs = K.shape[0]
    dofs = np.arange(n_dofs)

    dofs_no_bcs = list(set(dofs) - set(dofs_bcs))
    K_no_bcs = K[dofs_no_bcs, :][:, dofs_no_bcs]
    K_red = csr_matrix(K_no_bcs)

    f_red = f[dofs_no_bcs]
    q_red = spsolve(K_red, f_red)

    q_all = np.zeros(n_dofs)
    q_all[dofs_no_bcs] = q_red

    reactions = K[dofs_bcs, :][:, dofs_no_bcs] @ q_red

    return q_all, reactions
