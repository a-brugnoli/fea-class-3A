from src.fem.element_matrices import stiffness_bar_2d_element, mass_bar_2d_element
from scipy.sparse import lil_matrix
import numpy as np


def assemble_stiffness_truss_2d(coordinates, connectivity_table, EA):
    """
    Assemble the global stiffness matrix for a 2D truss structure.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_nodes, 2) containing the coordinates of the nodes.
    connectivity_table : np.ndarray
        Array of shape (n_elements, 2) containing the connectivity information.
    EA : float or np.ndarray of shape (n_elements, 1)
        Axial rigidity of the bar elements. If a single value is provided, it is assumed that all 
        elements have the same axial rigidity. If an array is provided, each element can have a 
        different axial rigidity.

    Returns
    -------
    K : scipy.sparse.csr_matrix
        Global stiffness matrix in CSR format.
    """
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]

    n_dofs = n_nodes * 2
    K = lil_matrix((n_dofs, n_dofs))

    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii]

        if np.isscalar(EA):
            EA_elem = EA
        else:
            assert len(EA) == n_elements
            EA_elem = EA[ii] 

        K_ii = stiffness_bar_2d_element(coordinates[left_node], coordinates[right_node], EA_elem)

        dof_left = 2 * left_node
        dof_right = 2 * right_node
        
        K[dof_left:dof_left+2, dof_left:dof_left+2] += K_ii[:2, :2]
        K[dof_left:dof_left+2, dof_right:dof_right+2] += K_ii[:2, 2:]
        K[dof_right:dof_right+2, dof_left:dof_left+2] += K_ii[2:, :2]
        K[dof_right:dof_right+2, dof_right:dof_right+2] += K_ii[2:, 2:]

    return K.tocsr()


def assemble_mass_truss_2d(coordinates, connectivity_table, rhoA, lumped=False):
    """
    Assemble the global stiffness matrix for a 2D truss structure.

    Parameters
    ----------
    coordinates : np.ndarray
        Array of shape (n_nodes, 2) containing the coordinates of the nodes.
    connectivity_table : np.ndarray
        Array of shape (n_elements, 2) containing the connectivity information.
    rhoA : float or np.ndarray of shape (n_elements, 1)
        Linear density of the bar elements. If a single value is provided, it is assumed that all 
        elements have the density. If an array is provided, each element can have a different density.

    Returns
    -------
    M : scipy.sparse.csr_matrix
        Global stiffness matrix in CSR format.
    """
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]

    n_dofs = n_nodes * 2
    M = lil_matrix((n_dofs, n_dofs))

    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii]

        if np.isscalar(rhoA):
            rhoA_elem = rhoA
        else:
            assert len(rhoA) == n_elements
            rhoA_elem = rhoA[ii] 

        M_ii = mass_bar_2d_element(coordinates[left_node], coordinates[right_node], rhoA_elem, lumped=lumped)

        dof_left = 2 * left_node
        dof_right = 2 * right_node
        
        M[dof_left:dof_left+2, dof_left:dof_left+2]     += M_ii[:2, :2]
        M[dof_left:dof_left+2, dof_right:dof_right+2]   += M_ii[:2, 2:]
        M[dof_right:dof_right+2, dof_left:dof_left+2]   += M_ii[2:, :2]
        M[dof_right:dof_right+2, dof_right:dof_right+2] += M_ii[2:, 2:]

    return M.tocsr()