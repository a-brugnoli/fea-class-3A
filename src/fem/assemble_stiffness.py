from src.fem.element_matrices import stiffness_truss_2d_element
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
        Axial rigidity of the truss elements. If a single value is provided, it is assumed that all 
        elements have the same axial rigidity. If an array is provided, each element can have a 
        different axial rigidity.

    Returns
    -------
    K : scipy.sparse.lil_matrix
        Global stiffness matrix in LIL format.
    elements_angle : np.ndarray
        Array of angles for each element.
    elements_length : np.ndarray
        Array of lengths for each element.
    """
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]

    n_dofs = n_nodes * 2
    K = lil_matrix((n_dofs, n_dofs))
    elements_angle = np.zeros(n_elements)
    elements_length = np.zeros(n_elements)

    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii]

        if np.isscalar(EA):
            EA_elem = EA
        else:
            assert EA.shape[0] == n_elements and EA.shape[1] == 1
            EA_elem = EA[ii] 

        K_ii, angle_ii, length_ii = stiffness_truss_2d_element(coordinates[left_node], coordinates[right_node], EA_elem)
        elements_angle[ii] = angle_ii
        elements_length[ii] = length_ii
        dof_left = 2 * left_node
        dof_right = 2 * right_node
        
        K[dof_left:dof_left+2, dof_left:dof_left+2] += K_ii[:2, :2]
        K[dof_left:dof_left+2, dof_right:dof_right+2] += K_ii[:2, 2:]
        K[dof_right:dof_right+2, dof_left:dof_left+2] += K_ii[2:, :2]
        K[dof_right:dof_right+2, dof_right:dof_right+2] += K_ii[2:, 2:]


    K = K.tocsr()

    return K, elements_angle, elements_length