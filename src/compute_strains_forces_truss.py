import numpy as np

def compute_strains_forces_truss_2d(coordinates, connectivity_table, displacement, EA):
    '''
    Function that computes the axial strains and forces of a 2D truss structure.
    Parameters
    ----------
    coordinates : np.ndarray of shape (n_nodes, 2)
        Coordinates of the nodes in the truss structure.
    connectivity_table : np.ndarray of shape (n_elements, 2)
        Connectivity table of the truss structure.
    displacement : np.ndarray of shape (n_dofs,)
        Displacement vector of the truss structure. For each dof this vector contains the global 
        horizontal and vertical displacements of the nodes.
    EA : float or np.ndarray of shape (n_elements, 1)
        Axial rigidity of the truss elements. If a single value is provided, it is assumed that all 
        elements have the same axial rigidity. If an array is provided, each element can have a 
        different axial rigidity.
    Returns
    -------
    axial_strains : np.ndarray of shape (n_elements,)
        Axial strains of the truss elements.
    axial_forces : np.ndarray of shape (n_elements,)    
    '''
    n_elements = connectivity_table.shape[0]


    axial_strains = np.zeros(n_elements)
    axial_forces = np.zeros(n_elements)

    for ii in range(n_elements):

        left_node = connectivity_table[ii, 0]
        right_node = connectivity_table[ii, 1]

        dof_left = 2*left_node
        dof_right = 2*right_node
        u_gl_left = displacement[dof_left: dof_left + 2]
        u_gl_right = displacement[dof_right: dof_right + 2]

        coord1 = coordinates[left_node]
        coord2 = coordinates[right_node]

        L = np.linalg.norm(coord2 - coord1) # length of the bar

        x1, y1 = coord1
        x2, y2 = coord2
        
        c = ( x2 - x1 ) / L # cosine of bar angle
        s = ( y2 - y1 ) / L # sine of bar angle

        theta = np.arctan2(s, c)

        u_loc_left = np.cos(theta) * u_gl_left[0] + np.sin(theta) * u_gl_left[1]
        u_loc_right = np.cos(theta) * u_gl_right[0] + np.sin(theta) * u_gl_right[1]

        axial_strains[ii] = (u_loc_right - u_loc_left) / L

        if np.isscalar(EA):
            axial_forces[ii] = EA * axial_strains[ii]
        else:
            assert EA.shape[0] == n_elements and EA.shape[1] == 1
            axial_forces[ii] = EA[ii] * axial_strains[ii]

        # print(f"Local axial force element {ii+1}: \n {axial_forces[ii]}")

    return axial_strains, axial_forces