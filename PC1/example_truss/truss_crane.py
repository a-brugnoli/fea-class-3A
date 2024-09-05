'''
author: Andrea Brugnoli

Python solution to the exercise proposed in
https://people.duke.edu/~hpgavin/cee421/truss-method.pdf
'''

from PC1.example_truss.data import EA, connectivity_table, coordinates, forces, nodes_bcs
import numpy as np
from src.element_stiffness import truss_2d_element
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from src.plot_mesh import plot_truss_structure



plot_truss_structure(coordinates, connectivity_table)

n_elements = connectivity_table.shape[0]
n_nodes = coordinates.shape[0]
n_dofs = 2 * n_nodes
K = lil_matrix((n_dofs, n_dofs))

theta_vec = np.zeros((n_elements, ))

for ii in range(n_elements):
    
    left_node, right_node = connectivity_table[ii]

    K_element, theta_element = truss_2d_element(coordinates[left_node], 
                                coordinates[right_node], 
                                EA)
    
    theta_vec[ii] = theta_element
        
    dof_left = 2*left_node
    dof_right = 2*right_node

    K[dof_left:dof_left+2, dof_left:dof_left+2] += K_element[0:2,0:2]
    K[dof_right:dof_right+2, dof_right:dof_right+2] += K_element[2:4,2:4]

    K[dof_left:dof_left+2, dof_right:dof_right+2] += K_element[0:2,2:4]
    K[dof_right:dof_right+2, dof_left:dof_left+2] += K_element[2:4,0:2]


dofs_bcs = [2*nodes_bcs[i] for i in range(len(nodes_bcs))] \
         + [2*nodes_bcs[i]+1 for i in range(len(nodes_bcs))]

dofs_bcs.sort()
dofs = np.arange(n_dofs)

dofs_no_bcs = list(set(dofs) - set(dofs_bcs))
K_no_bcs = K[dofs_no_bcs, :][:, dofs_no_bcs]
K_red = csr_matrix(K_no_bcs)

f = np.zeros((n_dofs, ))

for node, force in forces.items():
    dof = 2*node
    print(force)
    f[dof:dof+2] = force

f_red = f[dofs_no_bcs]
print(f_red)
q_red = spsolve(K_red, f_red)

print(f"Displacement solution: \n {q_red} \n")