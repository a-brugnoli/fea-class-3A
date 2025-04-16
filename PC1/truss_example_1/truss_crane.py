'''
author: Andrea Brugnoli

Python solution to the exercise proposed in
https://people.duke.edu/~hpgavin/cee421/truss-method.pdf

'''
from PC1.truss_example_1.data import EA, connectivity_table, coordinates, forces, nodes_bcs
import numpy as np
from src.fem.assemble_stiffness import assemble_stiffness_truss_2d
from src.postprocessing.plot_mesh import plot_truss_structure
from src.fem.solve_system import solve_system_homogeneous_bcs


plot_truss_structure(coordinates, connectivity_table)
K, _, _ = assemble_stiffness_truss_2d(coordinates, connectivity_table, EA)

f = np.zeros(K.shape[0])

for node, force in forces.items():
    dof = 2*node
    f[dof:dof+2] = force

dofs_bcs = [2*nodes_bcs[i] for i in range(len(nodes_bcs))] \
         + [2*nodes_bcs[i]+1 for i in range(len(nodes_bcs))]
print(type(dofs_bcs))

q_all, _ = solve_system_homogeneous_bcs(K, f, dofs_bcs)

print(f"Displacement solution in [inches]: \n {q_all} \n")