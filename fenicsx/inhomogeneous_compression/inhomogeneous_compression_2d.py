import dolfinx
from dolfinx import log, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver
import pyvista
import numpy as np
import ufl

from mpi4py import MPI
from dolfinx import fem, mesh, plot
L = 10.0
n_elements = 30

domain = mesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [L, L]], [n_elements, n_elements], \
                               cell_type= mesh.CellType.quadrilateral)

pol_degree = 2
V = fem.functionspace(domain, ("Lagrange", pol_degree, (domain.geometry.dim, )))

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    return np.isclose(x[0], L)

def bottom(x):
    return np.isclose(x[1], 0)

def top(x):
    return np.isclose(x[1], L)

def load_location(x):
    return np.logical_and(np.isclose(x[1], L), x[0] <= L/2)

fdim = domain.topology.dim - 1
left_facets = mesh.locate_entities_boundary(domain, fdim, left)
right_facets = mesh.locate_entities_boundary(domain, fdim, right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)

load_facets = mesh.locate_entities_boundary(domain, fdim, load_location)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([load_facets])
marked_values = np.hstack([np.full_like(load_facets, 5)])

# Here is not necessary to sort but we do it to keep consistency with other examples
sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

left_dofs_x = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, left_facets)
top_dofs_x = fem.locate_dofs_topological(V.sub(0), facet_tag.dim, top_facets)
bottom_dofs_y = fem.locate_dofs_topological(V.sub(1), facet_tag.dim, bottom_facets)

# No displacement along x on the left and on the top boundary 
# No displacement along y on the bottom boundary
u_bc = default_scalar_type(0)
bcs = [fem.dirichletbc(u_bc, left_dofs_x, V.sub(0)),
       fem.dirichletbc(u_bc, top_dofs_x, V.sub(0)),
       fem.dirichletbc(u_bc, bottom_dofs_y, V.sub(1))]

traction = fem.Constant(domain, default_scalar_type((0, 0)))


v = ufl.TestFunction(V)
u = fem.Function(V)

# Spatial dimensiona
d = len(u)
# Identity tensor
I = ufl.variable(ufl.Identity(d))
# Deformation gradient
F = ufl.variable(I + ufl.grad(u))
Finv_T = ufl.inv(F).T

# # Right Cauchy-Green tensor
C = ufl.variable(F.T * F)
# # Invariant of deformation tensors
Ic = ufl.variable(ufl.tr(C))
J = ufl.variable(ufl.det(F))

# Elasticity parameters
# Stored strain energy density (compressible neo-Hookean model)

mu = fem.Constant(domain, default_scalar_type(80.194))  #N/mm^2
lmbda = fem.Constant(domain, default_scalar_type(400889.8)) #N/mm^2

psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2
P = ufl.diff(psi, F)

# First Piola Stress tensor
# P = mu*(F - Finv_T) + lmbda * ufl.ln(J) * Finv_T

metadata = {"quadrature_degree": 4}
ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)
dx = ufl.Measure("dx", domain=domain, metadata=metadata)

# Define form F (we want to find u such that F(u) = 0)
residual = ufl.inner(ufl.grad(v), P) * dx - ufl.inner(v, traction) * ds(5)

problem = NonlinearProblem(residual, u, bcs)
solver = NewtonSolver(domain.comm, problem)

# Set Newton solver options
solver.atol = 1e-8
solver.rtol = 1e-8
solver.convergence_criterion = "incremental"

from pathlib import Path
pyvista.start_xvfb()
plotter = pyvista.Plotter()
current_directory = Path(__file__).resolve().parent
results_folder = Path(current_directory / 'results/')
results_folder.mkdir(exist_ok=True, parents=True)

plotter.open_gif(str(results_folder) + "/displacement.gif", fps=10)

topology, cells, geometry = plot.vtk_mesh(u.function_space)
function_grid = pyvista.UnstructuredGrid(topology, cells, geometry)

values = np.zeros((geometry.shape[0], 3))
values[:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
function_grid["u"] = values
function_grid.set_active_vectors("u")

# Warp mesh by deformation
warped = function_grid.warp_by_vector("u", factor=1)
warped.set_active_vectors("u")

# Add mesh to plotter and visualize
actor = plotter.add_mesh(warped, show_edges=True, lighting=False, clim=[0, 10])
# Show axes for reference
plotter.show_axes()

# Set view to see x-y plane clearly
plotter.view_xy()  # or plotter.view_isometric() for 3D view

# Compute magnitude of displacement to visualize in GIF
Vs = fem.functionspace(domain, ("Lagrange", pol_degree))
magnitude = fem.Function(Vs)
us = fem.Expression(ufl.sqrt(sum([u[i]**2 for i in range(len(u))])), Vs.element.interpolation_points())
magnitude.interpolate(us)
warped["mag"] = magnitude.x.array

log.set_log_level(log.LogLevel.INFO)
tval_fin = -600

top_point = np.array([[0, L, 0.0]])

n_times = 150
tval0 = tval_fin / n_times

load_time = np.linspace(0, tval_fin, n_times+1)
u_y_point = np.zeros((n_times+1))

# Find the cell containing the point
bb_tree = dolfinx.geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, top_point)
colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, top_point)
cells_top_point = []
if len(colliding_cells.links(0)) > 0:
    cells_top_point.append(colliding_cells.links(0)[0])

for n in range(1, n_times+1):
    traction.value[1] = n * tval0
    num_its, converged = solver.solve(u)
    assert (converged)
    u_y_point[n] = u.eval(top_point, cells=cells_top_point[0])[1]

    u.x.scatter_forward()
    print(f"Time step {n}, Number of iterations {num_its}, Load {traction.value}")
    function_grid["u"][:, :len(u)] = u.x.array.reshape(geometry.shape[0], len(u))
    magnitude.interpolate(us)
    warped.set_active_scalars("mag")
    warped_n = function_grid.warp_by_vector(factor=1)
    warped.points[:, :] = warped_n.points
    warped.point_data["mag"][:] = magnitude.x.array
    plotter.update_scalar_bar_range([0, 10])
    plotter.write_frame()
plotter.close()

import matplotlib.pyplot as plt
plt.figure()
plt.plot(-u_y_point, -load_time, '+')
plt.xlabel('Displacement at top left point (mm)')
plt.ylabel('Applied traction (N/mm²)')
plt.title('Load-Displacement Curve at Top Left Point')
plt.grid()
plt.savefig(results_folder / 'load_displacement_curve.pdf', format='pdf')
