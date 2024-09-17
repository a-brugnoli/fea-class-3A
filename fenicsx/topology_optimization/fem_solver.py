from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
L = 20
W = 5
E = default_scalar_type(1.0e4)
nu = default_scalar_type(0.3)

traction_y = -10
domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, W])],
                         [20, 6], cell_type=mesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))

mu = fem.Constant(domain, E / (2 * (1 + nu)))
lmbda = fem.Constant(domain, E * nu / (1 - nu**2))

def left(x):
    return np.isclose(x[0], 0)

def right(x):
    
    return (np.isclose(x[0], L)) & (x[1] <= 1)

fdim = domain.topology.dim - 1
clamped_facets = mesh.locate_entities_boundary(domain, fdim, left)
traction_facets = mesh.locate_entities_boundary(domain, fdim, right)

# Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
marked_facets = np.hstack([clamped_facets, traction_facets])
marked_values = np.hstack([np.full_like(clamped_facets, 1), \
                           np.full_like(traction_facets, 2)])

sorted_facets = np.argsort(marked_facets)
facet_tag = mesh.meshtags(domain, fdim, marked_facets[sorted_facets], \
                          marked_values[sorted_facets])

u_bc = np.array((0,) * domain.geometry.dim, dtype=default_scalar_type)

left_dofs = fem.locate_dofs_topological(V, facet_tag.dim, facet_tag.find(1))
bc = fem.dirichletbc(u_bc, left_dofs, V)

def epsilon(u):
    return ufl.sym(ufl.grad(u))  

def sigma(u):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx

T = fem.Constant(domain, default_scalar_type((0, traction_y)))
ds = ufl.Measure("ds",  domain=domain, subdomain_data=facet_tag)
L = ufl.dot(T, v) * ds(2)

problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

s = sigma(uh) - 1. / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

try:
    import pyvista

    topology, cell_types, geometry = plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    u_array_2d = uh.x.array.reshape((geometry.shape[0], domain.geometry.dim))
    u_array_3d = np.hstack((u_array_2d, np.zeros((u_array_2d.shape[0], 1))))
    grid["u"] = u_array_3d 
    warped = grid.warp_by_vector("u", factor=1.5)
    
    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True)
    plotter.add_mesh(warped, show_edges=True)
    plotter.show_axes()
    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb(wait=0.1)
        plotter.screenshot("uh.png")
    else:
        plotter.show()
except ModuleNotFoundError:
    print("'pyvista' is required to visualise the solution")
    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")