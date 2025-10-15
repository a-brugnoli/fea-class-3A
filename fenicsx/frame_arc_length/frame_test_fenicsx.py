import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
from dolfinx import fem, mesh, io
from dolfinx.fem import (Function, functionspace, dirichletbc, 
                         Constant, locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import locate_entities, meshtags

import ufl
from ufl import (TrialFunction, TestFunction, Identity, grad, det, tr, ln, 
                 dx, ds, dot, derivative, inner)

from force_control_fenicsx import force_control

# Note: You'll need to convert the mesh from XML to XDMF format
# Use: dolfin-convert mesh/lees-frame-2d.xml mesh/lees-frame-2d.xdmf

# Parameters
h = 10
L = 1200
l1 = L / 5
E = 7200
nu = 0

# Read mesh - assuming conversion from XML to XDMF
# For a new mesh, you would create it differently

import os 
print(os.getcwd())
try:

    with XDMFFile(MPI.COMM_WORLD, "./fenicsx/frame_arc_length/lee_frame.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    print("Mesh successfully read.")
except:
    print("Warning: Could not read mesh file. Please convert XML mesh to XDMF format.")
    print("Use: dolfin-convert mesh/lees-frame-2d.xml mesh/lees-frame-2d.xdmf")
    # Create a simple placeholder mesh for demonstration
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0, 0]), np.array([L+2*h, L+h])],
        [50, 50],
        mesh.CellType.triangle
    )

# Define function space
V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Define functions
du = TrialFunction(V)
v = TestFunction(V)
u = Function(V)

# Boundary condition functions
def Hinge1(x):
    return np.logical_and(
        np.isclose(x[0], 2*h, atol=1e-6),
        np.isclose(x[1], 0.0, atol=1e-6)
    )

def Hinge2(x):
    return np.logical_and(
        np.isclose(x[0], L+h, atol=1e-6),
        np.isclose(x[1], L-h, atol=1e-6)
    )

# Locate boundary DOFs
hinge1_dofs = locate_dofs_geometrical(V, Hinge1)
hinge2_dofs = locate_dofs_geometrical(V, Hinge2)

# Create boundary conditions
u_zero = np.array([0.0, 0.0], dtype=PETSc.ScalarType)
bc1 = dirichletbc(u_zero, hinge1_dofs, V)
bc2 = dirichletbc(u_zero, hinge2_dofs, V)
bcs = [bc1, bc2]

# Mark facets for force application
def force_boundary(x):
    return np.logical_and(
        x[0] >= l1 - l1/25,
        x[0] <= l1 + l1/25
    )

# Get facet dimension
tdim = domain.topology.dim
fdim = tdim - 1

# Locate facets for force application
domain.topology.create_connectivity(fdim, tdim)
force_facets = locate_entities(domain, fdim, force_boundary)

# Create meshtags for facets
facet_indices = np.array(force_facets, dtype=np.int32)
facet_markers = np.full(len(facet_indices), 1, dtype=np.int32)
facets = meshtags(domain, fdim, facet_indices, facet_markers)

# Define point load function
class PointLoadExpression:
    def __init__(self, x0, f, tol=1e-4):
        self.x0 = x0
        self.f = f
        self.tol = tol
    
    def __call__(self, x):
        values = np.zeros((2, x.shape[1]), dtype=PETSc.ScalarType)
        # Check if points are near the load location
        mask = np.logical_and(
            np.isclose(x[0], self.x0[0], atol=self.tol),
            np.isclose(x[1], self.x0[1], atol=self.tol)
        )
        values[0, mask] = self.f[0]
        values[1, mask] = self.f[1]
        return values

# Create point load expression
point_load_expr = PointLoadExpression(x0=[l1, L+h], f=[0, 1], tol=1e-4)

# Interpolate point load into function space
P = Function(V)
P.interpolate(point_load_expr)

# Kinematics
d = domain.geometry.dim
I = Identity(d)
F = I + grad(u)
C = F.T * F

# Invariants
Ic = tr(C)
J = det(F)

# Material parameters
mu = Constant(domain, PETSc.ScalarType(E/(2*(1 + nu))))
lmbda = Constant(domain, PETSc.ScalarType(E*nu/((1 + nu)*(1 - 2*nu))))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Load parameter
load = Constant(domain, PETSc.ScalarType(0.0))

# Applied loads
T = Constant(domain, PETSc.ScalarType((0.0, 1.0)))
B = Constant(domain, PETSc.ScalarType((0.0, 0.0)))

# Define measure for integration over marked facets
metadata = {"quadrature_degree": 4}
ds_custom = ufl.Measure('ds', domain=domain, subdomain_data=facets, metadata=metadata)

# Variational forms
F_int = derivative(psi*dx, u, v)
F_ext = derivative(dot(B, u)*dx(metadata=metadata) + load*dot(P, u)*ds_custom(1), u, v)
residual = F_int - F_ext
J_form = derivative(residual, u, du)

print("Setup complete. Ready for arc-length solver.")
print(f"Number of DOFs: {V.dofmap.index_map.size_global * V.dofmap.index_map_bs}")
print(f"Mesh cells: {domain.topology.index_map(tdim).size_global}")
print(f"Force application facets: {len(force_facets)}")

# Visualization setup
try:
    import pyvista
    # pyvista.start_xvfb()  # For headless rendering if needed
    
    # Create plotter
    plotter = pyvista.Plotter()
    
    # Extract mesh topology and geometry
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Get point load values
    P_values = P.x.array.reshape((-1, d))
    P_magnitudes = np.linalg.norm(P_values, axis=1)
    grid.point_data["PointLoad"] = P_magnitudes
    
    # Plot
    plotter.add_mesh(grid, scalars="PointLoad", show_edges=True, 
                     cmap="Reds", edge_color="black", line_width=0.5)
    plotter.view_xy()
    plotter.show()
    
except ImportError:
    print("PyVista not available for visualization.")
    print("Install with: pip install pyvista")


# Solver Parameters
psi = 1.0
abs_tol = 1.0e-6
lmbda0 = 4.0
max_iter = 30

# Set up arc-length solver
solver = force_control(psi=psi, abs_tol=abs_tol, lmbda0=lmbda0, max_iter=max_iter, u=u,
                       F_int=F_int, F_ext=F_ext, bcs=bcs, J=J, load_factor=load)


disp = [u.x.array]
lmbda = [0]

for ii in range(0,38):
    solver.solve()
    if solver.converged:       
        disp.append(u.x.array)
        lmbda.append(load.t)
