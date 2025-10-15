from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem import (Constant, Function, functionspace, 
                         dirichletbc, locate_dofs_geometrical)
import ufl
from ufl import (grad, tr, det, ln, Identity, derivative, dx, ds, 
                 TrialFunction, TestFunction, inner, dot)
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import matplotlib.pyplot as plt

# Import the original force control solver - UNCHANGED
from force_control_fenicsx import force_control

# Parameters
h = 10
L = 1200
l1 = L / 5
E = 7200
nu = 0

# Read mesh (convert XML to XDMF first using: meshio convert lees-frame-2d.xml lees-frame-2d.xdmf)
try:
    with io.XDMFFile(MPI.COMM_WORLD, "mesh/lee_frame.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
except:
    print("Warning: Could not load mesh. Creating placeholder mesh.")
    print("Convert mesh using: meshio convert mesh/lees-frame-2d.xml mesh/lees-frame-2d.xdmf")
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[0, 0], [L+2*h, L+h]],
        [120, 110],
        mesh.CellType.triangle
    )

# Define function space
V = functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

# Define functions
u = Function(V, name="Displacement")
v = TestFunction(V)
du = TrialFunction(V)

# Boundary conditions
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

# Locate DOFs for boundary conditions
hinge1_dofs = locate_dofs_geometrical(V, Hinge1)
hinge2_dofs = locate_dofs_geometrical(V, Hinge2)

# Create boundary conditions
bc1 = dirichletbc(np.array([0.0, 0.0], dtype=default_scalar_type), 
                  hinge1_dofs, V)
bc2 = dirichletbc(np.array([0.0, 0.0], dtype=default_scalar_type), 
                  hinge2_dofs, V)
bcs = [bc1, bc2]

# Create a load parameter class compatible with original code
class LoadParameter:
    """Wrapper to mimic dolfin Expression behavior"""
    def __init__(self, value=0.0):
        self.t = value
        self._constant = None
        self._domain = None
    
    def set_domain(self, domain):
        self._domain = domain
        self._constant = Constant(domain, default_scalar_type(self.t))
    
    def get_constant(self):
        if self._constant is None:
            raise RuntimeError("Domain not set for LoadParameter")
        return self._constant
    
    def update(self):
        if self._constant is not None:
            self._constant.value = self.t

load = LoadParameter(0.0)
load.set_domain(domain)

# Kinematics
d = len(u)
I = Identity(d)
F = I + grad(u)
C = F.T * F

# Invariants
Ic = tr(C)
J = det(F)

# Material parameters
mu = Constant(domain, default_scalar_type(E/(2*(1 + nu))))
lmbda_mat = Constant(domain, default_scalar_type(E*nu/((1 + nu)*(1 - 2*nu))))

# Strain energy density (compressible neo-Hookean)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda_mat/2)*(ln(J))**2

# Define force region for distributed load
def force_region(x):
    return np.logical_and(
        np.logical_and(x[0] >= l1 - l1/25, x[0] <= l1 + l1/25),
        np.isclose(x[1], L+h, atol=h/10)
    )

# Mark facets for loading
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

facet_indices, facet_markers = [], []
facets = mesh.locate_entities(domain, fdim, force_region)
facet_indices.append(facets)
facet_markers.append(np.full_like(facets, 1))

facet_indices = np.hstack(facet_indices).astype(np.int32)
facet_markers = np.hstack(facet_markers).astype(np.int32)
sorted_facets = np.argsort(facet_indices)
facet_tag = mesh.meshtags(domain, fdim, facet_indices[sorted_facets], 
                          facet_markers[sorted_facets])

# Define measure with marked facets
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tag)

# Point load vector (concentrated at one location)
T = Constant(domain, default_scalar_type((0.0, 1.0)))

# Body force
B = Constant(domain, default_scalar_type((0.0, 0.0)))

# Variational formulation
F_int = derivative(psi*dx, u, v)
# Use load.get_constant() to get the FEniCSx Constant object
F_ext = derivative(dot(B, u)*dx + load.get_constant()*dot(T, u)*ds(1), u, v)

residual = F_int - F_ext
J = derivative(residual, u, du)

# Visualization setup
try:
    import pyvista
    pyvista.start_xvfb()  # For headless rendering if needed
except:
    pass

# Solver Parameters - SAME AS ORIGINAL
psi_arc = 1.0
abs_tol = 1.0e-6
lmbda0 = 4.0
max_iter = 30

# Set up arc-length solver - USING ORIGINAL force_control FUNCTION
# Note: The force_control function will need to handle FEniCSx objects
# If it was written for dolfin, you may need a compatibility wrapper
solver = force_control(psi=psi_arc, abs_tol=abs_tol, lmbda0=lmbda0, 
                       max_iter=max_iter, u=u,
                       F_int=F_int, F_ext=F_ext, bcs=bcs, J=J, 
                       load_factor=load)

# Storage for results
disp = [u.x.array.copy()]
lmbda = [0]

# Solution loop - SAME AS ORIGINAL
print("Starting arc-length continuation...")
for ii in range(0, 38):
    print(f"\nStep {ii+1}/38")
    
    # Update the constant before solving
    load.update()
    
    solver.solve()
    
    if solver.converged:
        disp.append(u.x.array.copy())
        lmbda.append(load.t)
        print(f"  Converged. Load factor: {load.t:.6f}")
    else:
        print(f"  Failed to converge")
        break

print(f"\nCompleted {len(lmbda)-1} steps")

# Get force node DOF
coords = V.tabulate_dof_coordinates()
point_coords = np.array([l1, L+h])

# Find closest point
distances = np.linalg.norm(coords - point_coords, axis=1)
closest_idx = np.argmin(distances)

# Get y-component DOF (assuming [x, y, x, y, ...] ordering)
force_node = closest_idx
if force_node % 2 == 0:
    force_node += 1

# Extract displacement at force point
force_disp = []
for ii in range(0, len(disp)):
    force_disp.append(-disp[ii][force_node])

# Plot load-displacement curve - SAME AS ORIGINAL
plt.figure(figsize=(7, 7))
plt.plot(force_disp, lmbda, marker='o', color='k')
plt.xlabel('Displacement')
plt.ylabel('Load Factor')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('load_displacement_curve.png', dpi=150)
print("\nSaved load-displacement curve")

# Visualize final deformation
try:
    import pyvista
    topology, cell_types, geometry = io.vtxwriter.create_vtk_mesh(V)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    
    # Add displacement
    values = np.zeros((geometry.shape[0], 3))
    values[:, :domain.geometry.dim] = u.x.array.reshape(-1, domain.geometry.dim)
    grid.point_data["Displacement"] = values
    
    # Warp by displacement
    warped = grid.warp_by_vector("Displacement", factor=1.0)
    
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_mesh(warped, show_edges=True, edge_color='k', 
                     line_width=0.5, color='red')
    plotter.view_xy()
    plotter.screenshot('final_deformation.png')
    print("Saved final deformation visualization")
    
except Exception as e:
    print(f"Visualization skipped: {e}")

plt.show()

print("\nConversion complete!")
print("Note: Your force_control solver must be compatible with FEniCSx objects.")
print("If it uses dolfin-specific methods, you may need to create a wrapper.")