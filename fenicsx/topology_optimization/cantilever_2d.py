import numpy as np, sklearn.metrics.pairwise as sp
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type, log
import numpy
import ufl
from dolfinx import io
from pathlib import Path

## WORK IN PROGRESS

def main(nelx, nely, volfrac, penal, rmin):
    L, H = 180, 60
    
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])],
                         [nelx, nely], cell_type=mesh.CellType.quadrilateral)
    
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
    # D = fem.functionspace(domain, ("DG", 0))
    D = fem.functionspace(domain, ("CG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    v_0 = ufl.TestFunction(D)

    u_sol = fem.Function(V)
    density, density_old, density_new = fem.Function(D, name="density"), fem.Function(D), fem.Function(D)
    density.vector.set(volfrac)

    V0 = fem.assemble_vector(v_0 * ufl.dx)


    # DEFINE SUPPORT ---------------------------------------------------
    def support(x):
        return np.isclose(x[0], 0)
    
    def traction(x):
        return (np.isclose(x[0], L)) & (x[1] <=1)

    fdim = domain.topology.dim - 1
    support_facets = mesh.locate_entities_boundary(domain, fdim, support)
    traction_facets = mesh.locate_entities_boundary(domain, fdim, traction)

    u_D = np.array([0, 0], dtype=default_scalar_type)

    bcs = [fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, support_facets), V)]

    # DEFINE LOAD ------------------------------------------------------

    # load_marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    # CompiledSubDomain("x[0]==l && x[1]<=1", l=nelx).mark(load_marker, 1)
    # ds = Measure("ds")(subdomain_data=load_marker)
    # F = dot(v, Constant((0.0, -1.0))) * ds(1)

    # # SET UP THE VARIATIONAL PROBLEM AND SOLVER ------------------------
    mu, lmbda = fem.Constant(0.3846), fem.Constant(0.5769)
    sigma = lambda _u: 2.0 * mu * ufl.sym(ufl.grad(_u)) \
        + lmbda * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
    
    psi = lambda _u: lmbda / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) \
        + mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))
    
    # K = inner(density ** penal * sigma(u), grad(v)) * dx
    # solver = LinearVariationalSolver(LinearVariationalProblem(K, F, u_sol, bcs))
    # # PREPARE DISTANCE MATRICES FOR FILTER -----------------------------
    # midpoint = [cell.midpoint().array()[:] for cell in cells(mesh)]
    # distance_mat = np.maximum(rmin - sp.euclidean_distances(midpoint, midpoint), 0)
    # distance_sum = distance_mat.sum(1)

    # # Results folder 
    # current_directory = Path(__file__).resolve().parent
    # results_folder = Path(str(current_directory) + "/results")
    # results_folder.mkdir(exist_ok=True, parents=True)
    # filename = results_folder / "density"

    # with io.XDMFFile(mesh.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
    #     xdmf.write_mesh(mesh)

    # # START ITERATION --------------------------------------------------
    # loop, change = 0, 1
    # while change > 0.01 and loop < 100:
    #     loop = loop + 1
    #     density_old.assign(density)
    #     # FE-ANALYSIS --------------------------------------------------
    #     solver.solve()
    #     # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS ------------------
    #     objective = density ** penal * psi(u_sol)
    #     sensitivity = project(-diff(objective, density), D).vector()[:]
    #     # FILTERING/MODIFICATION OF SENSITIVITIES ----------------------
    #     sensitivity = np.divide(distance_mat @ np.multiply(density.vector()[:], sensitivity), np.multiply(density.vector()[:], distance_sum))
    #     # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD --------------
    #     l1, l2, move = 0, 100000, 0.2
    #     while l2 - l1 > 1e-4:
    #         l_mid = 0.5 * (l2 + l1)
    #         density_new.vector()[:] = np.maximum(0.001, np.maximum(density.vector()[:] - move, np.minimum(1.0, np.minimum(density.vector()[:] + move, density.vector()[:] * np.sqrt(-sensitivity / V0 / l_mid)))))
    #         current_vol = assemble(density_new * dx)
    #         l1, l2 = (l_mid, l2) if current_vol > volfrac * V0.sum() else (l1, l_mid)
    #     # PRINT RESULTS ------------------------------------------------
    #     change = max(density_new.vector()[:] - density_old.vector()[:])
    #     print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, project(objective, D).vector().sum(), current_vol / V0.sum(), change))
    #     density.assign(density_new)

    #     xdmf.write_function(density, loop)


# The real main driver
if __name__ == "__main__":
    log.set_log_level(log.LogLevel.WARNING)
    main(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=2.0)