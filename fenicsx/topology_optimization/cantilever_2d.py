import numpy as np, sklearn.metrics.pairwise as sp
from mpi4py import MPI
from dolfinx import fem, default_scalar_type, log, plot
from dolfinx.fem.petsc import create_vector, assemble_vector, \
                            LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags, \
    create_rectangle, CellType 
import ufl
from dolfinx import io
from pathlib import Path
import pyvista


def topopt(nelx, nely, volfrac, penal, rmin):

    L, H = nelx, nely
    
    domain = create_rectangle(MPI.COMM_WORLD, [np.array([0, 0]), np.array([L, H])],
                         [nelx, nely], cell_type=CellType.quadrilateral)
    
    
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, )))
    D = fem.functionspace(domain, ("DG", 0))
    u  = ufl.TrialFunction(V)
    v  = ufl.TestFunction(V)

    v_0 = ufl.TestFunction(D)

    density  = fem.Function(D, name="density")
    density_old, density_new = fem.Function(D), fem.Function(D)

    with density.vector.localForm() as density_loc:
        density_loc.set(volfrac)

    print(f"Density: {density.vector.array}")

    # DEFINE SUPPORT ---------------------------------------------------
    def support(x):
        return np.isclose(x[0], 0)
    
    def traction(x):
        return (np.isclose(x[0], L) & np.less_equal(x[1], 1))

    fdim = domain.topology.dim - 1
    support_facets = locate_entities_boundary(domain, fdim, support)
    traction_facets = locate_entities_boundary(domain, fdim, traction)

    # Concatenate and sort the arrays based on facet indices. Left facets marked with 1, right facets with two
    marked_facets = np.hstack([support_facets, traction_facets])
    marked_values = np.hstack([np.full_like(support_facets, 1), \
                               np.full_like(traction_facets, 2)])
    sorted_facets = np.argsort(marked_facets)
    facet_tag = meshtags(domain, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

    u_D = np.array([0, 0], dtype=default_scalar_type)

    support_dofs = fem.locate_dofs_topological(V, fdim, support_facets)
    bcs = [fem.dirichletbc(u_D, support_dofs, V)]

    # Define measures
    metadata = {"quadrature_degree": 2}
    dx = ufl.Measure("dx", domain=domain, metadata=metadata)
    ds = ufl.Measure('ds', domain=domain, subdomain_data=facet_tag, metadata=metadata)

    volume_lin = v_0 * dx
    volume_linear_form = fem.form(volume_lin)
    volume_vec = create_vector(volume_linear_form)
    assemble_vector(volume_vec, volume_linear_form)

    volume_values = volume_vec.array.copy()

    # DEFINE LOAD ------------------------------------------------------
    F = ufl.dot(v, fem.Constant(domain, (0.0, -1))) * ds(2)

    # # # SET UP THE VARIATIONAL PROBLEM AND SOLVER ------------------------

    mu, lmbda = fem.Constant(domain, 0.4), fem.Constant(domain, 0.6)

    sigma = lambda _u: 2.0 * mu * ufl.sym(ufl.grad(_u)) \
        + lmbda * ufl.tr(ufl.sym(ufl.grad(_u))) * ufl.Identity(len(_u))
    
    psi = lambda _u: lmbda / 2 * (ufl.tr(ufl.sym(ufl.grad(_u))) ** 2) \
        + mu * ufl.tr(ufl.sym(ufl.grad(_u)) * ufl.sym(ufl.grad(_u)))
    

    K = ufl.inner(density ** penal * sigma(u), ufl.grad(v)) * dx
    problem = LinearProblem(K, F, bcs)

    # PREPARE DISTANCE MATRICES FOR FILTER -----------------------------

    num_elems = density.vector.array.size
    midpoints = density.function_space.tabulate_dof_coordinates()[:num_elems]
    
    distance_mat = np.maximum(rmin - sp.euclidean_distances(midpoints, midpoints), 0)
    distance_sum = distance_mat.sum(1)

    # Results folder 
    current_directory = Path(__file__).resolve().parent
    results_folder = Path(str(current_directory) + "/results")
    results_folder.mkdir(exist_ok=True, parents=True)
    filename = results_folder / "density"

    with io.XDMFFile(domain.comm, filename.with_suffix(".xdmf"), "w") as xdmf:
        xdmf.write_mesh(domain)

    # START ITERATION --------------------------------------------------
    loop, change = 0, 1
    while change > 0.01 and loop < 400:
        loop = loop + 1

        # Assign density to density_old
        with density_old.vector.localForm() as density_old_loc:
            with density.vector.localForm() as density_loc:
                density_old_loc[:] = density_loc


        # FE-ANALYSIS --------------------------------------------------
        u_sol = problem.solve()

        # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS ------------------
        compliance = density ** penal * psi(u_sol) * dx

        dCdrho_form = fem.form(-ufl.derivative(compliance, density))
        dCdrho_vec = create_vector(dCdrho_form)
        assemble_vector(dCdrho_vec, dCdrho_form)

        # FILTERING/MODIFICATION OF SENSITIVITIES ----------------------
        density_vec = density.vector
        density_values = density_vec.array.copy()
        dCdrho_values = dCdrho_vec.array.copy()

        dCdrho_values = np.divide(distance_mat @ np.multiply(density_values, dCdrho_values),\
                            np.multiply(density_values, distance_sum))
        
        # DESIGN UPDATE BY THE OPTIMALITY CRITERIA METHOD --------------
        l1, l2, move = 0, 1e5, 0.2
        while l2 - l1 > 1e-4:
            l_mid = 0.5 * (l2 + l1)
            density_new_values = np.maximum(0.001, \
                np.maximum(density_values - move,\
                           np.minimum(1.0, 
                                    np.minimum(density_values + move, \
                                            density_values * np.sqrt(-dCdrho_values / volume_values / l_mid)
                    ))))
            
            with density_new.vector.localForm() as rho_new_local:
                rho_new_local[:] = density_new_values


            current_vol = fem.assemble_scalar(fem.form(density_new * dx))
            l1, l2 = (l_mid, l2) if current_vol > volfrac * volume_vec.sum() else (l1, l_mid)
        # PRINT RESULTS ------------------------------------------------

        change = max(density_new.vector[:] - density_old.vector[:])
        current_compliance = fem.assemble_scalar(fem.form(compliance))
        print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(loop, \
                                        current_compliance,\
                                        current_vol / volume_vec.sum(), \
                                        change))
        
        with density.vector.localForm() as density_loc:
            with density_new.vector.localForm() as density_new_loc:
                density_loc[:] = density_new_loc

        xdmf.write_function(density, loop)




# The real main driver
if __name__ == "__main__":
    log.set_log_level(log.LogLevel.WARNING)
    topopt(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=2.0)

    # Test 55 lines works
