import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector
from dolfinx.fem import Function
import ufl

class force_control:
    ''' The arc-length force control solver for FEniCSx
    
    An FEniCS based arc-length with prescribed traction. The code is heavily based on the paper:
    Kadapa, Chennakesava. "A simple extrapolated predictor for overcoming the starting and 
    tracking issues in the arc-length method for nonlinear structural mechanics." 
    Engineering Structures 234 (2021): 111755.
    
    Args:
        psi: the scalar arc-length parameter. When psi = 1, the method becomes the spherical 
             arc-length method and when psi = 0 the method becomes the cylindrical arc-length method
        lmbda0 : the initial load parameter
        max_iter : maximum number of iterations for the linear solver
        u : the solution function (dolfinx.fem.Function)
        F_int : First variation of strain energy (internal nodal forces) - UFL form
        F_ext : Externally applied load (external applied force) - UFL form
        J : The Jacobian of the residual with respect to the deformation (tangential stiffness matrix) - UFL form
        bcs : list of Dirichlet boundary conditions
        load_factor : The incremental load factor (fem.Constant)
        abs_tol (optional): absolute residual tolerance for the solver (default value: 1e-10)
        rel_tol (optional): relative residual tolerance for solver (default value: 1e-8)
        solver_type (optional): PETSc solver type (default: 'preonly')
        pc_type (optional): PETSc preconditioner type (default: 'lu')
    '''

    def __init__(self, psi, lmbda0, max_iter, u, F_int, F_ext, bcs, J, load_factor, 
                 abs_tol=1e-10, rel_tol=1e-8, solver_type='preonly', pc_type='lu'):
        # Initialize Variables
        self.psi = psi
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.lmbda = lmbda0
        self.max_iter = max_iter
        self.F_int = F_int
        self.F_ext = F_ext
        self.u = u
        self.J = J
        self.bcs = bcs
        self.load_factor = load_factor
        self.residual = F_int - F_ext
        self.solver_type = solver_type
        self.pc_type = pc_type
        self.counter = 0
        self.converged = True
        
        # Setup PETSc solver
        self.ksp = PETSc.KSP().create(u.function_space.mesh.comm)
        self.ksp.setType(solver_type)
        self.ksp.getPC().setType(pc_type)
        self.ksp.setTolerances(rtol=1e-10, atol=1e-10)
    
    def __update_nodal_values(self, u_new):
        '''
        Function to update solution (i.e. displacement) vector after each solver iteration
        
        Args:
            u_new: updated solution vector (PETSc.Vec)
        '''
        u_new.copy(result=self.u.x.petsc_vec)
        self.u.x.scatter_forward()

    def __initial_step(self):
        '''
        Initial step of the arc-length method. 
        '''        
        
        ii = 0
        print('Starting initial Force Control with Newton Method:')
        
        # Calculate inner(F,F)
        self.load_factor.value = 1.0  # for construction of FF
        self.F_ext_vec = create_vector(self.F_ext)
        assemble_vector(self.F_ext_vec, self.F_ext)
        
        # Apply boundary conditions to external force vector
        fem.petsc.apply_lifting(self.F_ext_vec, [self.J], [self.bcs])
        self.F_ext_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(self.F_ext_vec, self.bcs)
        
        self.FF = self.F_ext_vec.dot(self.F_ext_vec)
        
        self.load_factor.value = self.lmbda
        
        while True:
            # Assemble K and R
            K = create_matrix(self.J)
            assemble_matrix(K, self.J, bcs=self.bcs)
            K.assemble()
            
            R = create_vector(self.residual)
            assemble_vector(R, self.residual)
            fem.petsc.apply_lifting(R, [self.J], [self.bcs])
            R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(R, self.bcs)
            
            norm = R.norm(PETSc.NormType.NORM_2)
            
            # Define relative residual for first iteration
            if ii == 0:
                norm0 = norm
                
            print(f'Iteration {ii}: | \nAbsolute Residual: {norm:.4e}| Relative Residual: {norm/norm0:.4e}')
            if norm < self.abs_tol or norm/norm0 < self.rel_tol:
                self.delta_s = np.sqrt(
                    self.u.x.petsc_vec.dot(self.u.x.petsc_vec) + 
                    self.psi * self.lmbda**2 * self.FF
                )
                self.counter = 1
                K.destroy()
                R.destroy()
                break
            
            ii += 1
            assert ii <= self.max_iter, 'Newton Solver not converging'

            du = self.u.x.petsc_vec.duplicate()
            self.ksp.setOperators(K)
            self.ksp.solve(R, du)
            
            self.u.x.petsc_vec.axpy(-1.0, du)
            self.u.x.scatter_forward()
            
            K.destroy()
            R.destroy()
            du.destroy()
        
    def solve(self):
        '''
        Main function to increment through the arc-length scheme. 
        '''
        if self.counter == 0:
            print('Initializing solver parameters...')
            self.__initial_step()

        print('\nArc-Length Step', self.counter, ':')
        
        # Initialization
        if self.counter == 1:
            self.converged = False
            self.u_n = self.u.x.petsc_vec.duplicate()
            self.u_n_1 = self.u.x.petsc_vec.duplicate()
            self.u_n.set(0.0)
            self.u_n_1.set(0.0)
            self.lmbda_n = 0.0
            self.lmbda_n_1 = 0.0

        # Predictor Step:
        else:
            alpha = self.delta_s / self.delta_s_n
            
            # Calculate: u = (1+alpha) * u_n - alpha * u_n_1
            temp_vec = self.u.x.petsc_vec.duplicate()
            temp_vec.set(0.0)
            temp_vec.axpy((1+alpha), self.u_n)
            temp_vec.axpy(-alpha, self.u_n_1)
            self.__update_nodal_values(temp_vec)
            temp_vec.destroy()
            
            self.lmbda = (1+alpha) * self.lmbda_n - alpha * self.lmbda_n_1
         
        self.load_factor.value = self.lmbda
        
        # Calculate delta_u and delta_lmbda
        delta_u = self.u.x.petsc_vec.duplicate()
        delta_u.axpy(1.0, self.u.x.petsc_vec)
        delta_u.axpy(-1.0, self.u_n)
        delta_lmbda = self.lmbda - self.lmbda_n
           
        self.converged_prev = self.converged
        self.converged = False
        
        # Corrector Step (i.e. arc-length solver):
        solver_iter = 0
        norm = 1
        norm0 = 1

        while (norm > self.abs_tol or norm/norm0 > self.rel_tol) and solver_iter < self.max_iter:
            
            # Assemble K and R
            K = create_matrix(self.J)
            assemble_matrix(K, self.J, bcs=self.bcs)
            K.assemble()
            
            R = create_vector(self.residual)
            assemble_vector(R, self.residual)
            fem.petsc.apply_lifting(R, [self.J], [self.bcs])
            R.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(R, self.bcs)

            # Solve for d_lmbda, d_u:
            a = delta_u.duplicate()
            a.axpy(2.0, delta_u)
            b = 2 * self.psi * delta_lmbda * self.FF

            A = delta_u.dot(delta_u) + self.psi * delta_lmbda**2 * self.FF - self.delta_s**2
            R_norm = R.norm(PETSc.NormType.NORM_2)
            norm = np.sqrt(R_norm**2 + A**2)
            
            # Define relative residual for arc-length solver iteration
            if solver_iter == 0:
                norm0 = norm

            print(f'Iteration: {solver_iter} \n|Total Norm: {norm:.4e} |Residual: {R_norm:.4e} |A: {A:.4e}| Relative Norm: {norm/norm0:.4e}')
            if norm < self.abs_tol or norm/norm0 < self.rel_tol:
                self.converged = True
                a.destroy()
                K.destroy()
                R.destroy()
                break

            du_1 = self.u.x.petsc_vec.duplicate()
            du_2 = self.u.x.petsc_vec.duplicate()

            self.ksp.setOperators(K)
            self.ksp.solve(self.F_ext_vec, du_1)
            self.ksp.solve(R, du_2)

            dlmbda = (a.dot(du_2) - A) / (b + a.dot(du_1))
            
            # Calculate du = -du_2 + dlmbda * du_1
            du = du_2.duplicate()
            du.axpy(-1.0, du_2)
            du.axpy(dlmbda, du_1)

            # Update delta_u, delta_lmbda, u, lmbda
            delta_u.axpy(1.0, du)
            delta_lmbda += dlmbda
            
            self.u.x.petsc_vec.axpy(1.0, du)
            self.u.x.scatter_forward()

            self.lmbda += dlmbda
            self.load_factor.value = self.lmbda

            solver_iter += 1
            
            # Clean up
            a.destroy()
            du_1.destroy()
            du_2.destroy()
            du.destroy()
            K.destroy()
            R.destroy()

        # Solution Update
        if self.converged:
            if self.counter == 1:
                self.delta_s_max = self.delta_s
                self.delta_s_min = self.delta_s / 1024.0
            
            self.delta_s_n = self.delta_s
            
            # Update history
            self.u_n.copy(result=self.u_n_1)
            self.lmbda_n_1 = self.lmbda_n
        
            self.u.x.petsc_vec.copy(result=self.u_n)
            self.lmbda_n = self.lmbda
                
            self.counter += 1 
            
            if self.converged_prev:
                # Predictor update rule if solution converges
                self.delta_s = min(max(2*self.delta_s, self.delta_s_min), self.delta_s_max)
        else:
            # Predictor update rule if solution doesn't converge
            if self.converged_prev:
                # Rule if previous step converged
                self.delta_s = max(self.delta_s / 2, self.delta_s_min)
            else:
                # Rule if previous step did not converge
                self.delta_s = max(self.delta_s / 4, self.delta_s_min)
        
        # Clean up
        delta_u.destroy()
    
    def __del__(self):
        """Destructor to clean up PETSc objects"""
        if hasattr(self, 'ksp'):
            self.ksp.destroy()
        if hasattr(self, 'F_ext_vec'):
            self.F_ext_vec.destroy()
        if hasattr(self, 'u_n'):
            self.u_n.destroy()
        if hasattr(self, 'u_n_1'):
            self.u_n_1.destroy()