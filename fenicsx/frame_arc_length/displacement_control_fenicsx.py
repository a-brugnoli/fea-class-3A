import numpy as np
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_matrix, create_vector
from dolfinx.fem import Function
import ufl

class displacement_control: 
    ''' The arc-length displacement control solver for FEniCSx
    
    Args:
        psi: the scalar arc-length parameter. When psi = 1, the method becomes the spherical arc-length method and when psi = 0 the method becomes the cylindrical arc-length method
        lmbda0 : the initial displacement parameter
        max_iter : maximum number of iterations for the linear solver
        u : the solution function (dolfinx.fem.Function)
        F_int : First variation of strain energy (internal nodal forces) - UFL form
        F_ext : Externally applied load (external applied force) - UFL form
        J : The Jacobian of the residual with respect to the deformation (tangential stiffness matrix) - UFL form
        bcs : list of Dirichlet boundary conditions
        displacement_factor : The incremental displacement factor (fem.Constant)
        abs_tol (optional): absolute residual tolerance for the solver (default value: 1e-10)
        rel_tol (optional): relative residual tolerance for solver (default value: 1e-8)
        solver_type (optional): PETSc solver type (default: 'preonly')
        pc_type (optional): PETSc preconditioner type (default: 'lu')
    '''

    def __init__(self, psi, lmbda0, max_iter, u, F_int, F_ext, bcs, J, displacement_factor, 
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
        self.displacement_factor = displacement_factor
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
        For the displacement control formulation, this function constructs the constraint matrix and the initial arc-length step size.
        '''    

        ii = 0
        print('Starting initial Displacement Control with Newton Method:')
        
        # Find DoFs for homogeneous and non-homogeneous Dirichlet BCs:
        self.displacement_factor.value = 1.0  # set non-homogeneous DoFs to 1 
        
        # Get boundary DOFs from all BCs
        total_dofs = self.u.function_space.dofmap.index_map.size_local * self.u.function_space.dofmap.index_map_bs
        all_bc_dofs = []
        all_bc_values = []
        
        for bc in self.bcs:
            bc_dofs = bc.dof_indices()[0]
            all_bc_dofs.extend(bc_dofs)
            
        all_bc_dofs = np.unique(np.array(all_bc_dofs, dtype=np.int32))
        
        # Separate homogeneous and non-homogeneous DOFs
        self.displacement_factor.value = 1.0
        temp_vec = self.u.x.petsc_vec.duplicate()
        temp_vec.set(0.0)
        
        for bc in self.bcs:
            bc.set(temp_vec, 1.0)
        
        temp_array = temp_vec.getArray()
        
        self.dofs_hom = []
        self.dofs_nonhom = []
        
        for dof in all_bc_dofs:
            if abs(temp_array[dof]) < 1e-12:
                self.dofs_hom.append(dof)
            else:
                self.dofs_nonhom.append(dof)
        
        # Free DOFs
        all_dofs = np.arange(0, total_dofs, dtype=np.int32)
        self.dofs_free = np.setdiff1d(all_dofs, np.concatenate([self.dofs_hom, self.dofs_nonhom]))
        
        # Set up DOF vector of Dirichlet BCs:
        self.u_p = self.u.x.petsc_vec.duplicate()
        self.u_p.set(0.0)
        u_p_array = self.u_p.getArray()
        for dof in self.dofs_nonhom:
            u_p_array[dof] = 1.0
        self.u_p.setArray(u_p_array)
        
        # Construct Constraint Matrix C
        C_mat = PETSc.Mat().create(comm=self.u.function_space.mesh.comm)
        C_mat.setSizes((total_dofs, len(self.dofs_free)))
        C_mat.setType('aij')
        C_mat.setUp()
        
        for j, i in enumerate(self.dofs_free):
            C_mat.setValue(i, j, 1.0)
        
        C_mat.assemble()
        self.C_mat = C_mat
        
        # Initialize vectors
        self.u_f = self.C_mat.createVecRight()
        R_star = self.C_mat.createVecRight()
        self.Q = self.C_mat.createVecRight()
        du = self.C_mat.createVecLeft()

        # Apply all Dirichlet BCs to u:
        self.displacement_factor.value = self.lmbda
        for bc in self.bcs:
            bc.set(self.u.x.petsc_vec, self.lmbda)
        self.u.x.scatter_forward()

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

            # Get K*, Q, and R* (reduced matrices and vectors)
            K_star_mat = self.C_mat.transposeMatMult(K)
            K_star_mat = K_star_mat.matMult(self.C_mat)
            
            temp_mat = self.C_mat.transpose()
            temp_mat = temp_mat.matMult(K)
            temp_mat.mult(-self.u_p, self.Q)
            
            self.C_mat.multTranspose(R, R_star)

            norm = R_star.norm(PETSc.NormType.NORM_2)

            # Define relative residual for first iteration
            if ii == 0:
                norm0 = norm
            
            print(f'Iteration {ii}: | \nAbsolute Residual: {norm:.4e}| Relative Residual: {norm/norm0:.4e}')
            if norm < self.abs_tol or norm/norm0 < self.rel_tol:
                self.delta_s = np.sqrt(self.u_f.dot(self.u_f) + self.psi * self.lmbda**2 * self.Q.dot(self.Q))
                self.counter = 1
                K.destroy()
                R.destroy()
                K_star_mat.destroy()
                temp_mat.destroy()
                break

            ii += 1
            assert ii <= self.max_iter, 'Newton Solver not converging'

            du_f = self.C_mat.createVecRight()
            self.ksp.setOperators(K_star_mat)
            self.ksp.solve(R_star, du_f)
            
            self.C_mat.mult(du_f, du)
            
            self.u.x.petsc_vec.axpy(-1.0, du)
            self.u.x.scatter_forward()
            self.u_f.axpy(-1.0, du_f)
            
            K.destroy()
            R.destroy()
            K_star_mat.destroy()
            temp_mat.destroy()
        
        R_star.destroy()
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
        u_update = self.u.x.petsc_vec.duplicate()
        R_star = self.C_mat.createVecRight()
        
        if self.counter == 1:
            self.converged = False
            self.u_f_n = self.u_f.duplicate()
            self.u_f_n_1 = self.u_f.duplicate()
            self.lmbda_n = 0.0
            self.lmbda_n_1 = 0.0

        # Predictor Step: 
        else:
            alpha = self.delta_s / self.delta_s_n
            self.u_f.set(0.0)
            self.u_f.axpy((1+alpha), self.u_f_n)
            self.u_f.axpy(-alpha, self.u_f_n_1)
            
            self.C_mat.mult(self.u_f, u_update)
            self.__update_nodal_values(u_update)
            
            self.lmbda = (1+alpha) * self.lmbda_n - alpha * self.lmbda_n_1
            
        # Apply boundary conditions
        self.displacement_factor.value = self.lmbda
        for bc in self.bcs:
            bc.set(self.u.x.petsc_vec, self.lmbda)
        self.u.x.scatter_forward()
                    
        delta_u_f = self.u_f.duplicate()
        delta_u_f.axpy(1.0, self.u_f)
        delta_u_f.axpy(-1.0, self.u_f_n)
        
        delta_lmbda = self.lmbda - self.lmbda_n

        self.converged_prev = self.converged
        self.converged = False
        
        # Corrector Step (arc-length solver):
        solver_iter = 0
        norm = 1
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

            # Get K*, Q, and R* (reduced matrices and vectors)
            K_star_mat = self.C_mat.transposeMatMult(K)
            K_star_mat = K_star_mat.matMult(self.C_mat)
            
            temp_mat = self.C_mat.transpose()
            temp_mat = temp_mat.matMult(K)
            temp_mat.mult(-self.u_p, self.Q)
            
            self.C_mat.multTranspose(R, R_star)
            
            QQ = self.Q.dot(self.Q)

            # Solve for d_lmbda, d_u:
            a = delta_u_f.duplicate()
            a.axpy(2.0, delta_u_f)
            b = 2 * self.psi * delta_lmbda * QQ

            A = delta_u_f.dot(delta_u_f) + self.psi * delta_lmbda**2 * QQ - self.delta_s**2
            R_star_norm = R_star.norm(PETSc.NormType.NORM_2)
            norm = np.sqrt(R_star_norm**2 + A**2)

            # Define relative residual for arc-length solver iteration
            if solver_iter == 0:
                norm0 = norm

            print(f'Iteration: {solver_iter} \n|Total Norm: {norm:.4e} |Residual: {R_star_norm:.4e} |A: {A:.4e}| Relative Norm: {norm/norm0:.4e}')
            if norm < self.abs_tol or norm/norm0 < self.rel_tol:
                self.converged = True
                K.destroy()
                R.destroy()
                K_star_mat.destroy()
                temp_mat.destroy()
                a.destroy()
                break

            du_f_1 = self.C_mat.createVecRight()
            du_f_2 = self.C_mat.createVecRight()

            self.ksp.setOperators(K_star_mat)
            self.ksp.solve(self.Q, du_f_1)
            self.ksp.solve(R_star, du_f_2)

            dlmbda = (a.dot(du_f_2) - A) / (b + a.dot(du_f_1))
            du_f = du_f_2.duplicate()
            du_f.axpy(-1.0, du_f_2)
            du_f.axpy(dlmbda, du_f_1)

            # Update delta_u, delta_lmbda, u, lmbda
            delta_lmbda += dlmbda
            self.lmbda += dlmbda
            delta_u_f.axpy(1.0, du_f)
            self.u_f.axpy(1.0, du_f)
            
            self.C_mat.mult(self.u_f, u_update)
            self.__update_nodal_values(u_update)
            self.displacement_factor.value = self.lmbda

            solver_iter += 1

            for bc in self.bcs:
                bc.set(self.u.x.petsc_vec, self.lmbda)
            self.u.x.scatter_forward()
            
            K.destroy()
            R.destroy()
            K_star_mat.destroy()
            temp_mat.destroy()
            a.destroy()
            du_f_1.destroy()
            du_f_2.destroy()
            du_f.destroy()

        # Solution Update
        if self.converged:
            if self.counter == 1:
                self.delta_s_max = self.delta_s
                self.delta_s_min = self.delta_s / 1024.0
            
            self.delta_s_n = self.delta_s
            
            self.u_f.copy(result=self.u_f_n_1)
            self.u_f_n.copy(result=self.u_f_n_1)
            self.lmbda_n_1 = self.lmbda_n
            
            self.u_f.copy(result=self.u_f_n)
            self.lmbda_n = self.lmbda
                
            self.counter += 1 
            
            if self.converged_prev:
                self.delta_s = min(max(2*self.delta_s, self.delta_s_min), self.delta_s_max)
        else:
            if self.converged_prev:
                self.delta_s = max(self.delta_s / 2, self.delta_s_min)
            else:
                self.delta_s = max(self.delta_s / 4, self.delta_s_min)
        
        u_update.destroy()
        R_star.destroy()
        delta_u_f.destroy()