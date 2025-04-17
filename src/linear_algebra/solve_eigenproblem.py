import scipy.linalg as la
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()


def solve_sparse_generalized_eigenproblem(K, M, k=10, sigma=None, which='SM'):
    """
    Solve the generalized eigenvalue problem K*x = lambda*M*x for symmetric positive
    definite matrices K (stiffness) and M (mass).
    
    Parameters:
    ----------
    K : scipy.sparse matrix
        Symmetric positive definite stiffness matrix (sparse format)
    M : scipy.sparse matrix
        Symmetric positive definite mass matrix (sparse format)
    k : int
        Number of eigenvalues/eigenvectors to compute
    sigma : float, optional
        Value to shift spectrum by for shift-invert mode
    which : str, optional
        Which eigenvalues to find ('SM' - smallest magnitude, 'LM' - largest magnitude)
    
    Returns:
    -------
    eigenvalues : ndarray
        The computed eigenvalues
    eigenvectors : ndarray
        The computed eigenvectors, one per column (normalized with respect to M)
    """
    # Verify that inputs are sparse matrices
    if not sp.issparse(K) or not sp.issparse(M):
        raise ValueError("Both K and M must be sparse matrices")
    # Make sure the matrices are in CSR format for efficiency
    K = K.tocsr()
    M = M.tocsr()
    # Ensure matrices have float64 data type (dtype=np.float64 or 'd')
    K = K.astype(np.float64)
    M = M.astype(np.float64)
    
    # Check that matrices are square and have the same dimensions
    assert K.shape == M.shape, "K and M must be square matrices with the same dimensions"
    
    # For symmetric positive definite matrices, we can use ARPACK's eigsh function
    # If sigma is provided, use shift-invert mode for better numerical stability
    if sigma is not None:
        # Solve in shift-invert mode: (K - sigma*M)^-1 * M
        # This transforms eigenvalues as: lambda_new = 1/(lambda_old - sigma)
        # First create the operator (K - sigma*M)
        K_shifted = K - sigma * M
        
        # Setup the linear solver using LU factorization
        # We could use cholesky for SPD matrices, but LU is more general
        lu = spla.splu(K_shifted.tocsc())
        
        # Define the linear operator for shift-invert mode
        def matvec(x):
            return lu.solve(M @ x)
        
        # Create the LinearOperator
        n = K.shape[0]
        linear_op = spla.LinearOperator((n, n), matvec=matvec, dtype=np.float64)
        
        # Solve the eigenvalue problem using shift-invert mode
        eigenvalues, eigenvectors = spla.eigsh(linear_op, k=k, M=M, which=which)
        
        # Convert back to original eigenvalues
        eigenvalues = 1.0 / eigenvalues + sigma
    else:
        # Direct solve without shift-invert
        eigenvalues, eigenvectors = spla.eigsh(K, M=M, k=k, which=which)
    
    # Sort eigenvalues and corresponding eigenvectors in ascending order
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    return eigenvalues, eigenvectors

# Example usage: Create a simple 1D finite element problem
def create_1d_fem_matrices(n_el):
    """
    Create stiffness and mass matrices for a 1D finite element model
    with n_el elements using linear elements.
    We assume unitary physical parameters and length of 1.


    Parameters:
    ----------
    n_el : int
        Number of elements
    
    Returns:
    -------
    K : scipy.sparse.csr_matrix
        Stiffness matrix
    M : scipy.sparse.csr_matrix
        Mass matrix
    """

    length_el = 1/n_el
    # Element stiffness and mass matrices
    k_elem = 1./length_el * np.array([[1.0, -1.0], [-1.0, 1.0]],   dtype=np.float64)
    m_elem = length_el/6 * np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float64) 
    
    # Assemble global matrices
    rows = []
    cols = []
    k_data = []
    m_data = []
    
    for i in range(n_el):
        # Global indices for element nodes
        indices = [i, i+1]
        
        # Add element contributions to global matrices
        for r in range(2):
            for c in range(2):
                rows.append(indices[r])
                cols.append(indices[c])
                k_data.append(k_elem[r, c])
                m_data.append(m_elem[r, c])
    
    # Create sparse matrices with explicit dtype
    K = sp.csr_matrix((k_data, (rows, cols)), shape=(n_el + 1, n_el + 1), dtype=np.float64)
    M = sp.csr_matrix((m_data, (rows, cols)), shape=(n_el + 1, n_el + 1), dtype=np.float64)
    
    # Apply boundary conditions: fix the left end (remove first row and column)
    K = K[1:, 1:]
    M = M[1:, 1:]
    
    return K, M

# Demonstrate with an example
if __name__ == "__main__":
    # Create a 1D finite element problem
    n_elements = 100  # number of elements
    K, M = create_1d_fem_matrices(n_elements)

    # Solve for the first 10 modes
    print("Solving generalized eigenvalue problem...")
    omega_squared, eigenvectors = solve_sparse_generalized_eigenproblem(K, M, k=4)

    omega_vec = np.sqrt(np.real(omega_squared))

    n_om = len(omega_vec) 
    omega_analytical = [np.pi * (2*n+1) / 2 for n in range(n_om)]

    for i in range(n_om):
        print(f"Numerical omega_{i+1} = {omega_vec[i]:.3f}")
        print(f"Analytical omega_{i+1} = {omega_analytical[i]:.3f}")
        error = abs(omega_vec[i] - omega_analytical[i]) / omega_analytical[i] * 100
        print(f"Error: {error:.3f}%")

    # Plot the first 4 eigenmodes
    plt.figure(figsize=(12, 8))
    x = np.linspace(0, 1, n_elements)
    for i in range(min(4, n_om)):
        plt.subplot(2, 2, i+1)
        # Add zero at the fixed boundary
        mode = np.concatenate(([0], eigenvectors[:, i]))
        plt.plot(np.linspace(0, 1, n_elements+1), mode)
        plt.title(f"Mode {i+1}, $\\omega = {omega_vec[i]:.3f}$")
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    