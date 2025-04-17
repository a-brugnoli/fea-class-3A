import numpy as np

def apply_bcs(Matrix, bc_dofs):
    """Apply boundary conditions"""
    mask_rows = np.ones(Matrix.shape[0], dtype=bool)
    mask_rows[bc_dofs] = False

    Mat_red = Matrix[mask_rows, :][:, mask_rows]

    return Mat_red


def restore_data(reduced_data, dofs_bcs):
    """
    Restores a vector or matrix by inserting zeros at specified indices.
    ----------
    Parameters
        original_size (int): The original size (number of elements for vectors, rows for matrices).
        reduced_data (ndarray): The remaining data (1D vector or 2D matrix).
        dofs_bcs (set or list): Indices where elements/rows were removed.
    
    Returns
    ---------
        ndarray: The restored vector or matrix with zeros in removed positions.
    """
    # Determine if it's a vector (1D) or matrix (2D)
    
    if reduced_data.ndim == 1:
        original_size = len(reduced_data) + len(dofs_bcs)
        restored = np.zeros(original_size, dtype=reduced_data.dtype)
    else:
        original_size = reduced_data.shape[0] + len(dofs_bcs)
        restored = np.zeros((original_size, reduced_data.shape[1]), dtype=reduced_data.dtype)

    mask = np.ones(original_size, dtype=bool)  # Boolean mask for valid positions
    mask[list(dofs_bcs)] = False  # Mark removed positions as False
    restored[mask] = reduced_data  # Place back valid data
    return restored

# Example Usage:
if __name__=="main":
    # Restoring a Vector
    reduced_vector = np.array([1, 2, 3, 4])  
    dofs_bcs = [1, 4]  

    restored_vector = restore_data(reduced_vector, dofs_bcs)
    print("Restored Vector:\n", restored_vector)

    # Restoring a Matrix
    reduced_matrix = np.array([[1, 2], [3, 4], [5, 6]])  
    removed_indices_mat = {1, 3}  

    restored_matrix = restore_data(reduced_matrix, removed_indices_mat)
    print("Restored Matrix:\n", restored_matrix)