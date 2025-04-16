import numpy as np

def truss_2d_element(coord1, coord2, EA):
    """
    Compute the element stiffness matrix for a 2D truss bar in global coordinates
    Function translated to python from the MATLAB code in https://people.duke.edu/~hpgavin/cee421/truss-method.pdf
    
    Parameters
    ----------
    coord1 : 2*1 array
        x1, y1 coordinates of the first node
    coord2 : 2*1 array
        x2, y2 coordinates of the second node
    EA : float
        axial stiffness of the bar

    Returns
    -------
    K : numpy.ndarray of size (4, 4)
        stiffness matrix in the global reference frame
    theta : float
        angle of the bar in radians
    length : float
        length of the bar

    """

    length = np.linalg.norm(coord2 - coord1) # length of the bar

    x1, y1 = coord1
    x2, y2 = coord2
    
    c = ( x2 - x1 ) / length # cosine of bar angle
    s = ( y2 - y1 ) / length # sine of bar angle

    theta = np.arctan2(s, c)

    K = EA/length * np.array([[c**2, c*s, - c**2, - c*s],
                        [c * s, s**2,  - c*s, - s**2],
                        [-c**2, - c*s, c**2, c*s],
                        [ - c*s, - s**2, c*s, s**2 ] ] )
    return K, theta, length    
    


def beam_2d_element(coord1, coord2, EA, EI):
    """
    Compute the element stiffness matrix for a 2D beam in global coordinates
    
    Parameters
    ----------
    coord1 : 2*1 array
        x1, y1 coordinates of the first node
    coord2 : 2*1 array
        x2, y2 coordinates of the second node
    EA : float
        axial stiffness of the bar
    EI : float
        bending stiffness of the beam

    Returns
    -------
    K : numpy.ndarray of size (6, 6)
        stiffness matrix in the global reference frame
    theta : float
        angle of the bar in radians
    length : float
        length of the bar
    """

    L = np.linalg.norm(coord2 - coord1) # length of the bar

    x1, y1 = coord1
    x2, y2 = coord2
    
    c = ( x2 - x1 ) / L # cosine of bar angle
    s = ( y2 - y1 ) / L # sine of bar angle

    theta = np.arctan2(s, c)

    K_local= np.array([[EA/L, 0, 0, -EA/L, 0, 0],
                        [0, 12*EI/L**3, 6*EI/L**2, 0, -12*EI/L**3, 6*EI/L**2],
                        [0, 6*EI/L**2, 4*EI/L, 0, -6*EI/L**2, 2*EI/L],
                        [-EA/L , 0, 0, EA/L, 0, 0],
                        [0, -12 * EI/L**3, -6*EI/L**2, 0, 12*EI/L**3, -6*EI/L**2],
                        [0, 6 * EI/L**2, 2* EI/L, 0, -6 * EI/L**2, 4 * EI/L]] )
    
    T_gl_to_loc = np.array([[c, s, 0, 0, 0, 0],
                            [-s, c, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0],
                            [0, 0, 0, c, s, 0],
                            [0, 0, 0, -s, c, 0],
                            [0, 0, 0, 0, 0, 1]])
    
    K = T_gl_to_loc.T @ K_local @ T_gl_to_loc

    return K, theta, L      