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
    K : numpy.ndarray

    """

    L = np.linalg.norm(coord2 - coord1) # length of the bar

    x1, y1 = coord1
    x2, y2 = coord2
    
    c = ( x2 - x1 ) / L # cosine of bar angle
    s = ( y2 - y1 ) / L # sine of bar angle

    theta = np.arctan2(s, c)

    K = EA/L * np.array([[c**2, c*s, - c**2, - c*s],
                        [c * s, s**2,  - c*s, - s**2],
                        [-c**2, - c*s, c**2, c*s],
                        [ - c*s, - s**2, c*s, s**2 ] ] )
    return K, theta       
    