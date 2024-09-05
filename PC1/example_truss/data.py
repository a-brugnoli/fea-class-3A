import numpy as np
import matplotlib.pyplot as plt
'''
Data taken from
https://people.duke.edu/~hpgavin/cee421/truss-method.pdf
'''


E_kN_to_sq_inch = 30000
A_square_inch = 10

EA = E_kN_to_sq_inch * A_square_inch

L = 12 * 16
H = 12 * 12

coordinates = np.array([[0, 0], 
                        [L, 0],
                        [L, H],
                        [2*L, 0],
                        [2*L, H]])


connectivity_table = np.array([[1, 3], 
                               [1, 2],
                               [2, 3],
                               [3, 5],
                               [3, 4],
                               [2, 5], 
                               [2, 4], 
                               [4, 5]])


connectivity_table = connectivity_table - 1



Fy_2_kN = - 100
Fx_4_kN = 50
forces = {1 : np.array([0, Fy_2_kN]), 
          4 : np.array([Fx_4_kN, 0])}

nodes_bcs = [0, 3]