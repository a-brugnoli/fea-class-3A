import matplotlib.pyplot as plt
import numpy as np
from .plot_config import configure_matplotlib

configure_matplotlib()

def plot_truss_structure(coordinates, connectivity_table):
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]
    edge_coordinates = np.zeros((n_elements, 2, 2))

    plt.figure()
    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii]
        edge_coordinates[ii, 0, :] = coordinates[left_node, :]
        edge_coordinates[ii, 1, :] = coordinates[right_node, :]

        plt.plot(edge_coordinates[ii, :, 0], edge_coordinates[ii, :, 1], 'o-', label='Element ' + str(ii+1))
    plt.legend()
    plt.show()