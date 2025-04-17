import matplotlib.pyplot as plt
import numpy as np
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()


def plot_truss_structure_2d(coordinates, connectivity_table, ax=None, **options):
    """
    Plot the truss structure based on the coordinates and connectivity table.
    
    Parameters  
    ----------
    coordinates : np.ndarray
        Array of shape (n_nodes, 2) containing the coordinates of the nodes.
    connectivity_table : np.ndarray
        Array of shape (n_elements, 2) containing the connectivity information.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes will be created.
    options : dict
        Additional options for plotting
    
    Returns
    -------
    matplotlib.axes.Axes
        The axes that were plotted on
    """
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]
    edge_coordinates = np.zeros((n_elements, 2, 2))
    
    # Create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get line style and color options if provided
    linestyle = options.get("linestyle", "o-")
    color = options.get("color", None)
    
    # Plot elements with their numbers
    for ii in range(n_elements):
        left_node, right_node = connectivity_table[ii]
        edge_coordinates[ii, 0, :] = coordinates[left_node, :]
        edge_coordinates[ii, 1, :] = coordinates[right_node, :]
        
        # Plot the element line
        ax.plot(edge_coordinates[ii, :, 0], edge_coordinates[ii, :, 1], linestyle, color=color)
        
        # Check if element labels should be shown
        if options.get("show_element_labels", True):
            # Calculate the midpoint of the element to place the element number
            midpoint_x = (edge_coordinates[ii, 0, 0] + edge_coordinates[ii, 1, 0]) / 2
            midpoint_y = (edge_coordinates[ii, 0, 1] + edge_coordinates[ii, 1, 1]) / 2
            
            # Add small offset to the element number label to avoid overlapping with the line
            offset_x = (edge_coordinates[ii, 1, 0] - edge_coordinates[ii, 0, 0]) * 0.05
            offset_y = (edge_coordinates[ii, 1, 1] - edge_coordinates[ii, 0, 1]) * 0.05
            
            # Add element number label
            ax.text(midpoint_x + offset_x, midpoint_y + offset_y, f'$E{ii+1}$', 
                   fontsize=9, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Plot node numbers
    if options.get("show_node_labels", True):
        for i in range(n_nodes):
            # Add node number label with small offset
            ax.text(coordinates[i, 0] + 0.05, coordinates[i, 1] + 0.05, f'$N{i+1}$', 
                    fontsize=9, ha='left', va='bottom',
                    bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=1))
    
    # Add options to plot
    if "title" in options:
        ax.set_title(options["title"])
    if "xlabel" in options:
        ax.set_xlabel(options["xlabel"])
    if "ylabel" in options:
        ax.set_ylabel(options["ylabel"])
    if "xlim" in options:
        ax.set_xlim(options["xlim"])
    if "ylim" in options:
        ax.set_ylim(options["ylim"])
    
    # Add legend if specified
    if "label" in options:
        ax.plot([], [], linestyle.replace('o', ''), color=color, label=options["label"])
        ax.legend()
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')  # Ensure the plot has equal scaling
    
    # Only show the plot if we created a new figure
    if ax.figure.canvas.figure == ax.figure:
        plt.tight_layout()
    
    return ax

# Example: Plot original and deformed truss
if __name__ == "__main__":
    # Define a simple truss structure
    coordinates_original = np.array([
        [0, 0],      # Node 0
        [4, 0],      # Node 1
        [8, 0],      # Node 2
        [2, 3],      # Node 3
        [6, 3]       # Node 4
    ])
    
    connectivity_table = np.array([
        [0, 1],      # Element 1
        [1, 2],      # Element 2
        [0, 3],      # Element 3
        [1, 3],      # Element 4
        [1, 4],      # Element 5
        [2, 4],      # Element 6
        [3, 4]       # Element 7
    ])
    
    # Simulate deformation (scale exaggerated for visualization)
    # In a real application, this would come from your analysis results
    deformation_scale = 0.3
    deformation = np.array([
        [0, 0],              # Node 0 (fixed)
        [0.02, -0.05],       # Node 1
        [0, 0],              # Node 2 (fixed)
        [0.08, -0.15],       # Node 3
        [0.05, -0.12]        # Node 4
    ]) * deformation_scale
    
    coordinates_deformed = coordinates_original + deformation
    
    # Create a figure with a single axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the original structure with solid blue lines
    ax = plot_truss_structure_2d(
        coordinates_original, 
        connectivity_table, 
        ax=ax,
        color='blue',
        linestyle='o-',
        show_element_labels=True,
        show_node_labels=True,
        label='Original',
        xlabel='X (m)',
        ylabel='Y (m)',
        title='Truss Structure: Original vs Deformed'
    )
    
    # Plot the deformed structure with dashed red lines
    # Only show element numbers (not node numbers) for clarity
    ax = plot_truss_structure_2d(
        coordinates_deformed, 
        connectivity_table, 
        ax=ax,
        color='red',
        linestyle='o--',
        show_element_labels=False,
        show_node_labels=False,
        label=f'Deformed (scale: {deformation_scale})'
    )
    
    plt.tight_layout()
    plt.show()