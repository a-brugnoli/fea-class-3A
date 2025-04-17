import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from src.postprocessing.plot_config import configure_matplotlib
configure_matplotlib()


def animate_truss_structure(coordinates, connectivity_table, **options):
    """
    Create an animation of a truss structure changing over time.
    
    Parameters  
    ----------
    coordinates : np.ndarray
        Array of shape (n_nodes, 2, n_t) containing the coordinates of the nodes
        over time, where:
        - n_nodes: number of nodes
        - 2: x and y coordinates
        - n_t: number of time instants
    connectivity_table : np.ndarray
        Array of shape (n_elements, 2) containing the connectivity information.
    options : dict
        Additional options for plotting and animation
        
    Returns
    -------
    matplotlib.animation.Animation
        The animation object
    """
    n_nodes = coordinates.shape[0]
    n_elements = connectivity_table.shape[0]
    n_frames = coordinates.shape[2]
    
    # Get animation options
    interval = options.get("interval", 100)  # Time between frames in ms
    fps = options.get("fps", 10)  # Frames per second for saving video
    repeat = options.get("repeat", True)  # Whether to loop the animation
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=options.get("figsize", (10, 8)))
    
    # Initialize empty plot objects
    element_lines = []
    node_markers = []
    element_labels = []
    node_labels = []
    
    # Initialize the plots for each element
    for i in range(n_elements):
        line, = ax.plot([], [], 'o-', lw=2, color=options.get("element_color", 'blue'))
        element_lines.append(line)
        
        # Initialize element labels if requested
        if options.get("show_element_labels", True):
            label = ax.text(0, 0, f'E{i+1}', fontsize=9, ha='center', va='center',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                           visible=False)
            element_labels.append(label)
    
    # Initialize node labels if requested
    if options.get("show_node_labels", True):
        for i in range(n_nodes):
            label = ax.text(0, 0, f'N{i}', fontsize=9, ha='left', va='bottom',
                          bbox=dict(facecolor='yellow', alpha=0.7, edgecolor='none', pad=1),
                          visible=False)
            node_labels.append(label)
    
    # Set up plot styles
    if "title" in options:
        ax.set_title(options["title"])
    if "xlabel" in options:
        ax.set_xlabel(options["xlabel"])
    if "ylabel" in options:
        ax.set_ylabel(options["ylabel"])
        
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    
    # Find the range of coordinates to set fixed axis limits
    all_x = coordinates[:, 0, :].flatten()
    all_y = coordinates[:, 1, :].flatten()
    
    # Add some padding to the limits
    padding = 0.1 * max(np.ptp(all_x), np.ptp(all_y))
    
    # Set axis limits
    if "xlim" in options:
        ax.set_xlim(options["xlim"])
    else:
        ax.set_xlim(np.min(all_x) - padding, np.max(all_x) + padding)
        
    if "ylim" in options:
        ax.set_ylim(options["ylim"])
    else:
        ax.set_ylim(np.min(all_y) - padding, np.max(all_y) + padding)
    
    # Add a time indicator if requested
    time_indicator = None
    if options.get("show_time", True):
        time_text = options.get("time_format", "t = {:.2f}")
        time_indicator = ax.text(0.02, 0.98, "", transform=ax.transAxes, 
                               fontsize=12, ha='left', va='top',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', pad=5))
    
    # Time values for animation
    time_values = options.get("time_values", np.linspace(0, 1, n_frames))
    
    def init():
        """Initialize the animation with empty plots"""
        for line in element_lines:
            line.set_data([], [])
        
        for label in element_labels + node_labels:
            label.set_visible(False)
            
        if time_indicator:
            time_indicator.set_text("")
            
        return element_lines + element_labels + node_labels + ([time_indicator] if time_indicator else [])
    
    def animate(frame):
        """Update the truss for a specific frame"""
        # Extract current coordinates for this frame
        current_coords = coordinates[:, :, frame]
        
        # Update each element
        for i, line in enumerate(element_lines):
            node1, node2 = connectivity_table[i]
            x_vals = [current_coords[node1, 0], current_coords[node2, 0]]
            y_vals = [current_coords[node1, 1], current_coords[node2, 1]]
            line.set_data(x_vals, y_vals)
            
            # Update element label position
            if options.get("show_element_labels", True):
                midpoint_x = (x_vals[0] + x_vals[1]) / 2
                midpoint_y = (y_vals[0] + y_vals[1]) / 2
                
                # Add small offset
                offset_x = (x_vals[1] - x_vals[0]) * 0.05
                offset_y = (y_vals[1] - y_vals[0]) * 0.05
                
                element_labels[i].set_position((midpoint_x + offset_x, midpoint_y + offset_y))
                element_labels[i].set_visible(True)
        
        # Update node labels
        if options.get("show_node_labels", True):
            for i in range(n_nodes):
                node_labels[i].set_position((current_coords[i, 0] + 0.05, current_coords[i, 1] + 0.05))
                node_labels[i].set_visible(True)
        
        # Update time indicator
        if time_indicator and frame < len(time_values):
            time_indicator.set_text(time_text.format(time_values[frame]))
        
        return element_lines + element_labels + node_labels + ([time_indicator] if time_indicator else [])
    
    # Create the animation
    anim = FuncAnimation(fig, animate, frames=n_frames, init_func=init, 
                        interval=interval, blit=True, repeat=repeat)
    
    # Save animation if requested
    if "save_path" in options:
        anim.save(options["save_path"], writer='ffmpeg')
    
    plt.tight_layout()
    plt.close() if options.get("close_fig", False) else None
    
    return anim

# Example usage
if __name__ == "__main__":
    # Number of nodes, time steps
    n_nodes = 5
    n_time_steps = 50
    
    # Define a simple truss structure
    coordinates_initial = np.array([
        [0, 0],      # Node 0
        [4, 0],      # Node 1
        [8, 0],      # Node 2
        [2, 3],      # Node 3
        [6, 3]       # Node 4
    ])
    
    # Create a tensor for coordinates over time (n_nodes, 2, n_time_steps)
    coordinates = np.zeros((n_nodes, 2, n_time_steps))
    
    # Define the deformation (exaggerated for visualization)
    max_deformation = np.array([
        [0, 0],              # Node 0 (fixed)
        [0.05, -0.2],        # Node 1
        [0, 0],              # Node 2 (fixed)
        [0.3, -0.5],         # Node 3 (large vertical displacement)
        [0.2, -0.4]          # Node 4
    ])
    
    # Create coordinates for each time step with smooth transition
    for t in range(n_time_steps):
        # Use a sine function for smooth animation, starting and ending at original position
        factor = np.sin(np.pi * t / (n_time_steps - 1)) if t < n_time_steps - 1 else 0
        deformation = max_deformation * factor
        coordinates[:, :, t] = coordinates_initial + deformation
    
    # Connectivity table doesn't change over time
    connectivity_table = np.array([
        [0, 1],      # Element 1
        [1, 2],      # Element 2
        [0, 3],      # Element 3
        [1, 3],      # Element 4
        [1, 4],      # Element 5
        [2, 4],      # Element 6
        [3, 4]       # Element 7
    ])
    
    # Create the animation
    animation = animate_truss_structure(
        coordinates, 
        connectivity_table,
        title="Truss Structure Deformation Animation",
        xlabel="X (m)",
        ylabel="Y (m)",
        interval=50,  # milliseconds between frames
        show_element_labels=True,
        show_node_labels=True,
        show_time=True,
        time_values=np.linspace(0, 1, n_time_steps),
        time_format="t = {:.2f} s",
        # save_path="truss_animation.mp4"  # Uncomment to save animation
    )
    
    # Display animation in Jupyter notebook
    # HTML(animation.to_jshtml())
    
    # For non-notebook environments
    plt.show()