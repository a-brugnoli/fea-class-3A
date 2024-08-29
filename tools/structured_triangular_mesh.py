import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from src.plot_config import configure_matplotlib
configure_matplotlib()

# Parameters
n_x = 3  # Number of elements along x axis
n_y = 3  # Number of elements along y axis

# Generate grid points
x = np.linspace(0, 1, n_x + 1)
y = np.linspace(0, 1, n_y + 1)
X, Y = np.meshgrid(x, y)
points = np.vstack([X.ravel(), Y.ravel()]).T

# Create triangular mesh: the number of triangles is 2 *n_x * n_y
n_triangles = 2 * n_x * n_y
triangles = np.zeros((n_triangles, 3), dtype=int)
for i in range(n_y):
    for j in range(n_x):
        idx = i * (n_x + 1) + j
        i_triangle = 2 * (i * n_x + j)
        triangles[i_triangle, :] = [idx, idx + 1, idx + n_x + 1]
        triangles[i_triangle + 1, :] = [idx + 1, idx + n_x + 2, idx + n_x + 1]

triangles = np.array(triangles)

# Create the triangulation object
triangulation = tri.Triangulation(points[:, 0], points[:, 1], triangles)

# Identify the left boundary (x = 0)
left_boundary_indices = np.where(points[:, 0] == 0)[0]
left_boundary_edges = np.zeros((len(left_boundary_indices) - 1, 2), dtype=int)

# Create edges along the left boundary
for i in range(len(left_boundary_indices) - 1):
    left_boundary_edges[i] = [left_boundary_indices[i], left_boundary_indices[i + 1]]


# Plotting
plt.figure(figsize=(6, 6))
plt.triplot(triangulation, 'o-', lw=1)
plt.gca().set_aspect('equal')

# Highlight the left boundary
for edge in left_boundary_edges:
    plt.plot(points[edge, 0], points[edge, 1], 'r-', lw=3)

# Annotate the left boundary with Γ_D
mid_y = np.mean(points[left_boundary_indices, 1])  # Midpoint y-coordinate of the left boundary
plt.text(-0.15, mid_y, '$\Gamma_D$', fontsize=20, color='blue', verticalalignment='center')

plt.xlim((-0.2, 1.2))
plt.xlim((-0.2, 1.2))

plt.title("Triangular Structured Mesh")
plt.xlabel("$x$")
plt.ylabel("$y$")

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "structured_triangular_mesh.pdf")
plt.savefig(file_path, bbox_inches="tight")
plt.show()
