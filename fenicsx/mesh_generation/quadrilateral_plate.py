import gmsh
import sys, os

pwd = os.getcwd()
mesh_folder = pwd + "/fenicsx/meshes/"
if not os.path.exists(mesh_folder):
    os.makedirs(mesh_folder)
mesh_name = "plate_quadrilateral_mesh"

# Initialize GMSH
gmsh.initialize()

# Create a new model
gmsh.model.add(mesh_name)

# Define rectangle dimensions
width = 2.0
height = 1.0

# Create rectangle geometry
# Add points (x, y, z, mesh_size)
p1 = gmsh.model.geo.addPoint(0, 0, 0, 0.1)
p2 = gmsh.model.geo.addPoint(width, 0, 0, 0.1)
p3 = gmsh.model.geo.addPoint(width, height, 0, 0.1)
p4 = gmsh.model.geo.addPoint(0, height, 0, 0.1)

# Create lines connecting the points
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p2, p3)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

# Create a curve loop and surface
curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
surface = gmsh.model.geo.addPlaneSurface([curve_loop])

# Synchronize the CAD model
gmsh.model.geo.synchronize()

# Set meshing options for quadrilateral mesh
# Recombine triangles into quadrilaterals
gmsh.model.mesh.setRecombine(2, surface)

# Set the meshing algorithm (6 = Frontal-Delaunay for Quads)
gmsh.option.setNumber("Mesh.Algorithm", 6)

# Optional: Set transfinite meshing for structured quad mesh
# Uncomment the following lines for a structured mesh:
# gmsh.model.mesh.setTransfiniteCurve(l1, 21)  # 20 divisions
# gmsh.model.mesh.setTransfiniteCurve(l2, 11)  # 10 divisions
# gmsh.model.mesh.setTransfiniteCurve(l3, 21)  # 20 divisions
# gmsh.model.mesh.setTransfiniteCurve(l4, 11)  # 10 divisions
# gmsh.model.mesh.setTransfiniteSurface(surface)

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Save the mesh
gmsh.write(mesh_folder + mesh_name + ".msh")

# Optional: Launch the GUI to visualize the mesh
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

# Finalize GMSH
gmsh.finalize()

print("Quadrilateral mesh generated successfully!")
print("Mesh saved to: rectangle_quad_mesh.msh")