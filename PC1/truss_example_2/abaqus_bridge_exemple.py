"""
Abaqus Script - 2D Truss Analysis
"""

from abaqus import *
from abaqusConstants import *
import mesh

# =============================================================================
# INPUT DATA
# =============================================================================

# Node coordinates (x, y, z)
NODES = [
    (0, 0, 0),      # 1 - Left support
    (5, 0, 0),      # 2
    (5, 3.33, 0),   # 3
    (10, 0, 0),     # 4
    (10, 5.33, 0),  # 5
    (15, 0, 0),     # 6
    (15, 6.0, 0),   # 7 - Load point
    (20, 0, 0),     # 8
    (20, 5.33, 0),  # 9
    (25, 0, 0),     # 10
    (25, 3.33, 0),  # 11
    (30, 0, 0)      # 12 - Right support
]

# Element connectivity (node indices)
ELEMENTS = [
    [1, 2], [1, 3],
    [2, 3], [2, 4], [2, 5],
    [3, 4], [3, 5],
    [4, 5], [4, 6], [4, 7],
    [5, 6], [5, 7],
    [6, 7], [6, 8], [6, 9],
    [7, 8], [7, 9],
    [8, 9], [8, 10], [8, 11],
    [9, 10], [9, 11],
    [10, 11], [10, 12],
    [11, 12]
]

# Boundary conditions
FIXED_NODES = [1,12]    # Fixed nodes
SIMPLY_SUPPORTED_NODES = []    # Simply supported nodes
LOADED_NODES = [7]       # Loaded nodes

# Material properties
STEEL_E = 200000.0       # Young's modulus
STEEL_NU = 0.3           # Poisson's ratio
SECTION_AREA = 1.0       # Cross-sectional area

# Loading
APPLIED_FORCE = -1000.0  # Applied force

# =============================================================================
# MODEL CREATION
# =============================================================================

# Create model
model = mdb.models['Model-1']

# Create 2D deformable part
part = model.Part(
    name='Bridge',
    dimensionality=TWO_D_PLANAR,
    type=DEFORMABLE_BODY
)

# =============================================================================
# GEOMETRY CREATION
# =============================================================================

# Create nodes
node_list = []
for i, coords in enumerate(NODES):
    node = part.Node(coordinates=coords, label=i+1)
    node_list.append(node)

# Create elements
for i, element_nodes_idx in enumerate(ELEMENTS):
    # Convert indices (1-based to 0-based)
    element_nodes = [node_list[idx - 1] for idx in element_nodes_idx]
    part.Element(nodes=element_nodes, elemShape=LINE2, label=i+1)

# Create node sets
part.Set(name='FixedNodes', nodes=part.nodes.sequenceFromLabels(tuple(FIXED_NODES)))
if SIMPLY_SUPPORTED_NODES:
    part.Set(name='SimplySupportedNodes', nodes=part.nodes.sequenceFromLabels(tuple(SIMPLY_SUPPORTED_NODES)))
part.Set(name='LoadedNodes', nodes=part.nodes.sequenceFromLabels(tuple(LOADED_NODES)))

# =============================================================================
# MATERIAL PROPERTIES AND SECTIONS
# =============================================================================

# Define material
model.Material(name='Steel').Elastic(table=((STEEL_E, STEEL_NU),))

# Define section
model.TrussSection(name='TrussSection', material='Steel', area=SECTION_AREA)

# Assign element type
elemType = mesh.ElemType(elemCode=T2D2, elemLibrary=STANDARD)
part.setElementType(regions=(part.elements,), elemTypes=(elemType,))

# Assign section
part.SectionAssignment(region=(part.elements,), sectionName='TrussSection')

# =============================================================================
# ASSEMBLY
# =============================================================================

# Create assembly
assembly = model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)

# Create instance
assembly.Instance(name='BridgeInstance', part=part, dependent=ON)

# =============================================================================
# ANALYSIS STEP
# =============================================================================

# Create static analysis step
model.StaticStep(name='StaticAnalysis', previous='Initial')

# =============================================================================
# BOUNDARY CONDITIONS AND LOADING
# =============================================================================

# Apply boundary conditions (fixed supports)
model.DisplacementBC(
    name='FixedSupports',
    createStepName='Initial',
    region=assembly.instances['BridgeInstance'].sets['FixedNodes'],
    u1=0.0, u2=0.0
)

# Apply boundary conditions (simply supported)
if SIMPLY_SUPPORTED_NODES:
    model.DisplacementBC(
        name='SimplySupported',
        createStepName='Initial',
        region=assembly.instances['BridgeInstance'].sets['SimplySupportedNodes'],
        u2=0.0
    )

# Apply load
model.ConcentratedForce(
    name='AppliedLoad',
    createStepName='StaticAnalysis',
    region=assembly.instances['BridgeInstance'].sets['LoadedNodes'],
    cf2=APPLIED_FORCE
)

# =============================================================================
# JOB CREATION
# =============================================================================

# Regenerate objects
part.regenerate()
assembly.regenerate()

# Create job
job = mdb.Job(name='TrussAnalysis', model='Model-1')

print("Model created successfully!")
print("Number of nodes: {}".format(len(NODES)))
print("Number of elements: {}".format(len(ELEMENTS)))
print("Job created: {}".format(job.name))

# =============================================================================
# JOB EXECUTION
# =============================================================================
job.submit()
print("Job submitted...")
job.waitForCompletion()
print("Job completed !")