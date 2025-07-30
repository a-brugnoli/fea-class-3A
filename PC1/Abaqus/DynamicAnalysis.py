from abaqus import *
from abaqusConstants import *
from math import ceil
import mesh
import os
import sys

# =============================================================================
# USER INPUT IMPORT
# =============================================================================
# Robust method to get script directory that works in both CAE and command line
def get_script_directory():
    try:
        # Try using inspect (works in CAE)
        import inspect
        return os.path.dirname(inspect.getfile(inspect.currentframe()))
    except:
        try:
            # Try using sys.argv[0] (works in command line)
            return os.path.dirname(os.path.realpath(sys.argv[0]))
        except:
            # Fallback to current working directory
            return os.getcwd()

# Add script directory to Python path
script_dir = get_script_directory()
sys.path.append(script_dir)

# Import user input data from TrussUserInput.py
import TrussUserInput as user_input


# =============================================================================
# MODEL CREATION
# =============================================================================
mdb.Model(name='TrussModel')
del mdb.models['Model-1']
model = mdb.models['TrussModel']
part = model.Part(name='Truss', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)


# =============================================================================
# GEOMETRY CREATION
# =============================================================================
node_list = []
for i, coords in enumerate(user_input.NODES):
    node = part.Node(coordinates=coords, label=i+1)
    node_list.append(node)

for i, element_nodes_idx in enumerate(user_input.ELEMENTS):
    element_nodes = [node_list[idx - 1] for idx in element_nodes_idx]
    part.Element(nodes=element_nodes, elemShape=LINE2, label=i+1)


# =============================================================================
# SET CREATION
# =============================================================================
if user_input.FIXED_NODES:
    part.Set(name='FixedNodes', nodes=part.nodes.sequenceFromLabels(tuple(user_input.FIXED_NODES)))

if user_input.SIMPLY_SUPPORTED_NODES_X:
    part.Set(name='SimplySupportedNodesX', nodes=part.nodes.sequenceFromLabels(tuple(user_input.SIMPLY_SUPPORTED_NODES_X)))
    
if user_input.SIMPLY_SUPPORTED_NODES_Y:
    part.Set(name='SimplySupportedNodesY', nodes=part.nodes.sequenceFromLabels(tuple(user_input.SIMPLY_SUPPORTED_NODES_Y)))
    
if user_input.LOADED_NODES:
    part.Set(name='LoadedNodes', nodes=part.nodes.sequenceFromLabels(tuple(user_input.LOADED_NODES)))


# =============================================================================
# MATERIAL PROPERTIES AND SECTIONS
# =============================================================================
mat = model.Material(name='TrussMat')
mat.Density(table=((user_input.DENSITY, ),))
mat.Elastic(table=((user_input.E, ),))

model.TrussSection(name='TrussSection', material='TrussMat', area=user_input.SECTION_AREA)

elemType = mesh.ElemType(elemCode=T2D2, elemLibrary=STANDARD)
part.setElementType(regions=(part.elements,), elemTypes=(elemType,))
part.SectionAssignment(region=(part.elements,), sectionName='TrussSection')


# =============================================================================
# ASSEMBLY
# =============================================================================
assembly = model.rootAssembly
assembly.DatumCsysByDefault(CARTESIAN)
assembly.Instance(name='TrussInstance', part=part, dependent=ON)


# =============================================================================
# ANALYSIS STEP
# =============================================================================
model.ImplicitDynamicsStep(
    name='QuasiStaticLoading', 
    previous='Initial',
    timePeriod=1,    
    maxNumInc=101, 
    timeIncrementationMethod=FIXED, 
    initialInc=0.01,
    nohaf=OFF, 
    noStop=OFF
)

model.ImplicitDynamicsStep(
    name='DynamicUnloading', 
    previous='QuasiStaticLoading', 
    timePeriod=user_input.DURATION, 
    maxNumInc=int(ceil(user_input.DURATION / user_input.INC_SIZE)) + 1, 
    timeIncrementationMethod=FIXED, 
    initialInc=user_input.INC_SIZE, 
    nohaf=OFF, 
    noStop=OFF
)


# =============================================================================
# BOUNDARY CONDITIONS AND LOADING
# =============================================================================
if user_input.FIXED_NODES:
    model.DisplacementBC(
        name='FixedSupports',
        createStepName='Initial',
        region=assembly.instances['TrussInstance'].sets['FixedNodes'],
        u1=0.0, u2=0.0
    )

if user_input.SIMPLY_SUPPORTED_NODES_X:
    model.DisplacementBC(
        name='SimplySupportedX',
        createStepName='Initial',
        region=assembly.instances['TrussInstance'].sets['SimplySupportedNodesX'],
        u1=0.0
    )
    
if user_input.SIMPLY_SUPPORTED_NODES_Y:
    model.DisplacementBC(
        name='SimplySupportedY',
        createStepName='Initial',
        region=assembly.instances['TrussInstance'].sets['SimplySupportedNodesY'],
        u2=0.0
    )
    
if user_input.LOADED_NODES:
    model.TabularAmplitude(
        name='Amp-1', 
        timeSpan=STEP, 
        smooth=SOLVER_DEFAULT, 
        data=((0.0, 0.0), (1.0, 1.0))
    )
    model.ConcentratedForce(
        name='Load-1', 
        createStepName='QuasiStaticLoading', 
        region=assembly.instances['TrussInstance'].sets['LoadedNodes'], 
        cf2=user_input.APPLIED_FORCE, 
        amplitude='Amp-1', 
        distributionType=UNIFORM, 
        field='', 
        localCsys=None
    )
    model.loads['Load-1'].deactivate('DynamicUnloading')


# =============================================================================
# JOB CREATION & EXECUTION
# =============================================================================
part.regenerate()
assembly.regenerate()
model.fieldOutputRequests['F-Output-1'].setValues(frequency=1)
job = mdb.Job(name='TrussDynamicAnalysis', model='TrussModel')

print("Model created successfully!")
print("Number of nodes: {}".format(len(user_input.NODES)))
print("Number of elements: {}".format(len(user_input.ELEMENTS)))
print("Job {} created".format(job.name))

job.submit()
print("Job {} submitted...".format(job.name))
job.waitForCompletion()
print("Job completed !")    
