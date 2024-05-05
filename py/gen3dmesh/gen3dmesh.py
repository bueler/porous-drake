from firedrake import *
from firedrake.output import VTKFile

# read 2D base mesh
base_mesh = Mesh('pentagon.msh')
neumann_ind = 31  # would be needed for identifying Neumann (zero flux) boundary

# create 3D mesh with (temporary) height 1.0
mz = 22
mesh = ExtrudedMesh(base_mesh, layers=mz, layer_height=1.0/mz)
# see https://www.firedrakeproject.org/extruded-meshes.html#solving-equations-on-extruded-meshes regarding indexing boundary parts of extruded meshes

# compute a notional height function (for test purposes) on the base mesh
xb, yb = SpatialCoordinate(base_mesh)
Vbase = FunctionSpace(base_mesh,'CG',1)
height_base = Function(Vbase).interpolate(1.0 + 0.1 * (2.0 - xb) * xb + 0.2 * cos(3.0*yb))

# extend height defined on the base mesh to the extruded mesh
VR = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
height = Function(VR)
height.dat.data[:] = height_base.dat.data_ro[:]
x, y, z = SpatialCoordinate(mesh)
Vcoord = mesh.coordinates.function_space()
XYZ = Function(Vcoord).interpolate(as_vector([x, y, height * z]))
mesh.coordinates.assign(XYZ)

# generate notional field (for test purposes) on the 3D mesh, and output it
V = FunctionSpace(mesh, 'CG', 2)
u = Function(V, name='u(x,y)').interpolate(x**2 + sin(8.0*y) + z)
VTKFile("result.pvd").write(u)
