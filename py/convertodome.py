import numpy as np
A = np.loadtxt('obsidianDome_filt50m.txt')
# note A.shape = (1258,3) and 37 * 34 = 1258
mx, my = 37, 34

xx = np.reshape(A[:,0],(mx,my))
xx -= xx.min()
yy = np.reshape(A[:,1],(mx,my))
yy -= yy.min()
zz = np.reshape(A[:,2],(mx,my))
# print(xx[0,:])  # ascending
# print(yy[:,0])  # descending ... why?

if False:  # optional plot of raw 3D coordinates of points
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xx,yy,zz)
    plt.show()

from firedrake import *
from firedrake.output import VTKFile

# generate a Firedrake mesh, but note that it is *not* structured!:
# the ordering of vertices is *not* a straightforward map (i,j) --> (node index k)
mesh = RectangleMesh(mx-1, my-1, xx.max(), yy.max(), name='basemesh')
#mesh = RectangleMesh(mx-1, my-1, xx.max(), yy.max(), quadrilateral=True)  # also works
x_m = mesh.coordinates.dat.data_ro[:,0]
y_m = mesh.coordinates.dat.data_ro[:,1]

# create a function for the surface elevation
V = FunctionSpace(mesh, 'CG', 1)
z = Function(V, name='z')

# use interpn() to get z value for each mesh location (x_m[i],y_m[i])
from scipy.interpolate import interpn
z.dat.data[:] = interpn((xx[0,:], yy[:,0]), zz.T, (x_m, y_m))

if False:  # optional .pvd version
    print('writing surface elevation field z(x,y) to odome.pvd ...')
    VTKFile('odome.pvd').write(z)

# inspect contents of HDF5 file with
#   $ h5dump -n odome.h5 |less
with CheckpointFile('odome.h5', 'w') as afile:
    print('writing mesh and surface elevation field z(x,y) to odome.h5 ...')
    afile.save_mesh(mesh)
    afile.save_function(z)
