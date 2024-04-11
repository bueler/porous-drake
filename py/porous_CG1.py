# Solves a steady-state porous media equation using Firedrake by
# applying Newton's method.  See latex/porous.pdf for documentation.

from firedrake import *

m = 200               # resolution
lx = 100.0
ly = 24.0

injectleft = False   # case 1: False, case 2: True
injectbottom = False

#k1 = 1e-11              # permeability
#k2 = 1e-15

k1, phi1 = 6.87e-12, 0.500  # CV

R = 8.314462618
T = 293.15
M = 0.018015 
c = (R * T) / M            # ratio  M/RT in ideal gas law
mu = 0.000043           # dynamic viscosity
g = -9.8             # acceleration of gravity

print("Atmospheric pressure is", 0.6*(c), " Pa")

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = RectangleMesh(m, m, lx, ly)

H = FunctionSpace(mesh,'CG',1)
V = VectorFunctionSpace(mesh, "CG", 1)
w = TestFunction(H)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

k = ((k1*9) * sin((2*pi/lx)*(x+z))) + k1*10

## Does not converge for k1 = 1e-11 and k2 = 1e-18
#k = conditional(z < 40., conditional(z > 20., Constant(k1), Constant(k2)), Constant(k2)) # For 3D model -- can output to file to compare

# Define scalar and vector function space for density and darcy velocity, respectively
rho = Function(H, name='rho (density)')
#q =  Function(V, name='q(x,y)  darcy velocity')

F = ( k * rho * dot(grad(c * rho + rho * g * z), grad(w)) ) * dx

## Add Dirichlet BCs

## Steam density at atmospheric pressure
dens = 0.6 
## Overpressure at depth
mpa = 1000000.

## Steam density at 100m depth with 1 MPa overpressure
dens1 = ((2700. * g * ly) + mpa) / c
print("Steam density at", ly, " m deep, with an overpressure of", mpa, "Pa: ",  dens1, "kg/m^3")

BCs = DirichletBC(H, Constant(dens), (4,))  # rho = constant on top
BCs1 = DirichletBC(H, Constant(dens1), (3,))  # rho = constant on bottom

rho.assign(dens)  # initial iterate nonzero (and equals top b. c.)
solve(F == 0, rho, bcs=[BCs, BCs1],
      solver_parameters = {'snes_type': 'newtonls',
                           #'snes_linesearch_type': 'bt',
                           #'snes_view': None,
                           'snes_monitor': None,
                           'snes_converged_reason': None,
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

p0 = Function(H, name="p (pressure)").interpolate(rho * c)  # or P0.project((rho * c)) 

q = Function(V, name="q (darcy flux)").interpolate(-k/mu * grad(c * rho + rho * g * z)) # or q.project(-k/mu * grad(c * rho + rho * g * z))

VTKFile("result_CG1.pvd").write(rho, q, p0)
