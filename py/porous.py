from firedrake import *

m = 200               # resolution
lx = 200
ly = 100

injectleft = False   # case 1: False, case 2: True
injectbottom = False

k1 = 1e-11              # permeability
k2 = 1e-15
R = 8.314462618
T = 293.15
M = 0.018015 
c = (R * T) / M            # ratio  M/RT in ideal gas law
mu = 0.000043           # dynamic viscosity
g = 9.8             # acceleration of gravity

print("Atmospheric pressure is", 0.6*(c), " Pa")

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)

mesh = RectangleMesh(m, m, lx, ly)


H = FunctionSpace(mesh,'CG',1)
V = VectorFunctionSpace(mesh, "CG", 1)
w = TestFunction(H)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical


## Does not converge for k1 = 1e-11 and k2 = 1e-18
k = conditional(z < 40., conditional(z > 20., Constant(k1), Constant(k2)), Constant(k2)) # For 3D model -- can output to file to compare
 
# Define scalar and vector function space for density and darcy velocity, respectively
rho = Function(H, name='rho(x,y)  density')
q =  Function(V, name='q(x,y)  darcy velocity')

F = ( k * rho * dot(grad(c * rho + rho * g * z), grad(w)) ) * dx

# calculate reasonable inlet gas flux if we want a Neumann BC
#q_in = lx * (2700. * -1 * g * ly * M / (R * T)) 

if injectleft:
    qone = conditional(z < 0.4, conditional(z > 0.2, Constant(-3.0), Constant(0.0)), Constant(0.0))
    F += mu * rho * qone * w * ds(1)
elif injectbottom:
    qone = conditional(x < 90., conditional(x > 10., Constant(q_in), Constant(0.0)), Constant(0.0))
    F += mu * rho * qone * w * ds(3)

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

# ! Double check these ! #

p0 = Function(H, name="pressure").interpolate(rho * c)  # or P0.project((rho * c)) 

q = Function(V, name="darcy velocity").interpolate(-k/mu * grad(c * rho + rho * g * z)) # or q.project(-k/mu * grad(c * rho + rho * g * z))


# boundary flux
# u*rho*ds

phi = 0.3

# ! Double check this ! What is q[1]? Darcy velocity in z-direction? If so, why do we obtain a non-zero value at left and right boundary? #

bflux_top = assemble((q[1] / phi) * rho * ds(4)) 
bflux_inlet = assemble((q[1] / phi) * rho * ds(3))
bflux_side1 = assemble((q[1] / phi) * rho * ds(1))
bflux_side2 = assemble((q[1] / phi) * rho * ds(2))

print("Boundary flux top is", bflux_top*60*60*24*0.001, "t/m/d")
print("Boundary flux inlet is", bflux_inlet*60*60*24*0.001, "t/m/d")
print("Boundary flux side 1 is", bflux_side1*60*60*24*0.001, "t/m/d")
print("Boundary flux side 2 is", bflux_side2*60*60*24*0.001, "t/m/d")

output.VTKFile("result_inject_grav_gasLaw_rectangle.pvd").write(rho, q, p0)