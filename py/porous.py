from firedrake import *

m = 200               # resolution
lx = 200
ly = 100

injectleft = True   # case 1: False, case 2: True
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


# note: some stable triangular element choices are
#   RT_k x DG_{k-1}   for k = 1,2,3,...
#   BDM_k x DG_{k-1}  for k = 1,2,3,...
# note that 'DG'-->'CG' *does* give global conservation here

k = 1
S = FunctionSpace(mesh, 'RT', k)    # or 'BDM'
H = FunctionSpace(mesh, 'DG', k-1)
W = S * H

V = VectorFunctionSpace(mesh, "CG", 1)


w = Function(W)
sigma, rho = split(w)
omega, v = TestFunctions(W)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

## Does not converge for certain combinations of k1, k2 (e.g. k1 = 1e-11 and k2 = 1e-18)
k = conditional(z < 40., conditional(z > 20., Constant(k1), Constant(k2)), Constant(k2)) # For 3D model -- can output to file to compare

# Define scalar and vector function space for density and darcy velocity, respectively
##rho = Function(W, name='rho(x,y)  density')
q =  Function(V, name='q(x,y)  darcy velocity')


# calculate reasonable inlet gas flux if we want a Neumann BC
# q_in = lx * (2700. * -1 * g * ly * M / (R * T))


### ! attempt at mixed weak form ! ###

F = k * rho * dot((c * sigma + sigma * g * z),omega) * dx - k * rho * (c * rho + rho * g * z) * div(omega) * dx + div(sigma) * v * dx

if injectleft:
    qone = conditional(z < 0.4, conditional(z > 0.2, Constant(-3.0), Constant(0.0)), Constant(0.0))
    F += mu * rho * qone * v * ds(1)
elif injectbottom:
    qone = conditional(x < 90., conditional(x > 10., Constant(q_in), Constant(0.0)), Constant(0.0))
    F += mu * rho * qone * v * ds(3)


## Steam density at atmospheric pressure
dens = 0.6
## Overpressure at depth
mpa = 1000000.

## Steam density at 100m depth with 1 MPa overpressure
dens1 = ((2700. * g * ly) + mpa) / c
print("Steam density at", ly, " m deep, with an overpressure of", mpa, "Pa: ",  dens1, "kg/m^3")



## Add Dirichlet BCs -- how to do this for mixed methods?

# Neumann conditions on u for ids 1,2 is now Dirichlet on normal
# component of sigma = - grad(u), but we must set both components
# apparently

BCs = DirichletBC(W.sub(0), as_vector([dens,dens]), (4,))  # rho = constant on top
BCs1 = DirichletBC(W.sub(0), as_vector([dens1,dens1]), (3,))  # rho = constant on bottom

w.assign(dens)  # initial iterate nonzero (and equals top b. c.)
solve(F == 0, w, bcs=[BCs, BCs1],
      solver_parameters = {'snes_type': 'newtonls',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'snes_monitor': None,
                           'snes_converged_reason': None,
                           'pc_factor_mat_solver_type': 'mumps'})

sigma, rho = w.subfunctions
sigma.rename('sigma')
rho.rename('rho')
                              

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


### ! try different conservation calculation ! ###

# measure conservation success!
##uint = assemble(u * dx)
##fint = assemble(f * dx)
##n = FacetNormal(mesh)
##oflux = assemble(- dot(sigma,n) * ds)
##imbalance = - oflux - fint
##print(f'  u integral       = {uint:13.6e}')
##print(f'  f integral       = {fint:13.6e}')
##print(f'  flux out         = {oflux:13.6e}')
##print(f'  imbalance        = {imbalance:13.6e}')

output.VTKFile("result_inject_grav_gasLaw_rectangle.pvd").write(rho, q, p0)
