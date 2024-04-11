# Solves a steady-state porous media equation using Firedrake by
# applying Newton's method.  See latex/porous.pdf for documentation.

from firedrake import *

m = 200               # resolution
lx = 100.0
ly = 24.0

# unit properties
k1, phi1 = 6.87e-12, 0.500  # CV

# defined parameters
R = 8.314462618         # universal gas constant 
T = 293.15              # dome and gas temperature
M = 0.018015            # molar mass of steam 
c = (R * T) / M         # ratio  M/RT from ideal gas law
mu = 0.000043           # dynamic viscosity
g = -9.8                # acceleration of gravity
patm = 101000.          # atmospheric pressure
dens_r = 2700.          # overburden rock/lava density

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = RectangleMesh(m, m, lx, ly)

# define function spaces
H = FunctionSpace(mesh,'CG',1)
V = VectorFunctionSpace(mesh, "CG", 1)
w = TestFunction(H)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

# vary permeability smoothly across x and z
k = ((k1*9) * sin((2*pi/lx)*(x+z))) + k1*10

# define scalar function space for density
rho = Function(H, name='rho (density)')

# weak form
F = ( k * rho * dot(grad(c * rho + rho * g * z), grad(w)) ) * dx

# add Dirichlet BCs
# steam density at atmospheric pressure
dens = patm / c # kg/m^3
print("Steam density at surface is", dens, "kg/m^3")
# overpressure at depth
mpa = 1000000.

# steam density at depth with 1 MPa overpressure
dens1 = ((dens_r * g * ly) + mpa) / c
print("Steam density at", ly, " m deep, with an overpressure of", mpa, "Pa: ",  dens1, "kg/m^3")

BCs = DirichletBC(H, Constant(dens), (4,))  # rho = constant on top
BCs1 = DirichletBC(H, Constant(dens1), (3,))  # rho = constant on bottom

#solve
rho.assign(dens)  # initial iterate nonzero (and equals top b. c.)
solve(F == 0, rho, bcs=[BCs, BCs1],
      solver_parameters = {'snes_type': 'newtonls',
                           #'snes_linesearch_type': 'bt',
                           #'snes_view': None,
                           'snes_monitor': None,
                           'snes_converged_reason': None,
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

#calculate pressure 
p0 = Function(H, name="p (pressure)").interpolate(rho * c)

#calculate darcy flux, divide by phi?
q = Function(V, name="q (darcy flux)").interpolate(-k/(phi*mu) * grad(c * rho + rho * g * z))

VTKFile("result_CG1.pvd").write(rho, q, p0)
