# Solves a steady-state porous media equation using Firedrake by
# applying Newton's method.  See latex/porous.pdf for documentation.

from firedrake import *
from firedrake.output import VTKFile

m = 20               # resolution
injectleft = False   # case 1: False, case 2: True

k = 1.0              # permeability
c = 1.0              # ratio  RT/M  in ideal gas law
mu = 1.0             # dynamic viscosity
g = 1.0              # acceleration of gravity

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = UnitSquareMesh(m,m)

H = FunctionSpace(mesh,'CG',1)
w = TestFunction(H)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

rho = Function(H, name='rho(x,y)  density')
F = ( k * rho * dot(grad(c * rho + rho * g * z), grad(w)) ) * dx
if injectleft:
    qone = conditional(z < 0.4, conditional(z > 0.2, Constant(-3.0), Constant(0.0)), Constant(0.0))
    F += mu * rho * qone * w * ds(1)
bdry_ids = (4,)
BCs = DirichletBC(H, Constant(1.0), bdry_ids)  # rho = 1 on top

rho.assign(1.0)  # initial iterate nonzero (and equals top b. c.)
solve(F == 0, rho, bcs=[BCs],
      solver_parameters = {'snes_type': 'newtonls',
                           #'snes_linesearch_type': 'bt',
                           #'snes_view': None,
                           'snes_monitor': None,
                           'snes_converged_reason': None,
                           'ksp_type': 'preonly',
                           'pc_type': 'lu'})

VTKFile("result.pvd").write(rho)
