# Solves a steady-state porous media equation using Firedrake by
# applying Newton's method.  See latex/porous.pdf for documentation.

from firedrake import *

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = UnitSquareMesh(10,10)

H = FunctionSpace(mesh,'CG',1)
w = TestFunction(H)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

rho = Function(H, name='rho(x,y)  density')
F = ( rho * dot(grad(rho + rho * z), grad(w)) ) * dx
#if injectleft:
#    F += FIXME * ds(1)
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

File("result.pvd").write(rho)
