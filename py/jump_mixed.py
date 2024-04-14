# Solve llinear Poisson equation with discontinuous diffusivity
#   - Div (D(x,y) grad(u)) = 1
# with boundary values
#   u = 0                 left, right (x=0,1)
#   dot(grad(u),n) = 0    bottom, top (y=0,1)
# We convert it to a first order system:
#   sigma      + D(x,y) grad(u) = 0
#   Div(sigma)                  = 1
# which is then turned into a mixed weak form with a jump term
# to capture the discontinuity of D(x,y).  Uses stable elements
# RT1 x DG0.  Measures convergence against exact solution and
# saves highest-res result.

####  Regarding the jump term in the weak form  ####
# The jump term can be
#             + avg(u) * jump(D * omega, n) * dS
#   or:       + dot(avg(u * omega), jump(D, n)) * dS
#   or:       + dot(avg(u) * avg(omega), jump(D, n)) * dS
# The first might be preferred for speed of evaluation.
# However, combinations with "jump(u)" are unstable, such as
#   NO:       + dot(avg(omega), jump(D * u, n)) * dS
# The difference seems to be that "avg(u)" has only positive
# coefficients on u values while "jump(u)" has some negative
# and some positive.  Other combinations of "avg" and "jump"
# are actually inconsistent, though apparently stable, such as
#   NO:        + avg(u) * jump(omega, n) * jump(D) * dS
# Yet other combinations are not allowed because they generate
# "ValueError: Discontinuous type XX must be restricted"

from firedrake import *
from firedrake.output import VTKFile

# even m: discontinuity of D(x,y) aligned to element boundary

for m in [10, 20, 40, 80, 160]:
#for m in [11, 21, 41, 81, 161]:
      mesh = UnitSquareMesh(m, m)
      #mesh = UnitSquareMesh(m, m, quadrilateral=True)
      x, y = SpatialCoordinate(mesh)
      n = FacetNormal(mesh)

      # choose function spaces
      k = 1
      S = FunctionSpace(mesh, 'RT', k)    # or 'BDM'
      #for quads:  S = FunctionSpace(mesh, 'RTCF', k)   # or 'BDMCF'
      H = FunctionSpace(mesh, 'DG', k-1)
      W = S * H
      w = Function(W)
      sigma, u = split(w)
      omega, v = TestFunctions(W)

      # diffusivity D(x,y)
      D1, D2 = 1.0, 5.0
      D = Function(H).interpolate(conditional(x < 0.5, D1, D2))

      # weak form
      F = dot(sigma, omega) * dx - u * div(D * omega) * dx \
            + div(sigma) * v * dx - v * dx \
            + avg(u) * jump(D * omega, n) * dS   # see comment re THIS term

      # Neumann conditions on u are Dirichlet on dot(sigma, n)
      # (3,4) = (bottom,top)
      BCs = DirichletBC(W.sub(0), as_vector([0.0,0.0]), (3,4))

      solve(F == 0, w, bcs=[BCs,], options_prefix='s',
            solver_parameters = {'snes_type': 'ksponly',
                              'ksp_type': 'preonly',
                              'pc_type': 'lu',
                              'pc_factor_mat_solver_type': 'mumps'})

      sigma, u = w.subfunctions
      sigma.rename('sigma(x,y)')
      u.rename('u(x,y)')

      # exact solution and numerical error
      alpha1 = -1.0 / (2.0 * D1)
      alpha2 = -1.0 / (2.0 * D2)
      beta1 = (D1 - D2) / (4.0 * D1 * (D1 + D2))
      beta2 = D1 * beta1 / D2
      gamma = 1.0 / (4.0 * (D1 + D2))
      s = x - 0.5
      uex = conditional(x < 0.5, (alpha1 * s + beta1) * s + gamma,
                                 (alpha2 * s + beta2) * s + gamma)
      uexact = Function(H).interpolate(uex)
      uexact.rename('uexact(x,y)')
      udiff = Function(H).interpolate(u - uexact)
      udiff.rename('u - uexact')
      print(f'{m:3d} x {m:3d} mesh:  |u - u_exact|_2 = {norm(udiff):.3e}')

print(f'saving {m:3d} x {m:3d} result to "result.pvd" ...')
VTKFile("result.pvd").write(sigma, u, uexact, udiff)
