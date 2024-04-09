from firedrake import *
from firedrake.output import VTKFile

m = 200              # resolution
lx = 100.0
ly = 24.0

k1 = 6.87e-12  # CV
k2 = 4.94e-15  # OB
k3 = 2.18e-13  # FV

R = 8.314462618
T = 293.15
M = 0.018015
c = (R * T) / M            # ratio  M/RT in ideal gas law
mu = 0.000043           # dynamic viscosity
g = 9.8             # acceleration of gravity

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = RectangleMesh(m, m, lx, ly, quadrilateral=True)

k = 1
S = FunctionSpace(mesh, 'RTCF', k)    # or 'BDMCF' (on quadrilaterals)
H = FunctionSpace(mesh, 'DG', k-1)
W = S * H

w = Function(W)
sigma, rho = split(w)         # sigma is mass flux: sigma = rho * q
omega, v = TestFunctions(W)

x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical

# k field, guessed from COMSOL-generated(?) figure
kupper = conditional(z < 18.0, k2, conditional(abs(x - 50.0) < 12.0, k2, k3))
k = conditional(z < 12.0, k1, conditional(abs(x - 50.0) < 4.0, k1, kupper))

## Steam density at atmospheric pressure
dens = 0.6 # kg/m^3
print("Steam density at surface is", dens, "kg/m^3")
## Overpressure at depth
mpa = 1000000.
## Steam density at 100m depth with 1 MPa overpressure
dens1 = ((2700. * g * ly) + mpa) / c
print("Steam density at", ly, " m deep, with an overpressure of", mpa, "Pa: ",  dens1, "kg/m^3")

# mixed weak form
Phead = c * rho + rho * g * z
Phead_top = c * dens + dens * g * z
Phead_bottom = c * dens1 + dens1 * g * z
n = FacetNormal(mesh)
F = dot(sigma, omega) * dx - (1.0/mu) * Phead * div(k * rho * omega) * dx \
    + (1.0/mu) * k * dens1 * Phead_bottom * dot(omega, n) * ds(3) \
    + (1.0/mu) * k * dens * Phead_top * dot(omega, n) * ds(4) \
    + div(sigma) * v * dx

# Neumann conditions on u for ids 1,2 is now Dirichlet on normal
# component of sigma = - grad(u), but we must set both components
# apparently
BCs = DirichletBC(W.sub(0), as_vector([0.0,0.0]), (1,2))

print('main solve ...')
w.assign(dens)  # initial iterate nonzero (equals top b. c.)
solve(F == 0, w, bcs=[BCs,],
      solver_parameters = {'snes_type': 'newtonls',
                           'snes_linesearch_type': 'bt',
                           'snes_rtol': 1.0e-5,
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'snes_monitor': None,
                           'snes_converged_reason': None,
                           'pc_factor_mat_solver_type': 'mumps'})

sigma, rho = w.subfunctions
sigma.rename('sigma = rho q')
rho.rename('rho (density)')

print('solve to extract q ...')
q = Function(S, name="q (darcy velocity)")
tau = TestFunction(S)
Fextractq = (rho * dot(q, tau) - dot(sigma, tau)) * dx
solve(Fextractq == 0, q,
      solver_parameters = {'ksp_converged_reason': None,
                           'snes_converged_reason': None})

p = Function(H, name="pressure").interpolate(rho * c)
kout = Function(H, name="permeability").interpolate(k)

# measure conservation here?

VTKFile("result.pvd").write(sigma, rho, q, p, kout)
