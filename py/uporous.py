from firedrake import *
from firedrake.output import VTKFile

m = 100              # resolution
lx = 100.0           # width
lz = 22.0            # height

## for multiple units with discontinuous k:
k1, phi1 = 6.87e-12, 0.500  # CV
if True:
    k2, phi2 = 4.94e-15, 0.0324 # OB
    k3, phi3 = 2.18e-13, 0.232  # FV
else:  # flatten
    k2, phi2 = k1, phi1
    k3, phi3 = k1, phi1

# defined parameters
R = 8.314462618         # universal gas constant 
T = 293.15              # dome and gas temperature (K)
M = 0.018015            # molar mass of steam 
c = (R * T) / M         # ratio from ideal gas law
mu = 0.000043           # dynamic viscosity
g = 9.8                 # acceleration of gravity (m s-2)
patm = 101325.0         # atmospheric pressure (Pa)

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
mesh = RectangleMesh(m, m, lx, lz, quadrilateral=True)

# choose function spaces
S = FunctionSpace(mesh, 'RTCF', 1)
H = FunctionSpace(mesh, 'DG', 0)
W = S * H
w = Function(W)
sigma, u = split(w)              # sigma = (mass flux), u = rho^2/2
omega, v = TestFunctions(W)

# permeability and porosity
x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical
## k = permeability field, guessed from COMSOL-generated(?) figure
kupper = conditional(z < 18.0, k2, conditional(abs(x - 50.0) < 12.0, k2, k3))
k = conditional(z < 12.0, k1, conditional(abs(x - 50.0) < 4.0, k1, kupper))
## phi = corresponding porosity field; conditional structure the same
phiupper = conditional(z < 18.0, phi2, conditional(abs(x - 50.0) < 12.0, phi2, phi3))
phi = conditional(z < 12.0, phi1, conditional(abs(x - 50.0) < 4.0, phi1, phiupper))

# FIXME:  P0 = patm * exp((g/c) * (lz - z)

# steam density boundary conditions
dens_top = patm / c # kg/m^3
pressure_bottom = 1100000.0   # Pa; = 11 bar
dens_bottom = pressure_bottom / c

# mixed weak form; see doc.pdf
n = FacetNormal(mesh)
G = (c + g * z) / mu
F = dot(sigma, omega) * dx \
    + 2.0 * k * u * dot(grad(G), omega) * dx \
    - u * div(k * G * omega) * dx \
    + avg(u) * jump(k * omega, n) * dS \
    + div(sigma) * v * dx \
    + (k / 2.0) * dens_bottom**2 * dot(omega, n) * ds(3) \
    + (k / 2.0) * dens_top**2 * dot(omega, n) * ds(4)

# Neumann conditions on u for ids 1,2 is now Dirichlet on normal
# component of sigma; we must set both components apparently
BCs = DirichletBC(W.sub(0), as_vector([0.0,0.0]), (1,2))

print('solving weak mixed form for sigma, u ...')
sigma, u = w.subfunctions
u.assign(dens_top**2/2)  # initial iterate nonzero (equals top b. c.)
sigma.assign(as_vector([0.0,0.0]))
solve(F == 0, w, bcs=[BCs,], options_prefix='main',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_type': 'mumps'})

print('measuring mass conservation ...')
bottomflux = assemble(dot(sigma,n) * ds(3))
topflux = assemble(dot(sigma,n) * ds(4))
imbalance = topflux + bottomflux
print(f'  mass flux out of top    = {topflux:13.6e}')
print(f'  mass flux into bottom   = {-bottomflux:13.6e}')
print(f'  imbalance               = {imbalance:13.6e}')

print('computing mass flux (kg m-2 s-1) from units ...')
# surface portion indicator functions
CVind = conditional(abs(x - 50.0) < 4.0, 1.0, 0.0)
OBind = conditional(abs(x - 50.0) < 12.0, conditional(abs(x - 50.0) > 4.0, 1.0, 0.0), 0.0)
FVind = 1.0 - CVind - OBind
# corresponding mass fluxes
CVflux = assemble(dot(CVind * sigma,n) * ds(4))
OBflux = assemble(dot(OBind * sigma,n) * ds(4))
FVflux = assemble(dot(FVind * sigma,n) * ds(4))
fluxsum = CVflux + OBflux + FVflux
assert abs(topflux - fluxsum)  < 1.0e-14 * topflux  # consistency with topflux
print(f'  CV unit flux            = {CVflux:13.6e} ({100*CVflux/topflux:5.2f} %)')
print(f'  OB unit flux            = {OBflux:13.6e} ({100*OBflux/topflux:5.2f} %)')
print(f'  FV unit flux            = {FVflux:13.6e} ({100*FVflux/topflux:5.2f} %)')

sigma, u = w.subfunctions
sigma.rename('sigma (mass flux; kg m-2 s-1)')
u.rename('u (rho^2/2)')
rho = Function(H).interpolate(sqrt(2.0 * u))
rho.rename('rho (density; kg m-3')
q = Function(S).project(sigma / rho)  # interpolate throws error re dual basis
q.rename('q (Darcy volumetric flux; m s-1)')
v = Function(S).project(q / phi)      # ditto
v.rename('v (velocity; m s-1)')

print('saving fields to result.pvd ...')
VTKFile("result.pvd").write(sigma, u, rho, q, v)
