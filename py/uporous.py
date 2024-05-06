from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solves simple gas-in-porous-media problem using Darcy's law,
for discontinuous permeability.  This problem is linear in
the square of density: u = rho^2.  Implemented as a primal CG_j
method (j>=1).
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-flattenk', action='store_true', default=False,
                    help='flatten materials to k=k1 and phi=phi1')
parser.add_argument('-mx', type=int, default=100, metavar='MX',
                    help='x resolution')
parser.add_argument('-mz', type=int, default=22, metavar='MZ',
                    help='z resolution')
parser.add_argument('-order', type=int, default=1, metavar='J',
                    help='polynomial order for density rho')
parser.add_argument('-quad', action='store_true', default=False,
                    help='use quadrilateral elements')
args, passthroughoptions = parser.parse_known_args()

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile

# physical dimensions
lx = 100.0           # width (m)
lz = 22.0            # height (m)

## multiple units with discontinuous k
k1, phi1 = 6.87e-12, 0.500  # CV
if args.flattenk:
    k2, phi2 = k1, phi1
    k3, phi3 = k1, phi1
else:
    k2, phi2 = 4.94e-15, 0.0324 # OB
    k3, phi3 = 2.18e-13, 0.232  # FV
# note: k in m^2, phi dimensionless

# fixed parameters
R = 8.314462618         # universal gas constant       (J kg-1 mol-1)
T = 293.15              # dome and gas temperature     (K)
M = 0.018015            # molar mass of steam          (kg mol-1)
c = (R * T) / M         # ideal gas law is  c rho = P  (J kg-1)
mu = 0.000043           # dynamic viscosity            (Pa s)
Patm = 101325.0         # atmospheric pressure         (Pa)

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
print(f'{"quadrilateral" if args.quad else "triangular"} mesh of {args.mx} x {args.mz} elements ...')
if args.quad:
    mesh = RectangleMesh(args.mx, args.mz, lx, lz, quadrilateral=True)
else:
    mesh = RectangleMesh(args.mx, args.mz, lx, lz, diagonal='crossed')
n = FacetNormal(mesh)

# choose function spaces
H = FunctionSpace(mesh, 'CG', args.order)
u = Function(H, name='u (rho^2)')
v = TestFunction(H)

# permeability and porosity
x, z = SpatialCoordinate(mesh)   # x horizontal, z vertical
## k = permeability field, guessed from COMSOL-generated(?) figure
kupper = conditional(z < 18.0, k2, conditional(abs(x - 50.0) < 12.0, k2, k3))
k = conditional(z < 12.0, k1, conditional(abs(x - 50.0) < 4.0, k1, kupper))
## phi = corresponding porosity field; conditional structure the same
phiupper = conditional(z < 18.0, phi2, conditional(abs(x - 50.0) < 12.0, phi2, phi3))
phi = conditional(z < 12.0, phi1, conditional(abs(x - 50.0) < 4.0, phi1, phiupper))

# Dirichlet boundary conditions
u_top = (Patm / c)**2
Pbottom = 1100000.0   # Pa; = 11 bar
u_bottom = (Pbottom / c)**2

# primal CG weak form; see doc.pdf
alf = c / (2.0 * mu)
F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)
BCs = [DirichletBC(H, u_bottom, 3),
       DirichletBC(H, u_top, 4)]
print(f'solving primal CG{args.order} form for u ...')

# linear solve in fact
solve(F == 0, u, bcs=BCs, options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_type': 'mumps'})

# recover mass flux sigma to measure conservation
if args.quad:
    S = FunctionSpace(mesh, 'RTCF', args.order + 1)
else:
    S = FunctionSpace(mesh, 'RT', args.order + 1)
sigma = Function(S).project(- alf * k * grad(u))
sigma.rename('sigma (mass flux; kg m-2 s-1)')

# warn if negative u
uneg = Function(H).interpolate((-u + abs(u))/2.0)
uneg.rename('u_- (negative part of u)')
if norm(uneg, 'L2') > 0.0:
    print('WARNING: nonzero negative part of u(x,z) detected')
    print('         see "uneg" field in output file')
upos = Function(H).interpolate((u + abs(u))/2.0)

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

# recover physical scalars and vectors
rho = Function(H).interpolate(sqrt(abs(u)))
rho.rename('rho (density; kg m-3)')
P = Function(H).interpolate(c * rho)
P.rename('p (pressure; Pa)')
q = Function(S).project(sigma / rho)  # interpolate throws error re dual basis
q.rename('q (Darcy volumetric flux; m s-1)')
v = Function(S).project(q / phi)      # ditto
v.rename('v (velocity; m s-1)')

print('saving fields to result.pvd ...')
VTKFile("result.pvd").write(sigma, u, uneg, rho, P, q, v)
