from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
Solves simple gas-in-porous-media problem using Darcy's law,
for discontinuous permeability.  This problem is linear in
the square of density: u = rho^2.  Implemented as a primal CG
method with order j>=1.
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-k_type', metavar='X', default='',
                    help='alternative permeability structures: flat|verif')
parser.add_argument('-mx', type=int, default=100, metavar='MX',
                    help='x resolution')
parser.add_argument('-mz', type=int, default=22, metavar='MZ',
                    help='z resolution')
parser.add_argument('-order', type=int, default=1, metavar='J',
                    help='polynomial order for density rho')
parser.add_argument('-triangles', action='store_true', default=False,
                    help='use triangular elements')
args, passthroughoptions = parser.parse_known_args()

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile

# fixed parameters
R = 8.314462618         # universal gas constant       (J kg-1 mol-1)
T = 293.15              # dome and gas temperature     (K)
M = 0.018015            # molar mass of steam          (kg mol-1)
c = (R * T) / M         # ideal gas law is  c rho = P  (J kg-1)
mu = 0.000043           # dynamic viscosity            (Pa s)

# overall dimensions of domain
lx = 100.0              # width (m)
lz = 22.0               # height (m)

## multiple geological units with discontinuous k
# note: k in m^2, phi dimensionless
zCVOB, zOBFV = 12.0, 18.0
xc = lx / 2.0
xCVOB, xOBFV = 4.0, 12.0
kCV, phiCV = 6.87e-12, 0.500
if args.k_type == 'flat':
    kOB, phiOB = kCV, phiCV
    kFV, phiFV = kCV, phiCV
else:
    kOB, phiOB = 4.94e-15, 0.0324
    kFV, phiFV = 2.18e-13, 0.232

def getunits(x, z):
    dx = abs(x - xc)  # horizontal distance from center
    ## k = permeability field, from COMSOL-generated figure
    kupper = conditional(z < zOBFV, kOB, conditional(dx < xOBFV, kOB, kFV))
    k = conditional(z < zCVOB, kCV, conditional(dx < xCVOB, kCV, kupper))
    ## phi = porosity field
    phiupper = conditional(z < zOBFV, phiOB, conditional(dx < xOBFV, phiOB, phiFV))
    phi = conditional(z < zCVOB, phiCV, conditional(dx < xCVOB, phiCV, phiupper))
    return k, phi

def getunitsverif(x, z):
    kupper = conditional(z < zOBFV, kOB, kFV)
    k = conditional(z < zCVOB, kCV, kupper)
    phiupper = conditional(z < zOBFV, phiOB, phiFV)
    phi = conditional(z < zCVOB, phiCV, phiupper)
    return k, phi

# Dirichlet boundary conditions
Patm = 101325.0    # atmospheric pressure (Pa)
u_top = (Patm / c)**2
Pbot = 1100000.0   # Pa; = 11 bar
u_bot = (Pbot / c)**2

def uverif(x, z):
    # construct piecewise-linear exact solution in case where
    # k=k(z) and u=u(z) are independent of x, using linear system
    # M c = b
    import numpy as np
    z1, z2 = zCVOB, zOBFV
    dz1, dz2 = z2 - z1, lz - z2
    M = np.array([[-z1,  1.0,  0.0,  0.0,  0.0],
                  [kCV,  0.0, -kOB,  0.0,  0.0],
                  [0.0, -1.0, -dz1,  1.0,  0.0],
                  [0.0,  0.0,  kOB,  0.0, -kFV],
                  [0.0,  0.0,  0.0,  1.0,  dz2]])
    b = np.array([u_bot, 0.0, 0.0, 0.0, u_top])
    aCV, bOB, aOB, bFV, aFV = tuple(np.linalg.solve(M, b))
    uupper = conditional(z < z2, aOB * (z - z1) + bOB, aFV * (z - z2) + bFV)
    return conditional(z < z1, aCV * z + u_bot, uupper)  # = u_exact(z)

# indices of four boundaries/sides:
#   (1, 2, 3, 4) = (left, right, bottom, top)
elements = "triangular" if args.triangles else "quadrilateral"
print(f'{elements} mesh of {args.mx} x {args.mz} elements ...')
if args.triangles:
    mesh = RectangleMesh(args.mx, args.mz, lx, lz, diagonal='crossed')
else:
    mesh = RectangleMesh(args.mx, args.mz, lx, lz, quadrilateral=True)
n = FacetNormal(mesh)

# choose function spaces
H = FunctionSpace(mesh, 'CG', args.order)
u = Function(H, name='u (rho^2)')
v = TestFunction(H)

# permeability and porosity fields
x, z = SpatialCoordinate(mesh)
if args.k_type == 'verif':
    k, phi = getunitsverif(x, z)
else:
    k, phi = getunits(x, z)

# primal CG weak form
alf = c / (2.0 * mu)
F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)
BCs = [DirichletBC(H, u_bot, 3),
       DirichletBC(H, u_top, 4)]
print(f'solving primal CG{args.order} form for u ...')

# linear solve for u
solve(F == 0, u, bcs=BCs, options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_type': 'mumps'})

# recover mass flux sigma to measure conservation
if args.triangles:
    S = FunctionSpace(mesh, 'RT', args.order + 1)
else:
    S = FunctionSpace(mesh, 'RTCF', args.order + 1)
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
dx = abs(x - xc)
CVind = conditional(dx < xCVOB, 1.0, 0.0)
OBind = conditional(dx < xOBFV, conditional(abs(x - xc) > xCVOB, 1.0, 0.0), 0.0)
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

if args.k_type == 'verif':
    uexact = Function(H).interpolate(uverif(x,z))
    err = errornorm(u, uexact, norm_type='L2') / norm(uexact, norm_type='L2')
    print(f'verification case: |u-u_exact|_2/|u|_2 = {err:6.2e}')

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
