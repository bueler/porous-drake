from argparse import ArgumentParser, RawTextHelpFormatter
parser = ArgumentParser(description="""
  Solves gas flow in porous media problems using Darcy's law,
for discontinuous permeability.  This problem is linear in
the square of density: u = rho^2.  Implemented as a primal CG
method with order j>=1.
  The problem can be a 2D (x,z) test case, or the problem can
extrude to a 3D mesh generated by reading a 2D base mesh from
a file.  Note that options -k_type, -mx, -triangles are
ignored in the extrude case.
""", formatter_class=RawTextHelpFormatter)
parser.add_argument('-basemesh', metavar='FILE', default='',
                    help='file name for 2d base mesh (in problem type extrude)')
parser.add_argument('-k_type', metavar='X', default='synth', choices=['synth','flat','verif'],
                    help='permeability structures: synth|flat|verif')
parser.add_argument('-mx', type=int, default=100, metavar='MX',
                    help='x resolution (for 2d cases)')
parser.add_argument('-mz', type=int, default=22, metavar='MZ',
                    help='z resolution')
parser.add_argument('-order', type=int, default=1, metavar='J',
                    help='polynomial order for density rho')
parser.add_argument('-problem', metavar='X', default='2d', choices=['2d', 'extrude'],
                    help='problem type: 2d|extrude')
parser.add_argument('-triangles', action='store_true', default=False,
                    help='use triangular elements')
args, passthroughoptions = parser.parse_known_args()

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile
from physical import R, T, M, mu, Patm
from cases2d import getmesh2d, getgeounits2d, getgeounitsverif2d, \
                    getdirbcs2d, getuverif2d, printfluxes2d

if args.problem == '2d':
    elements = 'triangular' if args.triangles else 'quadrilateral'
    print(f'generating 2d {elements} mesh of {args.mx} x {args.mz} elements ...')
    mesh = getmesh2d(args.mx, args.mz, quad=not args.triangles)
else:
    assert NotImplementedError # FIXME extrude

# choose function spaces
H = FunctionSpace(mesh, 'CG', args.order)
u = Function(H, name='u (rho^2)')
v = TestFunction(H)

# permeability and porosity fields
if args.problem == '2d':
    if args.k_type == 'verif':
        k, phi = getgeounitsverif2d(mesh)
    else:
        k, phi = getgeounits2d(mesh, flat=(args.k_type == 'flat'))
else:
    assert NotImplementedError # FIXME extrude

# primal CG weak form
c = (R * T) / M   # ideal gas law is  c rho = P  (J kg-1)
alf = c / (2.0 * mu)
F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)

# linear solve for u
print(f'solving primal CG{args.order} weak form for u ...')
solve(F == 0, u, bcs=getdirbcs2d(mesh, H), options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_type': 'mumps'})

# recover mass flux sigma to measure conservation
if args.problem == '2d':
    if args.triangles:
        S = FunctionSpace(mesh, 'RT', args.order + 1)
    else:
        S = FunctionSpace(mesh, 'RTCF', args.order + 1)
else:
    assert NotImplementedError # FIXME extrude
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
n = FacetNormal(mesh)
if args.problem == '2d':
    bottomflux = assemble(dot(sigma,n) * ds(3))
    topflux = assemble(dot(sigma,n) * ds(4))
else:
    assert NotImplementedError # FIXME extrude
imbalance = topflux + bottomflux
print(f'  mass flux out of top    = {topflux:13.6e}')
print(f'  mass flux into bottom   = {-bottomflux:13.6e}')
print(f'  imbalance               = {imbalance:13.6e}')

print('computing mass flux (kg m-2 s-1) from units ...')
printfluxes2d(mesh, sigma, topflux)

if args.problem == '2d' and args.k_type == 'verif':
    uexact = Function(H).interpolate(getuverif2d(mesh))
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
