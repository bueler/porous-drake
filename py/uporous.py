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
parser.add_argument('-bottomz', type=float, default=0.0, metavar='ZB',
                    help='elevation of base of domain (-dim 3 only) [default 0.0]')
parser.add_argument('-dim', type=int, metavar='D', default=2,
                    help='problem dimension: 2|3 [default 2]')
parser.add_argument('-k_type', metavar='X', default='synth',
                    choices=['synth','flat','verif'],
                    help='permeability structures: synth|flat|verif (-dim 2 only) [default synth]')
parser.add_argument('-mesh2d', metavar='FILE', default='',
                    help='file name for 2d base mesh (-dim 3 only)')
parser.add_argument('-mx', type=int, default=100, metavar='MX',
                    help='x resolution (-dim 2 only) [default 100]')
parser.add_argument('-mz', type=int, default=22, metavar='MZ',
                    help='z resolution [default 22]')
parser.add_argument('-order', type=int, default=1, metavar='J',
                    help='polynomial order for density rho [default 1]')
parser.add_argument('-triangles', action='store_true', default=False,
                    help='use triangular elements (-dim 2 only)')
args, passthroughoptions = parser.parse_known_args()

import petsc4py
petsc4py.init(passthroughoptions)
from firedrake import *
from firedrake.output import VTKFile
from physical import R, T, M, mu, Patm
from cases2d import getmesh2d, getgeounits2d, getgeounitsverif2d, \
                    getdirbcs2d, getuverif2d, unitsurfacefluxes2d

if args.dim == 2:
    elements = 'triangular' if args.triangles else 'quadrilateral'
    print(f'generating 2d {elements} mesh of {args.mx} x {args.mz} elements ...')
    mesh = getmesh2d(args.mx, args.mz, quad=not args.triangles)
else: # 3d
    print(f'reading 2d surface topography ("basemesh") from {args.mesh2d} ...')
    with CheckpointFile(args.mesh2d, 'r') as afile:
        basemesh = afile.load_mesh('basemesh')
        ztopbm = afile.load_function(basemesh, 'z')
    print(f'generating 3d extruded mesh with {args.mz} layers ...')
    mesh = ExtrudedMesh(basemesh, layers=args.mz, layer_height=1.0/args.mz)
    # compute a notional height function (for test purposes) on the base mesh
    xb, yb = SpatialCoordinate(basemesh)
    Vbase = FunctionSpace(basemesh,'CG',1)
    # extend z defined on the basemesh to z coordinate on the extruded mesh
    VR = FunctionSpace(mesh, 'CG', 1, vfamily='R', vdegree=0)
    ztop = Function(VR)
    ztop.dat.data[:] = ztopbm.dat.data_ro[:]
    x, y, z01 = SpatialCoordinate(mesh)  # note  0 <= z01 <= 1  here
    Vcoord = mesh.coordinates.function_space()
    zrescale = (ztop - args.bottomz) * z01 + args.bottomz
    XYZ = Function(Vcoord).interpolate(as_vector([x, y, zrescale]))
    mesh.coordinates.assign(XYZ)

# choose function spaces
H = FunctionSpace(mesh, 'CG', args.order)
u = Function(H, name='u (rho^2)')
v = TestFunction(H)

# permeability and porosity fields
if args.dim == 2:
    if args.k_type == 'verif':
        k, phi = getgeounitsverif2d(mesh)
    else:
        k, phi = getgeounits2d(mesh, flat=(args.k_type == 'flat'))
else: # 3d
    # FIXME come up with preferred fields for k(x,y,z) and phi(x,y,z)
    from cases2d import kCV, phiCV
    k = Constant(kCV)
    phi = Constant(phiCV)

# primal CG weak form
c = M / (R * T)   # ideal gas law is  rho = c P
alf = 1.0 / (2.0 * c * mu)
F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)

if args.dim == 2:
    bcs = getdirbcs2d(mesh, H)
else: # 3d
    u_top = (c * Patm)**2
    Pbot = 1100000.0   # Pa; = 11 bar
    u_bot = (c * Pbot)**2
    bcs = [DirichletBC(H, u_bot, 'bottom'),
           DirichletBC(H, u_top, 'top')]

# linear solve for u
print(f'solving primal CG{args.order} weak form for u ...')
solve(F == 0, u, bcs=bcs, options_prefix='s',
      solver_parameters = {'snes_type': 'ksponly',
                           'ksp_type': 'preonly',
                           'pc_type': 'lu',
                           'pc_factor_mat_solver_type': 'mumps'})

# warn if negative u
uneg = Function(H).interpolate((-u + abs(u))/2.0)
uneg.rename('u_- (negative part of u)')
if norm(uneg, 'L2') > 0.0:
    print('WARNING: nonzero negative part of u(x,z) detected')
    print('         see "uneg" field in output file')
upos = Function(H).interpolate((u + abs(u))/2.0)

if args.dim == 2:
    # recover mass flux sigma to measure conservation
    if args.triangles:
        S = FunctionSpace(mesh, 'RT', args.order + 1)
    else:
        S = FunctionSpace(mesh, 'RTCF', args.order + 1)
    sigma = Function(S).project(- alf * k * grad(u))
    sigma.rename('sigma (mass flux; kg m-2 s-1)')

    print('mass conservation (kg m-2 s-1):')
    n = FacetNormal(mesh)
    if args.dim == 2:
        bottomflux = assemble(dot(sigma,n) * ds(3))
        topflux = assemble(dot(sigma,n) * ds(4))
    else: # -problem extrude
        assert NotImplementedError # FIXME extrude
    imbalance = topflux + bottomflux
    print(f'  flux out of top         = {topflux:13.6e}')
    print(f'  flux into bottom        = {-bottomflux:13.6e}')
    print(f'  imbalance               = {imbalance:13.6e}')

    if args.k_type == 'synth':
        print('mass flux (kg m-2 s-1) for each surface unit:')
        unitsurfacefluxes2d(mesh, sigma, topflux, printthem=True)

    if args.k_type == 'verif':
        uexact = Function(H).interpolate(getuverif2d(mesh))
        err = errornorm(u, uexact, norm_type='L2') / norm(uexact, norm_type='L2')
        print(f'verification case: |u-u_exact|_2/|u|_2 = {err:6.2e}')

    # recover and write physical scalars and vectors
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

else: # 3d
    # FIXME generate sigma, q, v; measure conservation; write sigma, q, v
    rho = Function(H).interpolate(sqrt(abs(u)))
    rho.rename('rho (density; kg m-3)')
    P = Function(H).interpolate(c * rho)
    P.rename('p (pressure; Pa)')
    print('saving fields to result.pvd ...')
    VTKFile("result.pvd").write(u, uneg, rho, P)
