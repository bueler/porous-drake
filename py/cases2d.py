from firedrake import *
from physical import g, R, T, M, mu, Patm

# overall dimensions of domain
Lx = 100.0              # width (m)
Lz = 22.0               # height (m)

## multiple geological units with discontinuous k
# note: k in m^2, phi dimensionless
zCVOB, zOBFV = -10.0, -4.0
xc = Lx / 2.0
xCVOB, xOBFV = 4.0, 12.0
kCV, phiCV = 6.87e-12, 0.500
kOB, phiOB = 4.94e-15, 0.0324
kFV, phiFV = 2.18e-13, 0.232

# data for Dirichlet boundary conditions
c = M / (R * T)    # ideal gas law is  rho = c P
u_top = (c * Patm)**2
Pbot = 1100000.0   # Pa; = 11 bar
u_bot = (c * Pbot)**2

def getmesh2d(mx, mz, quad=False):
    # generate mesh on [0,Lx] x [-Lz,0]
    # indices of four boundaries/sides:  (1, 2, 3, 4) = (left, right, bottom, top)
    if quad:
        mesh = RectangleMesh(mx, mz, Lx, Lz, quadrilateral=True)
    else:
        mesh = RectangleMesh(mx, mz, Lx, Lz, diagonal='crossed')
    # at this point mesh is on rectangle [0,Lx] x [0,Lz], so we shift
    x, z = SpatialCoordinate(mesh)
    Vcoord = mesh.coordinates.function_space()
    XZ = Function(Vcoord).interpolate(as_vector([x, z - Lz]))
    mesh.coordinates.assign(XZ)
    return mesh

def getgeounits2d(mesh, flat=False):
    if flat:
        k = Constant(kCV)
        phi = Constant(phiCV)
    else:
        x, z = SpatialCoordinate(mesh)
        dx = abs(x - xc)  # horizontal distance from center
        ## k = permeability field, from COMSOL-generated figure
        kupper = conditional(z < zOBFV, kOB, conditional(dx < xOBFV, kOB, kFV))
        k = conditional(z < zCVOB, kCV, conditional(dx < xCVOB, kCV, kupper))
        ## phi = porosity field
        phiupper = conditional(z < zOBFV, phiOB, conditional(dx < xOBFV, phiOB, phiFV))
        phi = conditional(z < zCVOB, phiCV, conditional(dx < xCVOB, phiCV, phiupper))
    return k, phi

def getgeounitsverif2d(mesh):
    x, z = SpatialCoordinate(mesh)
    kupper = conditional(z < zOBFV, kOB, kFV)
    k = conditional(z < zCVOB, kCV, kupper)
    phiupper = conditional(z < zOBFV, phiOB, phiFV)
    phi = conditional(z < zCVOB, phiCV, phiupper)
    return k, phi

def unitsurfacefluxes2d(mesh, sigma, topflux, printthem=False):
    x, z = SpatialCoordinate(mesh)
    # surface portion indicator functions
    dx = abs(x - xc)
    CVind = conditional(dx < xCVOB, 1.0, 0.0)
    OBind = conditional(dx < xOBFV, conditional(abs(x - xc) > xCVOB, 1.0, 0.0), 0.0)
    FVind = 1.0 - CVind - OBind
    # corresponding mass fluxes
    n = FacetNormal(mesh)
    CVflux = assemble(dot(CVind * sigma,n) * ds(4))
    OBflux = assemble(dot(OBind * sigma,n) * ds(4))
    FVflux = assemble(dot(FVind * sigma,n) * ds(4))
    fluxsum = CVflux + OBflux + FVflux
    assert abs(topflux - fluxsum)  < 1.0e-14 * topflux  # consistency with topflux
    CVper = 100*CVflux/topflux
    OBper = 100*OBflux/topflux
    FVper = 100*FVflux/topflux
    if printthem:
        print(f'  CV unit flux            = {CVflux:13.6e} ({CVper:5.2f} %)')
        print(f'  OB unit flux            = {OBflux:13.6e} ({OBper:5.2f} %)')
        print(f'  FV unit flux            = {FVflux:13.6e} ({FVper:5.2f} %)')
    return CVper, OBper, FVper

def getdirbcs2d(mesh, H):
    BCs = [DirichletBC(H, u_bot, 3),
           DirichletBC(H, u_top, 4)]
    return BCs

def getuverif2d(mesh):
    # construct piecewise-linear exact solution in case where
    # k=k(z) and u=u(z) are independent of x, using linear system
    #   M v = b
    import numpy as np
    z1, z2 = zCVOB, zOBFV
    k1, k2, k3 = kCV, kOB, kFV
    M = np.array([[z1+Lz,   0.0,   0.0,  -1.0],
                  [  0.0, z2-z1,   -z2,   1.0],
                  [   k1,   -k2,   0.0,   0.0],
                  [  0.0,    k2,   -k3,   0.0]])
    b = np.array([-u_bot, u_top, 0.0, 0.0])
    a1, a2, a3, b2 = tuple(np.linalg.solve(M, b))
    _, z = SpatialCoordinate(mesh)
    uupper = conditional(z < z2, a2 * (z - z1) + b2, a3 * z + u_top)
    return conditional(z < z1, a1 * (z + Lz) + u_bot, uupper)  # = u_exact(z)
