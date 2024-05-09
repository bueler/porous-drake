from firedrake import *
from physical import R, T, M, mu, Patm
from cases2d import getmesh2d, getgeounits2d, getgeounitsverif2d, \
                    getdirbcs2d, getuverif2d, unitsurfacefluxes2d

def test_verif2d():
    mesh = getmesh2d(10, 22, quad=True)
    # note mz=22 gives mesh aligned to k discontinuities
    H = FunctionSpace(mesh, 'CG', 1)
    u = Function(H)
    v = TestFunction(H)
    k, phi = getgeounitsverif2d(mesh)
    c = M / (R * T)
    alf = 1.0 / (2.0 * c * mu)
    F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)
    bcs = getdirbcs2d(mesh, H)
    solve(F == 0, u, bcs=bcs, options_prefix='s',
          solver_parameters = {'snes_type': 'ksponly',
                               'ksp_type': 'preonly',
                               'pc_type': 'lu',
                               'pc_factor_mat_solver_type': 'mumps'})
    uneg = Function(H).interpolate((-u + abs(u))/2.0)
    assert norm(uneg, 'L2') == 0.0
    uexact = Function(H).interpolate(getuverif2d(mesh))
    err = errornorm(u, uexact, norm_type='L2') / norm(uexact, norm_type='L2')
    assert err < 1.0e-14

def test_synth2d():
    mesh = getmesh2d(100, 22, quad=True)
    H = FunctionSpace(mesh, 'CG', 1)
    u = Function(H)
    v = TestFunction(H)
    k, phi = getgeounits2d(mesh)
    c = M / (R * T)
    alf = 1.0 / (2.0 * c * mu)
    F = alf * k * dot(grad(u), grad(v)) * dx(degree=4)
    bcs = getdirbcs2d(mesh, H)
    solve(F == 0, u, bcs=bcs, options_prefix='s',
          solver_parameters = {'snes_type': 'ksponly',
                               'ksp_type': 'preonly',
                               'pc_type': 'lu',
                               'pc_factor_mat_solver_type': 'mumps'})
    uneg = Function(H).interpolate((-u + abs(u))/2.0)
    assert norm(uneg, 'L2') == 0.0
    S = FunctionSpace(mesh, 'RTCF', 2)
    sigma = Function(S).project(- alf * k * grad(u))
    n = FacetNormal(mesh)
    bottomflux = assemble(dot(sigma,n) * ds(3))
    topflux = assemble(dot(sigma,n) * ds(4))
    imbalance = topflux + bottomflux
    assert imbalance < 2.0e-13
    CV, OB, FV = unitsurfacefluxes2d(mesh, sigma, topflux)
    assert abs(CV - 98.1) < 0.1
    assert abs(OB - 0.11) < 0.01
    assert abs(FV - 1.76) < 0.01

#test_verif2d()
#test_synth2d()
