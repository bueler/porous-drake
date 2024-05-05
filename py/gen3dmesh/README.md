# gen3dmesh/README.md

Demo of generating a 3D extruded mesh from a 2D mesh which is read from a `.msh` file.

To generate the (for test purposes) 2D mesh; creates `pentagon.msh`:

    $ gmsh -2 pentagon.geo

To read 2D mesh `pentagon.msh` into Firedrake, extrude a 3D mesh with variable surface elevation (height), and write-out a field on the 3D mesh in file `result.pvd`:

    $ python3 gen3dmesh.py

(Remember to activate the Firedrake venv.)

To view in Paraview:

    $ paraview result.pvd