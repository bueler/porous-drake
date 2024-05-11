# py/

Copyright 2024 Tara Shreve and Ed Bueler

## Introduction

The Python programs in this directory solve some gas flow problems in porous media problems using Darcy's law, for discontinuous permeability.  They use the following open-source libraries and tools:

  * [Firedrake](https://www.firedrakeproject.org/), a Python finite element library
  * [Paraview](https://www.paraview.org/), for visualization of mesh functions

Follow the instructions at the [Firedrake download page](https://www.firedrakeproject.org/download.html) to install Firedrake.  Before running any code here you will need to activate your [Firedrake](https://www.firedrakeproject.org/) virtual environment (venv) first:

    $ source ~/firedrake/bin/activate

The main program is `uporous.py`.  It is documented by `doc.pdf` in the `latex/` directory.  To see its allowed options run

    $ python3 uporous.py -h

## 2D test cases usage

In default usage the program `uporous.py` sets up a 2D domain ($x,z$ coordinates) with three textural units for permeability (`k`) and porosity (`phi`).  This is a simplified structure analogous to the Obsidian Dome configuration, but with a flat surface toporaphy and rectangular blocks for the units.  When you run

    $ python3 uporous.py

it solves on a low-resolution grid and saves the result in `result.pvd`.  Note that conservation of mass flux is reported, as well as the fractions of mass flux from the three units.

One can ask for a refined grid, higher-order elements, or triangular elements:

    $ python3 uporous.py -mx 400 -mz 88
    $ python3 uporous.py -order 2
    $ python3 uporous.py -triangles

Simpler cases provide a minimal verification.  These are documented in section 1.3 of `doc.pdf`.  For example, running 

    $ python3 uporous.py -g 0.0 -k_type verif

solves the case called ``Three horizontal units solution.''  This case reports the norm difference between the exact and numerical solutions.

If the vertical resolution is chosen so that the discontinuities in `k` are aligned to the grid then the reported error between the numerical solution and the exact solution is at the level of rounding error.  On the other hand, generic vertical refinement paths give converging results.  Compare:

    $ for Z in 22 44 88; do python3 uporous.py -g 0.0 -k_type verif -mz $Z; done   # aligned
    $ for Z in 20 40 80; do python3 uporous.py -g 0.0 -k_type verif -mz $Z; done   # not aligned

## 3D Obsidian Dome case

The problem can be solved on an extruded 3D mesh for the Obsidian Dome topography.  First one converts the topography data in `obsidianDome_filt50m.txt` as follows:

    $ python3 convertodome.py

This writes a 2D mesh, with a surface elevation field, to the HDF5 format file `odome.h5`.  Then 2D mesh is read, and extruded to 3D, by the solver as follows:

    $ python3 uporous.py -dim 3 -mesh2d odome.h5 -bottomz -100

Note that options `-mz` and `-order` also make sense in this case.

FIXME: this case is not fully implemented

TODO: useful units, generation of sigma/q/v outputs, reported mass conservation

FIXME: the direct solver has severe resolution limitations in 3D

## Visualization

The `.pvd` output files always include the gas density (`rho`) and pressure (`P`).  In the current code configuration, additional vector fields, mass flux (`sigma`), volumetric/Darcy flux (`q`), and gas velocity (`v`), are written in the 2D test cases.  Note that Paraview allows you to generate a stream-line plot of the vector functions.

## Testing

This runs a couple of verification cases and a couple of cases like the 2d figures in `doc.pdf`.  Do:

    $ pytest .