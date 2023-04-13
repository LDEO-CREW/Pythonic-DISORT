# Introduction
The PyDISORT package is a Discrete Ordinates Solver for the (1D) Radiative Transfer Equation in a single or multi-layer atmosphere.
It is coded entirely in Python 3 and in as "Pythonic" a manner as possible: we vectorize as well as use list comprehension and `map` as much as possible. 

It is based on Stamnes' FORTRAN DISORT (see references in the Jupyter Notebook) and has its main features: 
delta-M scaling, Nakajima-Tanaka (NT) corrections, only flux option, isotropic internal sources (thermal source),
Bi-Direction Reflectance Function (BDRF) and more.

This repository also includes our F2PY-wrapped Stamnes DISORT (version 4.0.99) in the `disort4.0.99_f2py` directory.
The original was downloaded from http://www.rtatmocn.com/disort/.

# Documentation
https://pythonic-disort.readthedocs.io/en/latest/.

Also see the accompanying Jupyter Notebook: `Pythonic-DISORT.ipynb` in the `docs` directory.
The Jupyter Notebook provides comprehensive documentation, suggested inputs, explanations, 
mathematical derivations and verification tests.
We highly recommend reading the non-optional parts of sections 1 and 2 before use.

Separate from the verification tests in the notebook, we used PyTest to recreate most of the test problems from Stamnes et. al.'s `disotest.f90`.
Execute the console command `pytest` in the `PyTests` directory to run these tests.

# Installation

Installation instructions: TODO (we need to first publish PyDISORT on PyPI).

## Requirements to run PyDISORT
* `numpy >= 1.12`
* `scipy >= 1.2.1`.

## Additional requirements to run the Jupyter Notebook
* `autograd >= 1.5`
* `jupyter > 1.0.0`
* `notebook > 6.5.2`

Our F2PY-wrapped Stamnes' DISORT (in the `disort4.0.99_f2py` directory) must also be set up to run the last section (section 6).

## Compatibility

The PyDISORT package should be system agnostic given its minimal dependencies and pure Python code.
We are not sure if the Jupyter Notebook or our F2PY-wrapped Stamnes' DISORT will run without issues on non-Windows systems.
Everything was built and tested on Windows 11 and not yet tested on other systems.

