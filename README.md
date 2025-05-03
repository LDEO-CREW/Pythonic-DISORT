[![Run PyTests](https://github.com/LDEO-CREW/Pythonic-DISORT/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/LDEO-CREW/Pythonic-DISORT/actions/workflows/continuous_integration.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.06442/status.svg)](https://doi.org/10.21105/joss.06442)

# Introduction
The PythonicDISORT package is a Discrete Ordinates Solver for the (1D) Radiative Transfer Equation 
in a plane-parallel, horizontally homogeneous atmosphere.
It is coded entirely in Python 3 and is a reimplementation instead of a wrapper.
While PythonicDISORT has been optimized for speed, it will naturally be slower than similar FORTRAN algorithms.
On the other hand, PythonicDISORT should be easier to install, use, and modify than FORTRAN-based Discrete Ordinates Solvers.

PythonicDISORT is based on Stamnes' FORTRAN DISORT (see References, in particular [2, 3, 8]) and has its main features: multi-layer solver, 
delta-_M_ scaling, Nakajima-Tanaka (NT) corrections, only flux option, direct beam source, isotropic internal source (blackbody emission), 
Dirichlet boundary conditions (diffuse flux boundary sources), Bi-Directional Reflectance Function (BDRF) for surface reflection,
and interpolation with respect to polar angle.
In addition, we added a subroutine to compute actinic fluxes to satisfy a user request, and integration with respect to optical depth was also added.
Further feature requests as well as feedback are welcome.

You may contact me, Dion, through dh3065@columbia.edu.

The **GitHub repository** is https://github.com/LDEO-CREW/Pythonic-DISORT.

Accompanying Journal of Open Source Software paper: https://joss.theoj.org/papers/10.21105/joss.06442.

# Documentation
https://pythonic-disort.readthedocs.io/en/latest/

Also see the accompanying Jupyter Notebook `Pythonic-DISORT.ipynb` in the `docs` directory
of our GitHub repository.
This Jupyter Notebook provides comprehensive documentation, suggested inputs, explanations, 
mathematical derivations and verification tests.
It is highly recommended that new users read the non-optional parts of sections 1 and 2.

## PyTest and examples of how to use PythonicDISORT

Not only are there verification tests in `Pythonic-DISORT.ipynb`, 
most of the test problems in Stamnes' `disotest.f90` (download DISORT 4.0.99 from http://www.rtatmocn.com/disort/) have also been recreated and enhanced.
In these tests, the solutions from PythonicDISORT are compared against solutions 
from a F2PY-wrapped Stamnes' DISORT (version 4.0.99; wrapper inspired by https://github.com/kconnour/pyRT_DISORT). With PyTest installed, execute the console command `pytest` 
in the `pydisotest` directory to run these tests. The `pydisotest` directory also contains Jupyter Notebooks to show the implementation of each test.
These notebooks double up as examples of how to use PythonicDISORT. The tests which have been implemented are:

* Test Problem 1: Isotropic Scattering
* Test Problem 2: Rayleigh Scattering, Beam Source
* Test Problem 3: Henyey-Greenstein Scattering
* Test Problem 4: Haze-L Scattering, Beam Source
* Test Problem 5: Cloud C.1 Scattering, Beam Source
* Test Problem 6: No Scattering, Increasingly Complex Sources (relevant for modeling longwave radiation)
* Test Problem 7: Absorption + Scattering + All Possible Sources, Lambertian and Hapke Surface Reflectivities (one layer)
* Test Problem 8: Absorbing / Isotropic-Scattering Medium (multiple layers)
* Test Problem 9: General Emitting / Absorbing / Scattering Medium (multiple layers)
* Test Problem 11: Single-Layer vs. Multiple Layers (no corresponding Jupyter Notebook)

# Installation

* From PyPI: `pip install PythonicDISORT`
* From Conda-forge: (TODO: need to first publish on Conda-forge)
* By cloning repository: `pip install .` in the `Pythonic-DISORT` directory; `pip install -r all_optional_dependencies.txt` to install all optional dependencies (see *Requirements to run PythonicDISORT*)

## Requirements to run PythonicDISORT
* Python 3.8+
* `numpy >= 1.8.0`
* `scipy >= 1.8.0`
* (OPTIONAL) `pytest >= 6.2.5` (Required to use the command `pytest`, see *PyTest and examples of how to use PythonicDISORT*)

## (OPTIONAL) Additional requirements to run the Jupyter Notebook
* `autograd >= 1.5`
* `jupyter > 1.0.0`
* `notebook > 6.5.2`
* `matplotlib >= 3.6.0`

In addition, a F2PY-wrapped Stamnes' DISORT, or equivalent, is required to properly run the last section (section 6).

## Compatibility

The PythonicDISORT package should be system agnostic given its minimal dependencies and pure Python code.
Everything in the repository was built and tested on Windows 11.

# Acknowledgements

I acknowledge funding from NSF through the Learning the Earth with Artificial intelligence and Physics (LEAP) 
Science and Technology Center (STC) (Award #2019625) under which this package was initially created.

# References
1) S. Chandrasekhar. 1960. *Radiative Transfer.*

2) Knut Stamnes and S-Chee Tsay and Warren Wiscombe and Kolf Jayaweera. 1988. *Numerically stable algorithm for discrete-ordinate-method radiative transfer in multiple scattering and emitting layered media.* http://opg.optica.org/ao/abstract.cfm?URI=ao-27-12-2502.

3) Stamnes, S.. 1999. *LLLab disort website.* http://www.rtatmocn.com/disort/.

4) Knut Stamnes and Paul Conklin. 1984. *A new multi-layer discrete ordinate approach to radiative transfer in vertically inhomogeneous atmospheres.* https://www.sciencedirect.com/science/article/pii/0022407384900311.

5) W. J. Wiscombe. 1977. *The Delta–M Method: Rapid Yet Accurate Radiative Flux Calculations for Strongly Asymmetric Phase Functions.* https://journals.ametsoc.org/view/journals/atsc/34/9/1520-0469_1977_034_1408_tdmrya_2_0_co_2.xml.

6) J. H. Joseph and W. J. Wiscombe and J. A. Weinman. 1976. *The Delta-Eddington Approximation for Radiative Flux Transfer.* https://journals.ametsoc.org/view/journals/atsc/33/12/1520-0469_1976_033_2452_tdeafr_2_0_co_2.xml.

7) Sykes, J. B.. 1951. *Approximate Integration of the Equation of Transfer.* https://doi.org/10.1093/mnras/111.4.377.

8) Stamnes, Knut and Tsay, Si-Chee and Wiscombe, Warren and Laszlo, Istvan and Einaudi, Franco. 2000. *General Purpose Fortran Program for Discrete-Ordinate-Method Radiative Transfer in Scattering and Emitting Layered Media: An Update of DISORT.*

9) Z. Lin and S. Stamnes and Z. Jin and I. Laszlo and S.-C. Tsay and W.J. Wiscombe and K. Stamnes. 2015. *Improved discrete ordinate solutions in the presence of an anisotropically reflecting lower boundary: Upgrades of the DISORT computational tool.* https://www.sciencedirect.com/science/article/pii/S0022407315000679.

10) Trefethen, L. N.. 1996. *Finite difference and spectral methods for ordinary and partial differential equations.* https://people.maths.ox.ac.uk/trefethen/pdetext.html.

11) Knut Stamnes. 1982. *On the computation of angular distributions of radiation in planetary atmospheres.* https://www.sciencedirect.com/science/article/pii/0022407382900966.

12) T. Nakajima and M. Tanaka. 1988. *Algorithms for radiative intensity calculations in moderately thick atmospheres using a truncation approximation.* https://www.sciencedirect.com/science/article/pii/0022407388900313.

13) Connour, Kyle and Wolff, Michael. 2020. *pyRT_DISORT: A pre-processing front-end to help make DISORT simulations easier in Python.* https://github.com/kconnour/pyRT_DISORT.
