# Introduction
The PythonicDISORT package is a Discrete Ordinates Solver for the (1D) Radiative Transfer Equation in a single or multi-layer atmosphere.
It is coded entirely in Python 3 and in as "Pythonic" a manner as possible: we vectorize as well as use list comprehension and `map` as much as possible. 

PythonicDISORT is based on Stamnes' FORTRAN DISORT (see References, in particular [2, 3, 8]) and has its main features: 
delta-M scaling, Nakajima-Tanaka (NT) corrections, only flux option, isotropic internal sources (thermal source),
Bi-Directional Reflectance Function (BDRF) and more.

This repository also includes our F2PY-wrapped Stamnes DISORT (version 4.0.99) in the `disort4.0.99_f2py` directory.
The original was downloaded from http://www.rtatmocn.com/disort/. The wrapper was inspired by https://github.com/kconnour/pyRT_DISORT.

# Documentation
https://pythonic-disort.readthedocs.io/en/latest/

Also see the accompanying Jupyter Notebook: `Pythonic-DISORT.ipynb` in the `docs` directory 
at https://github.com/LDEO-CREW/Pythonic-DISORT.
The Jupyter Notebook provides comprehensive documentation, suggested inputs, explanations, 
mathematical derivations and verification tests.
We highly recommend reading the non-optional parts of sections 1 and 2 before use.

## PyTest

Separate from the verification tests in the Jupyter Notebook, we used PyTest to recreate most of the test problems from Stamnes et. al.'s `disotest.f90`.
With PyTest installed, execute the console command `pytest` in the `pydisotest` directory to run these tests.

# Installation

* From PyPI: `pip install PythonicDISORT`
* From Conda-forge: (TODO: we need to first publish on Conda-forge)
* By cloning repository: `pip install .` in the `Pythonic-DISORT` directory; `pip install -r all_optional_dependencies.txt` to install all optional dependencies

## Requirements to run PythonicDISORT
* Python 3.8+
* `numpy >= 1.17.3`
* `scipy >= 1.8.0`
* (OPTIONAL) `joblib >= 1.0.0` (Required for parallelization)
* (OPTIONAL) `pytest >= 6.2.5` (Required for non-Notebook tests)

## Additional requirements to run the Jupyter Notebook
* `autograd >= 1.5`
* `jupyter > 1.0.0`
* `notebook > 6.5.2`

In addition, our F2PY-wrapped Stamnes' DISORT (in the `disort4.0.99_f2py` directory) must be set up to run the last section (section 6).

## Compatibility

The PythonicDISORT package should be system agnostic given its minimal dependencies and pure Python code, 
but we cannot say the same for our Jupyter Notebook and F2PY-wrapped Stamnes' DISORT.
Everything was built and tested on Windows 11 and not yet tested on other systems.

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

11) T. Nakajima and M. Tanaka. 1988. *Algorithms for radiative intensity calculations in moderately thick atmospheres using a truncation approximation.* https://www.sciencedirect.com/science/article/pii/0022407388900313.

12) K. Connour and A. Stcherbinine. 2022. *pyRT_DISORT.* https://github.com/kconnour/pyRT_DISORT.