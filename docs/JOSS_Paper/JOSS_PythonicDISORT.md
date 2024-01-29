---
title: 'PythonicDISORT: A Python reimplementation of the Discrete Ordinate Radiative Transfer package DISORT'
tags:
  - Python
  - Radiative Transfer
  - Discrete Ordinates Method
  - DISORT
authors:
  - name: Ho Jia Xu Dion
	orcid: 0009-0000-5829-5081
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
affiliations:
 - name: Department of Applied Physics and Applied Mathematics (APAM), Columbia University, USA
   index: 1
date: 29 January 2023
bibliography: PythonicDISORT.bib

---

# Summary

The Radiative Transfer Equation (RTE) describes the propagation of electromagnetic 
radiation through a medium. It models the processes of absorption, scattering and emission.
As the sun is the main driver of planetary climates, the 1D RTE is central to climate models
both of Earth and of exoplanets. The gold standard for numerically solving 
the 1D RTE is the Discrete Ordinate Radiative Transfer FORTRAN package `DISORT` that was first released in 
1988 [@STWJ1988] and has been widely used, for example by `MODTRAN`
[@Ber2014], `Streamer` [@Key1998], and `SBDART` [@Ric1998]. Our package `PythonicDISORT`
is a Python reimplementation of `DISORT` that replicates most of its functionality while 
being much easier to install, use and modify, though at the cost of computational speed.

# Statement of need

`PythonicDISORT` is a Discrete Ordinates Solver for the 1D RTE 
in a single or multi-layer plane-parallel atmosphere and it is 
coded entirely in Python 3. It is based on `DISORT` [@STWJ1988], 
which was written in FORTRAN 77, and has `DISORT`'s main features: 
delta-M scaling, Nakajima-Tanaka (NT) corrections, only flux option, 
isotropic internal sources (thermal sources), Dirichlet boundary conditions 
(diffuse flux boundary sources), Bi-Directional Reflectance Function (BDRF) 
for surface reflection, and more. In addition, `PythonicDISORT` has been 
tested against `DISORT` on `DISORT`'s own test problems.

`PythonicDISORT` is not meant to replace `DISORT`. Due to fundamental 
differences between the Python and FORTRAN languages, `PythonicDISORT` 
is about an order of magnitude slower than `DISORT`. Thus, projects which
prioritize computational speed should still use `DISORT`. Moreover, `PythonicDISORT`
lacks `DISORT`'s latest features, most notably spherical correction.

`PythonicDISORT` is instead designed with three goals in mind.
First, it is meant to be a pedagogical and exploratory tool. `PythonicDISORT`'s ease of
installation and use makes it a low-barrier introduction to Radiative Transfer and the Discrete 
Ordinates Method. Only the NumPy and SciPy Python packages are essential for 
`PythonicDISORT`, thus most operating systems can easily install it 
through `pip`, and we intend to implement `conda` installation. In addition, using 
`PythonicDISORT` is as simple as calling the Python function `pydisort`. In contrast,
`DISORT` requires FORTRAN compilers, has a lengthy and operating system dependent
installation process, and each call requires shell script for compilation and execution.
Even researchers who are experienced with Radiative Transfer may want to use `PythonicDISORT`
to test whether Discrete Ordinates Solvers suit their needs before upscaling with `DISORT`.

Second, `PythonicDISORT` is designed to be modified by users. 
Given that Python is a widely used high-level language, `PythonicDISORT`'s 
code should be understandable, at least more so than `DISORT`'s FORTRAN code. 
Moreover, `PythonicDISORT` comes with a Jupyter Notebook (our *Comprehensive Documentation*) 
that breaks down both the mathematics and code behind the solver. 
Users can in theory follow the Notebook to recode `PythonicDISORT` 
from scratch; it should at least help them make modifications.

Third, we intend for `PythonicDISORT` to be a testbed for research on the Discrete 
Ordinates Method. For the same reasons given above, we expect that it is easier 
to implement and test experimental features in `PythonicDISORT` than in `DISORT`.
This should expedite research and development for `DISORT` and similar algorithms.

`PythonicDISORT` was first released on PyPI and GitHub on May 30, 2023.
We know of its use in at least three ongoing projects: 
on the Two-Stream Equations, on atmospheric photolysis, 
and on the topographic mapping of Mars through photoclinometry.
We will continue to maintain and upgrade `PythonicDISORT`. Our latest version: 
`PythonicDISORT v0.4.2` was released on Nov 28, 2023.

# Acknowledgements

I am grateful to my PhD advisor Robert Pincus and my co-advisor Kui Ren for their advice
and contributions. I am also grateful to Learning the Earth with Artificial Intelligence and Physics (LEAP)
for its generous funding.

# References