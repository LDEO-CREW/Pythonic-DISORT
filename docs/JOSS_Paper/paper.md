---
title: '`PythonicDISORT`: A Python reimplementation of the Discrete Ordinate Radiative Transfer package `DISORT`'
tags:
  - Python
  - Radiative Transfer
  - Discrete Ordinates Method
  - Atmospheric Science
  - Climate Models
  - DISORT
authors:
  - name: Dion J. X. Ho
    orcid: 0009-0000-5829-5081
    affiliation: "1"
affiliations:
 - name: Columbia University, Department of Applied Physics and Applied Mathematics, USA
   index: 1
date: 11 February 2024
bibliography: paper.bib
---

<!---
title: 'Gala: A Python package for galactic dynamics'
tags:
  - Python
  - astronomy
  - dynamics
  - galactic dynamics
  - milky way
authors:
  - name: Adrian M. Price-Whelan
    orcid: 0000-0000-0000-0000
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
  - given-names: Ludwig
    dropping-particle: van
    surname: Beethoven
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, USA
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
--->

# Summary
The Radiative Transfer Equation (RTE) models the processes of absorption, scattering and emission 
as electromagnetic radiation propagates through a medium. We address the 1D RTE (\ref{RTE}) 
in a plane-parallel atmosphere and consider three sources: 
blackbody emission from the atmosphere $s(\tau)$, scattering from sunlight
$\frac{\omega I_0}{4 \pi} p\left(\mu, \phi ;-\mu_{0}, \phi_{0}\right) \exp\left(-\mu_{0}^{-1} \tau\right)$,
and incoming radiation from other atmospheric layers or the Earth's surface modeled by
Dirichlet boundary conditions.

\begin{align}
\begin{split}
\mu \frac{\partial u(\tau, \mu, \phi)}{\partial \tau} = u(\tau, \mu, \phi) &-\frac{\omega}{4 \pi} \int_{-1}^{1} \int_{0}^{2 \pi} p\left(\mu, \phi ; \mu', \phi'\right) u\left(\tau, \mu', \phi'\right) \mathrm{d} \phi' \mathrm{d} \mu' \\
&-\frac{\omega I_0}{4 \pi} p\left(\mu, \phi ;-\mu_{0}, \phi_{0}\right) \exp\left(-\mu_{0}^{-1} \tau\right) - s(\tau)
\end{split} \label{RTE}
\end{align}

The RTE is important in many fields of science and engineering.
The gold standard for numerically solving the 1D RTE is the Discrete Ordinate Radiative Transfer 
package `DISORT` which was coded in FORTRAN 77 and first released in 1988 [@STWJ1988].
It has been widely used, for example by `MODTRAN` [@Ber2014], `Streamer` [@Key1998], and `SBDART` [@Ric1998],
all of which are comprehensive radiative transfer models that are themselves widely used in atmospheric science.
`DISORT` implements the Discrete Ordinates Method which has two key steps.
First, the diffuse intensity function $u$ is expanded as the Fourier cosine series:

$$
u\left(\tau, \mu, \phi\right) = \sum_{m=0} u^m\left(\tau, \mu\right)\cos\left(m\left(\phi_0 - \phi\right)\right)
$$

This addresses the $\phi'$ integral in (\ref{RTE}) and decomposes the problem into solving

$$
\mu \frac{d u^m(\tau, \mu)}{d \tau}=u^m(\tau, \mu)-\int_{-1}^1 D^m\left(\mu, \mu'\right) u^m\left(\tau, \mu'\right) \mathrm{d} \mu' - Q^m(\tau, \mu) - \delta_{0m}s(\tau)
$$

for each Fourier mode of $u$. The second key step is to discretize the $\mu'$ integral using 
some quadrature scheme; `DISORT` uses the double-Gauss quadrature scheme from @Syk1951.
This results in a system of ODEs that can be solved using standard methods.

Our package `PythonicDISORT` is a Python 3 reimplementation of `DISORT` that replicates 
most of its functionality while being easier to install, use, and modify, 
though at the cost of computational speed. It has `DISORT`'s main features: 
multi-layer solving, delta-M scaling, Nakajima-Tanaka (NT) corrections, only flux option, 
isotropic internal sources (thermal sources), Dirichlet boundary conditions 
(diffuse flux boundary sources), Bi-Directional Reflectance Function (BDRF) 
for surface reflection, and more. In addition, `PythonicDISORT` has been 
tested against `DISORT` on `DISORT`'s own test problems. As far as we know, all
prior attempts at creating Python interfaces for `DISORT` have focused
on creating wrappers and `PythonicDISORT` is the first true Python reimplementation.

# Statement of need

We clarify that `PythonicDISORT` is not meant to replace `DISORT`. Due to fundamental 
differences between Python and FORTRAN, `PythonicDISORT`, though optimized,
remains about an order of magnitude slower than `DISORT`. Thus, projects which
prioritize computational speed should still use `DISORT`. Moreover, `PythonicDISORT`
lacks `DISORT`'s latest features, most notably its pseudo-spherical correction.

`PythonicDISORT` is instead designed with three goals in mind.
First, it is meant to be a pedagogical and exploratory tool. 
`PythonicDISORT`'s ease of installation and use makes it a low-barrier 
introduction to Radiative Transfer and Discrete Ordinates Solvers. 
Even researchers who are experienced in the field may find it useful to experiment 
with `PythonicDISORT` before deciding whether and how to upscale with `DISORT`.
Installation of `PythonicDISORT` through `pip` should be system agnostic
as `PythonicDISORT`'s core dependencies are only `NumPy` and `SciPy`.
We also intend to implement `conda` installation. In addition, using 
`PythonicDISORT` is as simple as calling the Python function `pydisort`. In contrast,
`DISORT` requires FORTRAN compilers, has a lengthy and system dependent
installation process, and each call requires shell script for compilation and execution.

Second, `PythonicDISORT` is designed to be modified by users to suit their needs.
Given that Python is a widely used high-level language, `PythonicDISORT`'s 
code should be understandable, at least more so than `DISORT`'s FORTRAN code. 
Moreover, `PythonicDISORT` comes with a Jupyter Notebook 
(our *Comprehensive Documentation*) that breaks down both the mathematics 
and code behind the solver. Users can in theory follow the Notebook 
to recode `PythonicDISORT` from scratch; 
it should at least help them make modifications.

Third, we intend for `PythonicDISORT` to be a testbed.
For the same reasons given above, we expect that it is easier 
to implement and test experimental features in `PythonicDISORT` than in `DISORT`.
This should expedite research and development for `DISORT` and similar algorithms.

`PythonicDISORT` was first released on PyPI and GitHub on May 30, 2023.
We know of its use in at least three ongoing projects: 
on the Two-Stream Approximations, on atmospheric photolysis, 
and on the topographic mapping of Mars through photoclinometry.
We will continue to maintain and upgrade `PythonicDISORT`. Our latest version: 
`PythonicDISORT v0.4.2` was released on Nov 28, 2023.

# Acknowledgements

I acknowledge funding from NSF through the Learning the Earth with Artificial intelligence and Physics (LEAP) 
Science and Technology Center (STC) (Award #2019625). I am also grateful to my Columbia University PhD advisor 
Dr. Robert Pincus and co-advisor Dr. Kui Ren for their advice and contributions.

# References