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
<!---
The Radiative Transfer Equation (RTE) models the processes of absorption, scattering and emission 
as electromagnetic radiation propagates through a medium. I address the 1D RTE (\ref{RTE}) 
in a plane-parallel atmosphere and consider three sources: 
blackbody emission from the atmosphere $s(\tau)$, scattering from sunlight
$\frac{\omega I_0}{4 \pi} p\left(\mu, \phi ;-\mu_{0}, \phi_{0}\right) \exp\left(-\mu_{0}^{-1} \tau\right)$,
and incoming radiation from other atmospheric layers or the Earth's surface modeled by
Dirichlet boundary conditions.
--->

The Radiative Transfer Equation (RTE) models the processes of absorption, scattering and emission 
as electromagnetic radiation propagates through a medium. 
Consider a plane-parallel, horizontally homogeneous atmosphere with vertical coordinate 
$\tau$ (optical depth) increasing from top to bottom and directional coordinates $\phi$ for the azimuthal angle (positive is counterclockwise) 
and $\mu=\cos\theta$ for the polar direction ($\theta$ is the polar angle measured from the surface normal), 
with $\mu > 0$ pointing up following the convention of [@STWJ1988].
Given three possible sources: 
blackbody emission from the atmosphere $s(\tau)$, 
scattering from a collimated beam of starlight with intensity $I_0$ and incident cosine polar and azimuthal angles $\mu_0, \phi_0$,
and/or radiation from other atmospheric layers or the Earth's surface which is modeled by Dirichlet boundary conditions,
the diffuse intensity $u(\tau, \mu, \phi)$ propagating in direction $(\mu, \phi)$ 
is described by the 1D RTE [@Cha1960; @STWJ1988]:

\begin{align}
\begin{split}
\mu \frac{\partial u(\tau, \mu, \phi)}{\partial \tau} = u(\tau, \mu, \phi) &-\frac{\omega}{4 \pi} \int_{-1}^{1} \int_{0}^{2 \pi} p\left(\mu, \phi ; \mu', \phi'\right) u\left(\tau, \mu', \phi'\right) \mathrm{d} \phi' \mathrm{d} \mu' \\
&-\frac{\omega I_0}{4 \pi} p\left(\mu, \phi ;-\mu_{0}, \phi_{0}\right) \exp\left(-\mu_{0}^{-1} \tau\right) - s(\tau)
\end{split} \label{RTE}
\end{align}

Here $\omega$ is the single-scattering albedo and $p$ the scattering phase function.
These are assumed to be independent of $\tau$, i.e. homogeneous in the atmospheric layer.
An atmosphere with $\tau$-dependent $\omega$ and $p$ can be modelled by 
a multi-layer atmosphere with different $\omega$ and $p$ for each layer.

The RTE is important in many fields of science and engineering,
for example, in the retrieval of optical properties from measurements [@TCCGL1999; @MRO/CRISM2008; @TLZWSY2020].
The gold standard for numerically solving the 1D RTE is the Discrete Ordinate Radiative Transfer 
package `DISORT` which was coded in FORTRAN 77 and first released in 1988 [@STWJ1988; @Sta1999].
It has been widely used, for example by `MODTRAN` [@Ber2014], `Streamer` [@Key1998], and `SBDART` [@Ric1998],
all of which are comprehensive radiative transfer models that are themselves widely used in atmospheric science,
and by the three retrieval papers @TCCGL1999; @MRO/CRISM2008; @TLZWSY2020.
`DISORT` implements the Discrete Ordinates Method which has two key steps.
First, the diffuse intensity function $u$ and phase function $p$ are expanded as the Fourier cosine series and Legendre series respectively:

$$
\begin{aligned}
u\left(\tau, \mu, \phi\right) &\approx \sum_{m=0} u^m\left(\tau, \mu\right)\cos\left(m\left(\phi_0 - \phi\right)\right) \\
p\left(\mu, \phi ; \mu', \phi'\right) = p\left(\cos\gamma\right) &\approx \sum_{\ell=0} (2\ell + 1) g_\ell P_\ell\left(\cos\gamma\right)
\end{aligned}
$$

where $\gamma$ is the scattering angle.
These address the $\phi'$ integral in (\ref{RTE}) and decompose the problem into solving

$$
\mu \frac{d u^m(\tau, \mu)}{d \tau}=u^m(\tau, \mu)-\int_{-1}^1 D^m\left(\mu, \mu'\right) u^m\left(\tau, \mu'\right) \mathrm{d} \mu' - Q^m(\tau, \mu) - \delta_{0m}s(\tau)
$$

for each Fourier mode of $u$. The terms $D^m$ are derived from $p$ 
and are thus also independent of $\tau$. The second key step is to discretize the 
$\mu'$ integral using some quadrature scheme; `DISORT` 
uses the double-Gauss quadrature scheme from @Syk1951. 
This results in a system of ordinary differential equations that can be solved using standard methods.

Our package `PythonicDISORT` is a Python 3 reimplementation of `DISORT` that replicates 
most of its functionality while being easier to install, use, and modify, 
though at the cost of computational speed. It has `DISORT`'s main features: 
multi-layer solver, delta-M scaling, Nakajima-Tanaka (NT) corrections, only flux option, 
direct beam source, isotropic internal source (blackbody emission), Dirichlet boundary conditions 
(diffuse flux boundary sources), Bi-Directional Reflectance Function (BDRF)
for surface reflection, and more. In addition, `PythonicDISORT` has been 
tested against `DISORT` on `DISORT`'s own test problems. While packages 
that wrap `DISORT` in Python already exist [@CM2020; @Hu2017],
`PythonicDISORT` is the first time `DISORT`
has been reimplemented from scratch in Python.

# Statement of need

`PythonicDISORT` is not meant to replace `DISORT`. Due to fundamental 
differences between Python and FORTRAN, `PythonicDISORT`, though optimized,
remains about an order of magnitude slower than `DISORT`. Thus, projects that
prioritize computational speed should still use `DISORT`. I will continue to optimize
`PythonicDISORT`; there remain avenues for code vectorization among other optimizations.
It is unlikely that `PythonicDISORT` can be optimized to achieve the speed of `DISORT` though.
In addition, `PythonicDISORT` currently lacks `DISORT`'s latest features, 
most notably its pseudo-spherical correction, though
I am open to adding new features and I added a subroutine 
to compute actinic fluxes to satisfy a user request.

`PythonicDISORT` is instead designed with three goals in mind.
First, it is meant to be a pedagogical and exploratory tool. 
`PythonicDISORT`'s ease of installation and use makes it a low-barrier 
introduction to Radiative Transfer and Discrete Ordinates Solvers. 
Even researchers who are experienced in the field may find it useful to experiment 
with `PythonicDISORT` before deciding whether and how to upscale with `DISORT`.
Installation of `PythonicDISORT` through `pip` should be system agnostic
as `PythonicDISORT`'s core dependencies are only `NumPy` [@NumPy] and `SciPy` [@SciPy].
I also intend to implement `conda` installation. In addition, using 
`PythonicDISORT` is as simple as calling the Python function `pydisort`. In contrast,
`DISORT` requires FORTRAN compilers, has a lengthy and system dependent
installation, and each call requires shell script for compilation and execution.

Second, `PythonicDISORT` is designed to be modified by users to suit their needs.
Given that Python is a widely-used high-level language, `PythonicDISORT`'s 
code should be understandable, at least more so than `DISORT`'s FORTRAN code. 
Moreover, `PythonicDISORT` comes with a Jupyter Notebook [@JupyterNotebook] -- 
our [*Comprehensive Documentation*](https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html) --
that breaks down both the mathematics 
and code behind the solver. Users can in theory follow the Notebook 
to recode `PythonicDISORT` from scratch; 
it should at least help them make modifications.

Third, `PythonicDISORT` is intended to be a testbed.
For the same reasons given above, it should be easier 
to implement and test experimental features in `PythonicDISORT` than in `DISORT`.
This should expedite research and development for `DISORT` and similar algorithms.

`PythonicDISORT` was first released on [PyPI](https://pypi.org/project/PythonicDISORT/) 
and [GitHub](https://github.com/LDEO-CREW/Pythonic-DISORT) on May 30, 2023.
I know of its use in at least three ongoing projects: 
on the Two-Stream Approximations, on atmospheric photolysis, 
and on the topographic mapping of Mars through photoclinometry.
I will continue to maintain and upgrade `PythonicDISORT`. Our latest version: 
`PythonicDISORT v0.8.0` was released on June 10, 2024.

# Acknowledgements

I acknowledge funding from NSF through the Learning the Earth with Artificial intelligence and Physics (LEAP) 
Science and Technology Center (STC) (Award #2019625). I am also grateful to my Columbia University PhD advisor 
Dr. Robert Pincus and co-advisor Dr. Kui Ren for their advice and contributions.

# References