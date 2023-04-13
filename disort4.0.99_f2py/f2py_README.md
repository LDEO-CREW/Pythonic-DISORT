# Installation instructions for Windows with Conda

## DISCLAIMER
This setup is meant for personal use and
as such we provide no alternatives. Apologies in advance if they
do not work on your system. We added this readme, the makefile,
and a few lines of code (`Cf2py intent(in, out) [...]`) to `BDREF.f` and `DISORT.f`
which are required by F2PY but have no impact on the FORTRAN code. 
Everything else is this directory is exactly Stamnes' DISORT 4.0.99
as downloaded from http://www.rtatmocn.com/disort/.


In a new Conda environment, in the `Pythonic-DISORT` directory:
--------------------------------------------------------------------------------------------------

1) `cd disort4.0.99_f2py`
2) `conda install numpy`
3) `conda install -c msys2 m2w64-toolchain`
4) `mingw32-make`

Paste `disort.cp311-win_amd64.pyd` and everything in the `disort\.libs` directory 
into the `site-packages` directory in the desired conda environment.

--------------------------------------------------------------------------------------------------


Thank you to K. Connour and A. Stcherbinine (https://github.com/kconnour/pyRT_DISORT)
for providing inspiration for this setup.
