# Installation instructions for F2PY-wrapped Stamnes' DISORT

## DISCLAIMER
This setup is meant for personal use on Windows 11 with Conda. Apologies in advance if it
does not work on your system. Everything in this directory is exactly Stamnes' DISORT 4.0.99
as downloaded from http://www.rtatmocn.com/disort/, 
except for our addition of this readme, the makefile
and a few lines of code (`Cf2py intent(in, out) [...]`) to `BDREF.f` and `DISORT.f`.
The code additions are required by F2PY but have no impact on the FORTRAN code
for the FORTRAN compiler treats them as comments.


In a fresh Conda environment, in the `Pythonic-DISORT` directory:
--------------------------------------------------------------------------------------------------

1) `cd disort4.0.99_f2py`
2) `conda install numpy`
3) `conda install -c msys2 m2w64-toolchain`
4) `mingw32-make`

Paste `disort.cp311-win_amd64.pyd` and everything in the `disort\.libs` directory 
into `site-packages` in the Conda environment directory.

--------------------------------------------------------------------------------------------------


Thank you to K. Connour and A. Stcherbinine for providing inspiration for this wrapper
through their `pyRT_DISORT` project: https://github.com/kconnour/pyRT_DISORT.
