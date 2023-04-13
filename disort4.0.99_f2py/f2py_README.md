Installation instructions for Windows with Conda

DISCLAIMER: This setup is meant for personal use and
as such we provide no alternatives. Apologies in advance if they
do not work on your system. We added this readme, the makefile,
and a few lines of code ("Cf2py intent(in, out) ...") to "BDREF.f" and "DISORT.f"
which are required by F2PY but have no impact on the FORTRAN code. 
Everything else is this directory is exactly Stamnes' DISORT 4.0.99
as downloaded from http://www.rtatmocn.com/disort/.


In a new Conda environment:
--------------------------------------------------------------------------------------------------

Execute:
"conda install numpy"
"conda install -c msys2 m2w64-toolchain"
"mingw32-make"

Paste "disort.cp311-win_amd64.pyd" and everything in the "disort\.libs" folder 
into the "site-packages" folder in the appropriate conda environment.

--------------------------------------------------------------------------------------------------


Thank you to K. Connour and A. Stcherbinine (https://github.com/kconnour/pyRT_DISORT)
for providing inspiration for this setup.