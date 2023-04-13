# Create the library disort.pyd, which can be imported in python code

FILES=DISORT.f BDREF.f DISOBRDF.F ERRPACK.f LAPACK.f LINPAK.f RDI1MACH.f

disort.cp311-win_amd64.pyd: $(FILES)
	f2py -c --f90flags='-O3' -m disort $(FILES)

