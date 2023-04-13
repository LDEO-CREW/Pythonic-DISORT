# $Rev: 90 $ $Date: 2017-11-30 20:01:24 -0500 (Thu, 30 Nov 2017) $

This is a README for an upgraded version of DISORT — DISORTv4.0.x — 
available from the Light and Life Lab website at Stevens Institute 
of Technology:

  http://lllab.phy.stevens.edu/disort/

and the NASA ftp site:

  ftp://climate1.gsfc.nasa.gov/wiscombe/Multiple_Scatt/

One significant improvement in DISORTv3+ is that the BRDF boundary condition 
is now being calculated more efficiently, so that NMUG can be set to a 
large value (now 200 by default) for accurate calculations.

FILE		DESCRIPTION			User modification needed?
-------------------------------------------------------------------------
disotest.f90	Driver program			Yes
-------------------------------------------------------------------------
DISOBRDF.f	Calculates BRDF for DISORT	Maybe
-------------------------------------------------------------------------
BDREF.f		Calculates BRDF			Maybe
-------------------------------------------------------------------------
DISORT.f	DISORT subroutine		No
-------------------------------------------------------------------------
DISOTESTAUX.f	Auxiliary routines		No
-------------------------------------------------------------------------
LAPACK.f	LAPACK				No
-------------------------------------------------------------------------
LINPACK.f	LINPACK				No
-------------------------------------------------------------------------
RDI1MACH.f	Sets machine precision		No
-------------------------------------------------------------------------
ERRPACK.f	Error messages			No
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The user will need to modify disotest.f90 to set-up the input to DISORT. 

For BRDF setup, a demo is provided as problem 15 in disotest.f90. 
"BRDF_TYPE" controls pre-defined BRDF types, and includes:
BRDF_TYPE = 1        Hapke BRDF
BRDF_TYPE = 2        Cox—Munk Gaussian BRDF
BRDF_TYPE = 3        RPV BRDF
BRDF_TYPE = 4        Ross-Li BRDF
"BRDF_ARG" controls BRDF parameters for each BRDF type.

To apply the Nakajima/Tanaka intensity correction, the user must input enough
phase function moments to guarantee an accurate phase function expansion, which
should be much larger than the number of streams ("NMOM" > "NSTR"). On the
other hand, if the user do not need N/T correction, then simply input the
number of moments to be equal to the number of streams ("NMOM" = "NSTR”).

In DISORT, all necessary changes are in the driver file:
disotest.f90. However, in two cases changes are needed in other files
as well:

1) To produce a new BRDF type (other than the 4 types provided in DIOSRT
version 4.0), users must write their own BRDF subroutine in BDREF.f, set up a
new "BRDF_TYPE”, and provide their own "BRDF_ARG" parameters in disotest.f90. 

2) The driver code demonstrates how to use FORTRAN dynamic memory allocation
arrays to run DISORT. The user must set up the following variables: NLYR, NMOM,
NSTR, NUMU, NPHI, NTAU and use the "allocate_disort_allocatable_arrays"
subroutine to allocate all arrays needed by DISORT.    

If you have any issues/comments/questions please send an email to Snorre Stamnes 
at snorre.a.stamnes@nasa.gov.

The following scripts expect the gfortran compiler (versions 4.x and 3.x)
to be installed. But if you have a different compiler it will most 
likely work (substitute gfortran for your compiler, e.g. ifort.)

This package should run on Linux, Mac OS X, *BSD, Solaris, etc.

Compile and run DISORT by executing the run_disort_unit_tests.sh script:
 ./run_disort_unit_tests.sh

--------------------------------------
DISORT code style guidelines

Currently the DISORT code is written in Fortran 77, so we will try to adhere to
Fortran 77 syntax. However, as of DISORTv4.0, we will no longer require the use
of 6 letter names, but will allow for longer variable names, and the underscore
character. Also, comments made using the "!" character will be allowed. While no
rule is absolute, the following is a list of guidelines for the DISORT code base:

1. Capital letters should be used everywhere (e.g. variable, function, and
subroutine names, statements) except for comments.
2. Underscore character is permitted.
3. Indentation: two spaces. Ideally, if DISORT was 'flatter', we could use more
spaces, but due to Fortran 77 width constraints we will use 2 spaces.
4. In a subroutine/program, statements should thus start in the "6th column",
and then indentation will be at the 8th column, 10th column, etc.
5. Prefer "ENDDO","ENDIF", "ELSEIF" over "END DO", "END IF" and "ELSE IF".
6. No common blocks or include statements.
7. Use of "GOTO" statements should be avoided where possible.
8. Use the "c" character for comments (1st column) and the "&" character for
continuation (5th column).
9. Keeping in line with its development, the DISORT code base should be kept as
general purpose as possible, and not modified so as to target a particular user
application.
10. Efficiency should be pursued, but should not necessarily trump readability.
11. Indent comments at the same column as the rest of the code.
12. Use less than 80 characters per line.
13. Avoid use of assumed-size arrays, i.e. ARRAY(A,B,*) should ideally be fully
declared so there is no assumed-size, that is ARRAY(A,B,C).
14. Prefer "IF(..) THEN \ FOO \ ENDIF" over single-line "IF(..) FOO" statements,
since they are more readable.

In the future, after upgrading to Fortran 90 we can start indenting from "column
1", and then we can use "modules" for passing variables around and to determine
precision.

--------------------------------------
The standard DISORT README file:

README:  DISORT version 2.0 (beta release)

This new version of DISORT is being released as a beta version.  While we have
every confidence that it is working correctly, and while it passes all the test
problems, there may still be subtle problems that we would appreciate hearing
about (our e-mail addresses are in DISORT.doc).

The major changes from version 1.2 are:

* Nakajima intensity corrections, allowing use of many fewer streams (NSTR) to
obtain the same intensity accuracy

* a proper lower-boundary bidirectional reflectance option

In DISORTv4.0, most of the LINPACK routines are replaced by their LAPACK 
alternatives to improve the speed.
--------------------------------------
The standard DISORT RDIMACH Readme file:

{R,D,I}1MACH:  The routines we needed but hated
W. Wiscombe (wiscombe@gsfc.nasa.gov)
July 1998

The machine-constant routines R1MACH, D1MACH, I1MACH caused more 
problems for me and for users of my programs than any others.  Their 
functions were simple, but people just had a hard time getting the 
versions distributed on netlib (and which I re-distributed) to work 
correctly.

At this point in time, it no longer makes sense to distribute or 
use these routines.  Fortran-90 contains intrinsic functions which 
contain all the functionality of R1MACH and D1MACH, and almost all of 
I1MACH.  Eric Grosse of Bell Labs has been kind enough to provide versions 
of {R,D,I}1MACH which use these new F-90 intrinsic functions.  I have 
slightly edited his routines.  The package is called RDI1MACH.f and 
is self-documenting.

Fortran-90 compilers have matured on most platforms and we 
highly recommend buying/using them (see http://www.fortran.com/fortran/).
There are even some free Fortran-90 subset compilers (http://www.lahey.com/).
Soon, Fortran-77 compilers may no longer be supported.  Fortran-90 
compilers can compile any Fortran-77 programs or routines, so once they 
became reliable it is inevitable that support for f77 compilers will wither.

Since Fortran-90 is entirely backward compatible with Fortran-77, 
you need not use any other feature of Fortran-90; you can just use 
RDI1MACH.f inside an old Fortran-77 program.  Remember that the .f
extension only means the file is in fixed source form, not that it is
pure Fortran-77.  In fact, any .f file can contain f90 constructs.

Those without access to Fortran-90 compilers can obtain the old 
versions of {R,D,I}1MACH at http://www.netlib.org/.
