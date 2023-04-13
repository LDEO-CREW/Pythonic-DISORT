#!/bin/sh
#
# This script compiles and executes DISORT
#

rm -f a.out
cat DISOTESTAUX.f DISORT.f BDREF.f DISOBRDF.f ERRPACK.f LINPAK.f LAPACK.f RDI1MACH.f > code.f
#gfortran -O0 -g -fcheck=all -fdump-core -ffpe-trap=invalid,zero,overflow,underflow,denormal -Wall rte_driver.f90 code.f 
#gfortran -O0 -g -fcheck=all -fdump-core -ffpe-trap=invalid,zero,overflow -Wall rte_driver.f90 code.f 
gfortran -O3 -g -fcheck=all -fdump-core -fbounds-check -Wall disotest.f90 code.f -o disort4_unit_tests.exe
chmod u+x ./disort4_unit_tests.exe
time ./disort4_unit_tests.exe
#rm -f code.f
