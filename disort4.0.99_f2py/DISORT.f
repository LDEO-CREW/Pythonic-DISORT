c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
c $Rev: 90 $ $Date: 2017-11-30 20:01:24 -0500 (Thu, 30 Nov 2017) $
c FORTRAN 77
c ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      SUBROUTINE DISORT( MAXCLY, MAXMOM, MAXCMU, 
     &                   MAXUMU, MAXPHI, MAXULV,
     &                   USRANG, USRTAU, IBCND, ONLYFL, PRNT,
     &                   PLANK, LAMBER, DELTAMPLUS, DO_PSEUDO_SPHERE,
     &                   DTAUC, SSALB, PMOM, TEMPER, WVNMLO, WVNMHI,
     &                   UTAU, UMU0, PHI0, UMU, PHI, FBEAM,
     &                   FISOT, ALBEDO, BTEMP, TTEMP, TEMIS,
     &                   EARTH_RADIUS, H_LYR, 
     &                   RHOQ, RHOU, RHO_ACCURATE, BEMST, EMUST,
     &                   ACCUR,  HEADER,
     &                   RFLDIR, RFLDN, FLUP, DFDT, UAVG, UU,
     &                   ALBMED, TRNMED )    

c *******************************************************************
c       Plane-parallel discrete ordinates radiative transfer program
c             ( see DISORT.doc for complete documentation )
c *******************************************************************
c
c +------------------------------------------------------------------+
c  Calling Tree (omitting calls to ERRMSG):
c  (routines in parentheses are not in this file)
c
c  DISORT-+-(R1MACH)
c         +-SLFTST-+-(TSTBAD)
c         +-ZEROIT
c         +-CHEKIN-+-(WRTBAD)
c         |        +-(WRTDIM)
c         |        +-DREF
c         +-ZEROAL
c         +-SETDIS-+-QGAUSN-+-(D1MACH)
c         +-PRTINP
c         +-ALBTRN-+-LEPOLY
c         |        +-ZEROIT
c         |        +-SOLEIG-+-ASYMTX-+-(D1MACH)
c         |        +-TERPEV
c         |        +-SETMTX-+-ZEROIT
c         |        +-(SGBCO)
c         |        +-SOLVE1-+-ZEROIT
c         |        |        +-(SGBSL)
c         |        +-ALTRIN
c         |        +-SPALTR
c         |        +-PRALTR
c         +-PLKAVG-+-(R1MACH)
c         |        +-ZEROIT
c         +-SOLEIG-+-ASYMTX-+-(D1MACH)
c         +-UPBEAM-+-(DGETRF) (version 3)
c         |        +-(DGETRS) (version 3)
c         +-UPISOT-+-(SGECO)
c         |        +-(SGESL)
c         +-TERPEV
c         +-TERPSO
c         +-SETMTX-+-ZEROIT
c         +-SOLVE0-+-ZEROIT
c         |        +-(SGBTRF)
c         |        +-(SGBTRS)
c         +-FLUXES--ZEROIT
c         +-ZEROIT
c         +-USRINT
c         +-CMPINT
c         +-PRAVIN
c         +-ZEROIT
c         +-RATIO--(R1MACH)
c         +-INTCOR-+-SINSCA
c         |        +-SECSCA-+-XIFUNC
c         +-INTCOR_BEAM_REFLEC (version 3)     
c         +-PRTINT    
c                   
c *** Intrinsic Functions used in DISORT package which take
c     non-negligible amount of time:
c
c    EXP :  Called by- ALBTRN, ALTRIN, CMPINT, FLUXES, SETDIS,
c                      SETMTX, SPALTR, USRINT, PLKAVG
c
c    SQRT : Called by- ASYMTX, SOLEIG
c
c +-------------------------------------------------------------------+
c
c  Index conventions (for all DO-loops and all variable descriptions):
c
c     IU     :  for user polar angles
c
c  IQ,JQ,KQ  :  for computational polar angles ('quadrature angles')
c
c   IQ/2     :  for half the computational polar angles (just the ones
c               in either 0-90 degrees, or 90-180 degrees)
c
c     J      :  for user azimuthal angles
c
c     K,L    :  for Legendre expansion coefficients or, alternatively,
c               subscripts of associated Legendre polynomials
c
c     LU     :  for user levels
c
c     LC     :  for computational layers (each having a different
c               single-scatter albedo and/or phase function)
c
c    LEV     :  for computational levels
c
c    MAZIM   :  for azimuthal components in Fourier cosine expansion
c               of intensity and phase function
c
c +------------------------------------------------------------------+
c
c               I N T E R N A L    V A R I A B L E S
c
c   AMB(IQ/2,IQ/2)    First matrix factor in reduced eigenvalue problem
c                     of Eqs. SS(12), STWJ(8E), STWL(23f)
c                     (used only in SOLEIG)
c
c   APB(IQ/2,IQ/2)    Second matrix factor in reduced eigenvalue problem
c                     of Eqs. SS(12), STWJ(8E), STWL(23f)
c                     (used only in SOLEIG)
c
c   ARRAY(IQ,IQ)      Scratch matrix for SOLEIG and UPISOT
c                     (see each subroutine for definition)
c
c   B()               Right-hand side vector of Eq. SC(5) going into
c                     SOLVE0,1;  returns as solution vector
c                     vector  L, the constants of integration
c
c   BDR(IQ/2,0:IQ/2)  Bottom-boundary bidirectional reflectivity for a
c                     given azimuthal component.  First index always
c                     refers to a computational angle.  Second index:
c                     if zero, refers to incident beam angle UMU0;
c                     if non-zero, refers to a computational angle.
c
c   BEM(IQ/2)         Bottom-boundary directional emissivity at compu-
c                     tational angles.
c
c   BPLANK            Intensity emitted from bottom boundary
c
c   CBAND()           Matrix of left-hand side of the linear system
c                     Eq. SC(5), scaled by Eq. SC(12);  in banded
c                     form required by LAPACK/LINPACK solution routines
c
c   CC(IQ,IQ)         C-sub-IJ in Eq. SS(5)
c
c   CMU(IQ)           Computational polar angles (Gaussian)
c
c   CWT(IQ)           Quadrature weights corresponding to CMU
c
c   CORINT            When set TRUE, correct intensities for
c                     delta-scaling effects (see Nakajima and Tanaka,
c                     1988). When FALSE, intensities are not corrected.
c                     In general, CORINT should be set true when beam
c                     source is present (FBEAM is not zero) and DELTAM
c                     is TRUE in a problem including scattering.
c                     However, execution is faster when CORINT is FALSE,
c                     and intensities outside the aureole may still be
c                     accurate enough.  When CORINT is TRUE, it is
c                     important to have a sufficiently high order of
c                     Legendre approximation of the phase function. This
c                     is because the intensities are corrected by
c                     calculating the single-scattered radiation, for
c                     which an adequate representation of the phase
c                     function is crucial.  In case of a low order
c                     Legendre approximation of an otherwise highly
c                     anisotropic phase function, the intensities might
c                     actually be more accurate when CORINT is FALSE.
c                     When only fluxes are calculated (ONLYFL is TRUE),
c                     or there is no beam source (FBEAM=0.0), or there
c                     is no scattering (SSALB=0.0 for all layers) CORINT
c                     is set FALSE by the code.
c
c   DELM0             Kronecker delta, delta-sub-M0, where M = MAZIM
c                     is the number of the Fourier component in the
c                     azimuth cosine expansion
c
c   DELTAM            TRUE,  use delta-M method ( see Wiscombe, 1977 );
c                     FALSE, do not use delta-M method. In general, for
c                     a given number of streams, intensities and
c                     fluxes will be more accurate for phase functions
c                     with a large forward peak if DELTAM is set true.
c                     Intensities close to the forward scattering
c                     direction are often less accurate, however, when
c                     the delta-M method is applied. The intensity
c                     correction of Nakajima and Tanaka is used to
c                     improve the accuracy of the intensities.
c
c  DELTAMPLUS         use new delta-M plus method
c
c
c
c   DITHER            Small quantity subtracted from single-scattering
c                     albedos of unity, in order to avoid using special
c                     case formulas;  prevents an eigenvalue of exactly
c                     zero from occurring, which would cause an
c                     immediate overflow
c
c   DTAUCP(LC)        Computational-layer optical depths (delta-M-scaled
c                     if DELTAM = TRUE, otherwise equal to DTAUC)
c
c   EMU(IU)           Bottom-boundary directional emissivity at user
c                     angles.
c
c   EVAL(IQ)          Temporary storage for eigenvalues of Eq. SS(12)
c
c   EVECC(IQ,IQ)      Complete eigenvectors of SS(7) on return from
c                     SOLEIG; stored permanently in  GC
c
c   EXPBEA(LC)        Transmission of direct beam in delta-M optical
c                     depth coordinates
c
c   FLYR(LC)          Separated fraction in delta-M method
c
c   GL(K,LC)          Phase function Legendre polynomial expansion
c                     coefficients, calculated from PMOM by
c                     including single-scattering albedo, factor
c                     2K+1, and (if DELTAM=TRUE) the delta-M
c                     scaling
c
c   GC(IQ,IQ,LC)      Eigenvectors at polar quadrature angles,
c                     g  in Eq. SC(1)
c
c   GU(IU,IQ,LC)      Eigenvectors interpolated to user polar angles
c                     ( g  in Eqs. SC(3) and S1(8-9), i.e.
c                       G without the L factor )
c
c   IPVT(LC*IQ)       Integer vector of pivot indices for LAPACK/LINPACK
c                     routines
c
c   KK(IQ,LC)         Eigenvalues of coeff. matrix in Eq. SS(7)
c
c   KCONV             Counter in azimuth convergence test
c
c   LAYRU(LU)         Computational layer in which user output level
c                     UTAU(LU) is located
c
c   LL(IQ,LC)         Constants of integration L in Eq. SC(1),
c                     obtained by solving scaled version of Eq. SC(5)
c
c   LYRCUT            TRUE, radiation is assumed zero below layer
c                     NCUT because of almost complete absorption
c
c   NAZ               Number of azimuthal components considered
c
c   NCUT              Computational layer number in which absorption
c                     optical depth first exceeds ABSCUT
c
c   OPRIM(LC)         Single scattering albedo after delta-M scaling
c
c   PASS1             TRUE on first entry, FALSE thereafter
c
c   PKAG(0:LC)        Integrated Planck function for internal emission
c
c   PRNTU0(L)         logical flag to trigger printing of azimuthally-
c                     averaged intensities:
c                       L    quantities printed
c                      --    ------------------
c                       1    azimuthally-averaged intensities at user
c                               levels and computational polar angles
c                       2    azimuthally-averaged intensities at user
c                               levels and user polar angles
c
c   PSI0(IQ)          Sum just after square bracket in  Eq. SD(9)
c
c   PSI1(IQ)          Sum in  Eq. STWL(31d)
c
c   RMU(IU,0:IQ)      Bottom-boundary bidirectional reflectivity for a
c                     given azimuthal component.  First index always
c                     refers to a user angle.  Second index:
c                     if zero, refers to incident beam angle UMU0;
c                     if non-zero, refers to a computational angle.
c
c   SQT(k)            Square root of k (used only in LEPOLY for
c                     computing associated Legendre polynomials)
c
c   TAUC(0:LC)        Cumulative optical depth (un-delta-M-scaled)
c
c   TAUCPR(0:LC)      Cumulative optical depth (delta-M-scaled if
c                     DELTAM = TRUE, otherwise equal to TAUC)
c
c   TPLANK            Intensity emitted from top boundary
c
c   UUM(IU,LU)        Expansion coefficients when the intensity
c                     (u-super-M) is expanded in Fourier cosine series
c                     in azimuth angle
c
c   U0C(IQ,LU)        Azimuthally-averaged intensity at quadrature
c                     angle
c
c   U0U(IU,LU)        If ONLYFL = FALSE, azimuthally-averaged intensity
c                     at user angles and user levels
c
c                     If ONLYFL = TRUE and MAXUMU.GE.NSTR,
c                     azimuthally-averaged intensity at computational
c                     (Gaussian quadrature) angles and user levels;
c                     the corresponding quadrature angle cosines are
c                     returned in UMU.  If MAXUMU.LT.NSTR, U0U will be
c                     zeroed, and UMU, NUMU will not be set.
c
c   UTAUPR(LU)        Optical depths of user output levels in delta-M
c                     coordinates;  equal to  UTAU(LU) if no delta-M
c
c   WK()              scratch array
c
c   XR0(LC)           X-sub-zero in expansion of thermal source func-
c                     tion preceding Eq. SS(14)(has no mu-dependence);
c                     b-sub-zero in Eq. STWL(24d)
c
c   XR1(LC)           X-sub-one in expansion of thermal source func-
c                     tion; see  Eqs. SS(14-16); b-sub-one in STWL(24d)
c
c   YLM0(L,1)         Normalized associated Legendre polynomial
c                     of subscript L at the beam angle (not saved
c                     as function of superscipt M)
c
c   YLMC(L,IQ)        Normalized associated Legendre polynomial
c                     of subscript L at the computational angles
c                     (not saved as function of superscipt M)
c
c   YLMU(L,IU)        Normalized associated Legendre polynomial
c                     of subscript L at the user angles
c                     (not saved as function of superscipt M)
c
c   Z()               scratch array used in SOLVE0, ALBTRN to solve
c                     a linear system for the constants of integration
c
c   Z0(IQ)            Solution vectors Z-sub-zero of Eq. SS(16)
c
c   Z0U(IU,LC)        Z-sub-zero in Eq. SS(16) interpolated to user
c                     angles from an equation derived from SS(16)
c
c   Z1(IQ)            Solution vectors Z-sub-one  of Eq. SS(16)
c
c   Z1U(IU,LC)        Z-sub-one in Eq. SS(16) interpolated to user
c                     angles from an equation derived from SS(16)
c
c   ZBEAM(IU,LC)      Particular solution for beam source
c
c   ZJ(IQ)            Right-hand side vector  X-sub-zero in
c                     Eq. SS(19), also the solution vector
c                     Z-sub-zero after solving that system
c
c   ZZ(IQ,LC)         Permanent storage for the beam source vectors ZJ
c
c   ZPLK0(IQ,LC)      Permanent storage for the thermal source
c                     vectors  Z0  obtained by solving  Eq. SS(16)
c
c   ZPLK1(IQ,LC)      Permanent storage for the thermal source
c                     vectors  Z1  obtained by solving  Eq. SS(16)
c
c +-------------------------------------------------------------------+
c
c  LOCAL SYMBOLIC DIMENSIONS (have big effect on storage requirements):
c
c       MAXCLY  = Max no. of computational layers
c       MAXULV  = Max no. of output levels
c       MAXCMU  = Max no. of computation polar angles
c       MAXUMU  = Max no. of output polar angles
c       MAXPHI  = Max no. of output azimuthal angles
c       MAXSQT  = Max no. of square roots of integers (for LEPOLY)
c +-------------------------------------------------------------------+

!      USE PARAMETERS 
      INTEGER   MAXCLY, MAXMOM, MAXPHI, MAXULV, MAXUMU, MAXCMU
c     ..
c     .. Scalar Arguments ..
      CHARACTER HEADER*127
      LOGICAL   LAMBER, ONLYFL, PLANK, USRANG, USRTAU
      INTEGER   IBCND, NLYR, NMOM, NPHI, NSTR, NTAU, NUMU
      REAL      ACCUR, ALBEDO, BTEMP, FBEAM, FISOT, PHI0, TEMIS, TTEMP,
     &          UMU0, WVNMHI, WVNMLO

c     ..
c     .. Array Arguments ..
      LOGICAL   PRNT( 5 )
      REAL      ALBMED( MAXUMU ), DFDT( MAXULV ), DTAUC( MAXCLY ),
     &          FLUP( MAXULV ), PHI( MAXPHI ), 
     &          PMOM( 0:MAXMOM, MAXCLY ),
     &          RFLDIR( MAXULV ), RFLDN( MAXULV ), SSALB( MAXCLY ),
     &          TEMPER( 0:MAXCLY ), TRNMED( MAXUMU ), UAVG( MAXULV ),
     &          UMU( MAXUMU ), UTAU( MAXULV ),
     &          UU( MAXUMU, MAXULV, MAXPHI )

c     ..
c     .. Local Scalars ..
      LOGICAL   COMPAR, CORINT, DELTAM, LYRCUT, PASS1 
      INTEGER   IQ, IU, J, KCONV, L, LC, LEV, LU, MAZIM, NAZ, NCOL,
     &          NCOS, NCUT, NN
      REAL      ANGCOS, AZERR, AZTERM, BPLANK, COSPHI, DELM0, DITHER,
     &          DUM, PI, RPD, SGN, TPLANK

Cf2py intent(in, out) RFLDIR
Cf2py intent(in, out) RFLDN
Cf2py intent(in, out) FLUP
Cf2py intent(in, out) DFDT
Cf2py intent(in, out) UAVG
Cf2py intent(in, out) UU
Cf2py intent(in, out) ALBMED
Cf2py intent(in, out) TRNMED

c     ..
c     .. Local Arrays ..
      LOGICAL   PRNTU0( 2 )
      INTEGER   IPVT(MAXCMU*MAXCLY ), LAYRU( MAXULV )
      REAL      AMB(MAXCMU/2,MAXCMU/2), APB(MAXCMU/2,MAXCMU/2), 
     &          ARRAY(MAXCMU,MAXCMU),
     &          B( MAXCMU*MAXCLY ), BDR(MAXCMU/2,0:MAXCMU/2), 
     &          BEM(MAXCMU/2),
     &          CBAND( 9*(MAXCMU/2)-2, MAXCMU*MAXCLY ), 
     &          CC(MAXCMU, MAXCMU),
     &          CMU( MAXCMU), CWT( MAXCMU ), DTAUCP( MAXCLY ),
     &          EMU( MAXUMU ), EVAL( MAXCMU/2 ), EVECC(MAXCMU, MAXCMU),
     &          EXPBEA( 0:MAXCLY ), FLDIR( MAXULV), FLDN( MAXULV),
     &          FLYR( MAXCLY ), GC( MAXCMU, MAXCMU, MAXCLY ),
     &          GL( 0:MAXCMU, MAXCLY ), GU( MAXUMU, MAXCMU, MAXCLY ),
     &          KK(MAXCMU, MAXCLY), LL(MAXCMU, MAXCLY), OPRIM( MAXCLY),
     &          PHASA( MAXCLY ), PHAST( MAXCLY ), PHASM( MAXCLY ),
     &          PHIRAD( MAXPHI ), PKAG( 0:MAXCLY ), PSI0( MAXCMU ),
     &          PSI1( MAXCMU ), RMU( MAXUMU, 0:MAXCMU/2 ), 
     &          SQT(2*MAXCMU),
     &          TAUC( 0:MAXCLY), TAUCPR( 0:MAXCLY), 
     &          U0C(MAXCMU, MAXULV),
     &          U0U( MAXUMU, MAXULV ), UTAUPR( MAXULV ),
     &          UUM( MAXUMU, MAXULV ), WK( MAXCMU ), XR0( MAXCLY ),
     &          XR1( MAXCLY ), YLM0( 0:MAXCMU,1), 
     &          YLMC(0:MAXCMU,MAXCMU),
     &          YLMU( 0:MAXCMU, MAXUMU ), Z(MAXCMU*MAXCLY), 
     &          Z0( MAXCMU ),
     &          Z0U(MAXUMU, MAXCLY), Z1(MAXCMU), Z1U(MAXUMU, MAXCLY),
     &          ZBEAM( MAXUMU, MAXCLY )
      REAL      ZJ( MAXCMU), ZPLK0( MAXCMU, MAXCLY),
     &          ZPLK1( MAXCMU, MAXCLY ), ZZ( MAXCMU, MAXCLY )
      DOUBLE PRECISION AAD(MAXCMU/2,MAXCMU/2), EVALD(MAXCMU/2), 
     &          EVECCD(MAXCMU/2,MAXCMU/2), WKD( MAXCMU )

c     ..
c     .. Version 3 .. 
      REAL      RHOQ(MAXCMU/2, 0:MAXCMU/2, 0:(MAXCMU-1)), 
     &          RHOU(MAXUMU,   0:MAXCMU/2, 0:(MAXCMU-1)),
     &          EMUST(MAXUMU), BEMST(MAXCMU/2)
      REAL      UMU0DI, UMU0SQ, DENOM
      REAL      RHO_ACCURATE(MAXUMU,MAXPHI)  
      REAL      EIGEN_MAT(MAXCMU/2, MAXCMU/2 )  

c     .. Version 3: spherical correction ..
      REAL      EARTH_RADIUS, H_LYR(0:MAXCLY)
      REAL      UMU0L(MAXCLY)
      LOGICAL   DO_PSEUDO_SPHERE

c     .. Version 3: deltam plus
      LOGICAL   DELTAMPLUS
c     ..
c     .. External Functions ..
      REAL      PLKAVG, R1MACH, RATIO
      EXTERNAL  PLKAVG, R1MACH, RATIO

c     ..
c     .. External Subroutines ..
      EXTERNAL  ALBTRN, CHEKIN, CMPINT, FLUXES, INTCOR, LEPOLY, PRAVIN,
     &          PRTINP, PRTINT, SETDIS, SETMTX, SLFTST, SOLEIG, SOLVE0,
     &          SURFAC, TERPEV, TERPSO, UPBEAM, UPISOT, USRINT, ZEROAL,
     &          ZEROIT

c     ..
c     .. Intrinsic Functions ..
      INTRINSIC ABS, ASIN, COS, FLOAT, LEN, MAX, SQRT

c     ..
c     .. SAVE and DATA Statements ..
      SAVE      DITHER, PASS1, PI, RPD
      DATA      PASS1 / .TRUE. /, PRNTU0 / 2*.FALSE. /

      NLYR = MAXCLY
      NMOM = MAXMOM
      NSTR = MAXCMU
      NUMU = MAXUMU
      NPHI = MAXPHI
      NTAU = MAXULV

      IF( DELTAMPLUS .AND. ( NMOM .LT. NSTR + 1 ) ) THEN
         CALL ERRMSG( 'To use DeltaM+, NMOM must be  '//
     &                'at least equal to NSTR + 1, '//
     &                ' increase NMOM. ',
     &                 .True.)
      ENDIF 
      
      IF (DELTAMPLUS) THEN
        DELTAM = .FALSE.
        CORINT = .FALSE.
      ELSE
        DELTAM = .TRUE.
        CORINT = .TRUE.
      END IF

      IF (DELTAMPLUS) THEN 
        DO L = 1, NLYR 
        
           IF ( PMOM(NSTR,L) .LT. 1.0e-4 ) THEN
             DELTAMPLUS = .FALSE.
             DELTAM     = .TRUE.
             CORINT = .TRUE.
           ENDIF
         
           IF ( PMOM(NSTR+1,L) .LT. PMOM(NSTR,L)*0.7 ) THEN
             DELTAMPLUS = .FALSE.
             DELTAM     = .TRUE.
             CORINT = .TRUE.
           ENDIF
           
        ENDDO
      ENDIF
      
C      CORINT = .TRUE.
      
C      DELTAM = .FALSE.
C      DELTAMPLUS = .FALSE.
C      CORINT = .FALSE.

c     ** Disable these at your own risk
c      DELTAM = .FALSE.
c      CORINT = .FALSE.

c     ** For debugging purposes only
c      PASS1 = .FALSE.

      IF( IBCND.EQ.1 .AND. ONLYFL ) THEN 
         CALL ERRMSG( 'ONLYFL must be .FALSE. for  '//
     &                'IBCND = 1 shortcut. '//
     &                'Please see DISORT.txt file'//
     &                ' for more details about this option.',
     &                 .True.)
      ENDIF
      
      IF( PASS1 ) THEN

        PI     = 2.*ASIN( 1.0 )
        DITHER = 10.*R1MACH( 4 )

c       ** Must dither more on high (>= 14-digit) precision machine
        IF( DITHER.LT.1.E-10 ) THEN
          DITHER = 10.*DITHER
        ENDIF

        RPD  = PI / 180.0


c       ** Set input values for self-test
c       ** Ensure that SLFTST sets all print flags off
        COMPAR = .FALSE.

        CALL SLFTST( CORINT, ACCUR, ALBEDO, BTEMP, DELTAM, DTAUC( 1 ),
     &               FBEAM, FISOT, IBCND, LAMBER, NLYR, PLANK, NPHI,
     &               NUMU, NSTR, NTAU, ONLYFL, PHI( 1 ), PHI0, NMOM,
     &               PMOM( 0,1 ), PRNT, PRNTU0, SSALB( 1 ), TEMIS,
     &               TEMPER( 0 ), TTEMP, UMU( 1 ), USRANG, USRTAU,
     &               UTAU( 1 ), UMU0, WVNMHI, WVNMLO, COMPAR, DUM,
     &               DUM, DUM, DUM, DO_PSEUDO_SPHERE, DELTAMPLUS )

      ENDIF

   20 CONTINUE

      IF( .NOT.PASS1 .AND. LEN( HEADER ).NE.0 ) THEN
         WRITE( *,'(//,1X,100(''*''),/,A,/,1X,100(''*''))' )
     &    ' DISORT: '//HEADER
      ENDIF

c     ** Calculate cumulative optical depth and dither single-scatter albedo
c     ** to improve numerical behavior of eigenvalue/vector computation.
      CALL ZEROIT( TAUC, MAXCLY + 1 )
      DO 30 LC = 1, NLYR
        IF( SSALB( LC ).EQ.1.0 ) THEN
          SSALB( LC ) = 1.0 - DITHER
        ENDIF

        TAUC( LC ) = TAUC( LC - 1 ) + DTAUC( LC )
   30 CONTINUE

c     ** Check input dimensions and variables
      CALL CHEKIN( NLYR, DTAUC, SSALB, NMOM, PMOM, TEMPER, WVNMLO,
     &             WVNMHI, USRTAU, NTAU, UTAU, NSTR, USRANG,
     &             NUMU, UMU, NPHI, PHI, IBCND, FBEAM, UMU0,
     &             PHI0, FISOT, LAMBER, ALBEDO, BTEMP, TTEMP,
     &             TEMIS, PLANK, ONLYFL, DELTAM, CORINT, ACCUR,
     &             TAUC, MAXCLY, MAXULV, MAXUMU, MAXPHI, MAXMOM,
     &             MAXCMU )

c     ** Zero internal and output arrays
      CALL  ZEROAL( MAXCLY, EXPBEA(1), FLYR, OPRIM, PHASA, PHAST, PHASM,
     &                     TAUCPR(1), XR0, XR1,
     &              MAXCMU, CMU, CWT, PSI0, PSI1, WK, Z0, Z1, ZJ,
     &              MAXCMU+1, YLM0,
     &              NSTR**2, ARRAY, CC, EVECC,
     &              (NSTR+1)*NLYR, GL,
     &              (MAXCMU+1)*MAXCMU, YLMC,
     &              (MAXCMU+1)*MAXUMU, YLMU,
     &              MAXCMU*MAXCLY, KK, LL, ZZ, ZPLK0, ZPLK1,
     &              MAXCMU**2*MAXCLY, GC,
     &              MAXULV, LAYRU, UTAUPR,
     &              MAXUMU*MAXCMU*MAXCLY, GU,
     &              MAXUMU*MAXCLY, Z0U, Z1U, ZBEAM,
     &              NSTR/2, EVAL,
     &              (NSTR/2)**2, AMB, APB,
     &              NSTR*NLYR, IPVT, Z,
     &              MAXULV, RFLDIR, RFLDN, FLUP, UAVG, DFDT,
     &              MAXUMU, ALBMED, TRNMED,
     &              MAXUMU*MAXULV, U0U,
     &              MAXUMU*MAXULV*MAXPHI, UU )

c     ** Perform various setup operations
      CALL SETDIS( CMU, CWT, DELTAM, DTAUC, DTAUCP, EXPBEA, FBEAM, FLYR,
     &             GL, IBCND, LAYRU, LYRCUT, MAXMOM, MAXUMU, MAXCMU,
     &             NCUT, NLYR, NTAU, NN, NSTR, PLANK, NUMU, ONLYFL,
     &             CORINT, OPRIM, PMOM, SSALB, TAUC, TAUCPR, UTAU,
     &             UTAUPR, UMU, UMU0, USRTAU, USRANG, NAZZ, MI, SQT,
     &             DO_PSEUDO_SPHERE, EARTH_RADIUS, H_LYR, UMU0L,
     &             DELTAMPLUS )

c     ** Print input information
      IF( PRNT( 1 ) ) THEN
        CALL PRTINP( NLYR, DTAUC, DTAUCP, SSALB, NMOM, PMOM, TEMPER,
     &               WVNMLO, WVNMHI, NTAU, UTAU, NSTR, NUMU, UMU,
     &               NPHI, PHI, IBCND, FBEAM, UMU0, PHI0, FISOT,
     &               LAMBER, ALBEDO, BTEMP, TTEMP, TEMIS, DELTAM,
     &               PLANK, ONLYFL, CORINT, ACCUR, FLYR, LYRCUT,
     &               OPRIM, TAUC, TAUCPR, MAXMOM, PRNT( 5 ),
     &               DO_PSEUDO_SPHERE, H_LYR, DELTAMPLUS)
      ENDIF

c     ** Handle special case for getting albedo and transmissivity of medium for
c     ** many beam angles at once.
      IF( IBCND.EQ.1 ) THEN
        CALL ALBTRN( ALBEDO, AMB, APB, ARRAY, B, BDR, CBAND, 
     &                CC, CMU, CWT, DTAUCP, EVAL, EVECC, 
     &                GL, GC, GU, IPVT, KK,
     &                LL, NLYR, NN, NSTR, NUMU, PRNT, TAUCPR, UMU, U0U,
     &                WK, YLMC, YLMU, Z, AAD, EVALD, EVECCD, WKD,
     &                MAXUMU, MAXCMU,MAXUMU, SQT, ALBMED,
     &                TRNMED )
        RETURN
      ENDIF

c     ** Calculate Planck functions
      IF( .NOT.PLANK ) THEN
        BPLANK = 0.0
        TPLANK = 0.0
        CALL ZEROIT( PKAG,  MAXCLY + 1 )
      ELSE
        TPLANK = TEMIS*PLKAVG( WVNMLO, WVNMHI, TTEMP )
        BPLANK =       PLKAVG( WVNMLO, WVNMHI, BTEMP )
        DO 40 LEV = 0, NLYR
          PKAG( LEV ) = PLKAVG( WVNMLO, WVNMHI, TEMPER( LEV ) )
   40   CONTINUE
      ENDIF

c ========  BEGIN LOOP TO SUM AZIMUTHAL COMPONENTS OF INTENSITY  =======
c           (EQ STWJ 5, STWL 6)

      KCONV  = 0
      NAZ    = NSTR - 1

c     ** Azimuth-independent case
      IF( FBEAM.EQ.0.0 .OR. ABS(1.-UMU0).LT.1.E-5 .OR. ONLYFL .OR.
     &   ( NUMU.EQ.1 .AND. ABS(1.-UMU(1)).LT.1.E-5 ) .OR.
     &   ( NUMU.EQ.1 .AND. ABS(1.+UMU(1)).LT.1.E-5 ) .OR.
     &   ( NUMU.EQ.2 .AND. ABS(1.+UMU(1)).LT.1.E-5 .AND.
     &     ABS(1.-UMU(NUMU)).LT.1.E-5 ) ) THEN 
!     &     ABS(1.-UMU(2)).LT.1.E-5 ) ) THEN
        NAZ = 0
      ENDIF


      DO 180 MAZIM = 0, NAZ

        IF( MAZIM.GT.0 ) THEN
          DELM0  = 0.0
        ELSEIF( MAZIM.EQ.0 ) THEN
          DELM0  = 1.0
        ENDIF

c       ** Get normalized associated Legendre polynomials for
c       ** (a) incident beam angle cosine
c       ** (b) computational and user polar angle cosines

        IF( FBEAM.GT.0.0 ) THEN
          NCOS   = 1
          ANGCOS = -UMU0
          !CALL LEPOLY( NCOS, MAZIM, MAXCMU, NSTR-1, ANGCOS, SQT, YLM0 )
          CALL LEPOLY0( MAZIM, MAXCMU, NSTR-1, ANGCOS, SQT, YLM0 )

        ENDIF


        IF( .NOT.ONLYFL .AND. USRANG ) THEN
          CALL LEPOLY( NUMU, MAZIM, MAXCMU, NSTR-1, UMU, SQT, YLMU )
        ENDIF

        CALL LEPOLY( NN, MAZIM, MAXCMU, NSTR-1, CMU, SQT, YLMC )

c       ** Get normalized associated Legendre polys.  with negative arguments
c       ** from those with positive arguments; Dave/Armstrong Eq. (15),
c       ** STWL(59).
        SGN  = -1.0
        DO 70 L = MAZIM, NSTR - 1
          SGN  = -SGN
          DO 60 IQ = NN + 1, NSTR
            YLMC( L, IQ ) = SGN*YLMC( L, IQ - NN )
   60     CONTINUE

   70   CONTINUE

c       ** Specify users bottom reflectivity and emissivity properties
        IF( .NOT.LYRCUT ) THEN
          CALL SURFAC( ALBEDO, FBEAM, LAMBER, MAXCMU/2, MAZIM,
     &                 MAXUMU, NN, NUMU, ONLYFL, UMU, 
     &                 USRANG, BDR, EMU, BEM, RMU,
     &                 RHOQ, RHOU, EMUST, BEMST, NAZZ )
        ENDIF


c ===================  BEGIN LOOP ON COMPUTATIONAL LAYERS  =============
        DO 80 LC = 1, NCUT

c         ** Solve eigenfunction problem in Eq. STWJ(8B), STWL(23f); return
c         ** eigenvalues and eigenvectors
c         ** Version 3: update SOLEIG argument: EIGEN_MAT
          CALL SOLEIG( AMB, APB, EIGEN_MAT, CMU, CWT, GL( 0,LC ), 
     &                 MAZIM, MAXCMU, NN, NSTR,YLMC,CC, EVECC, EVAL,
     &                 KK( 1,LC ), GC( 1,1,LC ), AAD, EVECCD, EVALD,
     &                 WKD )


c         ** Version 3: fix singularity problem in particular solution
          IF( FBEAM.GT. 0.0) THEN
            UMU0DI = UMU0L(LC)
            UMU0SQ = UMU0L(LC)*UMU0L(LC)
            DO 85 IQ = 1,NN
              DENOM = 1. - UMU0SQ*KK(IQ,LC)*KK(IQ,LC)
!              IF( ABS(DENOM).LT. 1.E-4 ) THEN
              IF( ABS(DENOM).LT. 10.0*DITHER ) THEN
                UMU0DI = 0.999 * UMU0L(LC)
              ENDIF
 85         CONTINUE
          ENDIF

c         ** Calculate particular solutions of Eq. SS(18), STWL(24a) for
c         ** incident beam source.
c         ** Version 3: upgraded subroutine UPBEAM to use reduced order matrix
c         ** and LAPACK.
          IF( FBEAM.GT.0.0 ) THEN
            CALL UPBEAM( EIGEN_MAT, APB, AMB,
     &                   NN, MAZIM, 
     &                   MAXCMU, CMU, DELM0, FBEAM, 
     &                   GL(0,LC), YLM0, YLMC, PI, UMU0DI,
     o                   ZJ, ZZ( 1,LC) )
          ENDIF

c         ** Calculate particular solutions of Eq. SS(15), STWL(25) for
c         ** thermal emission source.
          IF( PLANK .AND. MAZIM.EQ.0 ) THEN
            XR1( LC ) = 0.0

            IF( DTAUCP( LC ).GT.0.0 ) THEN
              XR1( LC ) = ( PKAG( LC ) - PKAG( LC-1 ) ) / DTAUCP( LC )
            ENDIF

            XR0( LC ) = PKAG( LC-1 ) - XR1( LC )*TAUCPR( LC-1 )

            CALL UPISOT( ARRAY, CC, CMU, IPVT, MAXCMU, NN, NSTR,
     &                   OPRIM( LC ), WK, XR0( LC ), XR1( LC ),
     &                   Z0, Z1, ZPLK0( 1,LC ), ZPLK1( 1,LC ) )
          ENDIF

          IF( .NOT.ONLYFL .AND. USRANG ) THEN

c           ** Interpolate eigenvectors to user angles
            CALL TERPEV( CWT, EVECC, 
     &                   GL( 0,LC ), GU( 1,1,LC ), MAZIM,
     &                   MAXCMU, MAXUMU, NN, NSTR, NUMU, WK, YLMC,
     &                   YLMU )

c           ** Interpolate source terms to user angles
            CALL TERPSO( CWT, DELM0, FBEAM, GL( 0,LC ), MAZIM, MAXCMU,
     &                   PLANK, NUMU, NSTR, OPRIM( LC ),PI,YLM0,
     &                   YLMC, YLMU, PSI0, PSI1, XR0( LC ),
     &                   XR1( LC ), Z0, Z1, ZJ, ZBEAM( 1,LC ),
     &                   Z0U( 1,LC ), Z1U( 1,LC ) )
          ENDIF

   80   CONTINUE
c ===================  END LOOP ON COMPUTATIONAL LAYERS  ===============


c       ** Set coefficient matrix of equations combining boundary and layer
c       ** interface conditions.
        CALL SETMTX( BDR, CBAND, CMU, CWT, DELM0, DTAUCP, GC, KK,
     &               LAMBER, LYRCUT, MAXCMU, NCOL, NCUT,
     &               NLYR, NN, NSTR, TAUCPR, WK )

c       ** Solve for constants of integration in homogeneous solution (general
c       ** boundary conditions).
c       ** Version 3 upgrade: LAPACK solver
        CALL SOLVE0( B, BDR, BEM, BPLANK, CBAND, CMU, CWT, EXPBEA,
     &               FBEAM, FISOT, IPVT, LAMBER, LL, LYRCUT, MAZIM, 
     &               MAXCMU, NCOL, NCUT, NN, NSTR, NLYR, PI,
     &               TPLANK, TAUCPR, UMU0, ZZ, ZPLK0, ZPLK1 )

c       ** Compute upward and downward fluxes
        IF( MAZIM.EQ.0 ) THEN
          CALL FLUXES( CMU, CWT, FBEAM, GC, KK, LAYRU, LL, LYRCUT,
     &                  MAXULV, MAXCMU, MAXULV, NCUT, NN, NSTR, NTAU,
     &                  PI, PRNT, PRNTU0( 1 ), SSALB, TAUCPR, UMU0,
     &                  UTAU, UTAUPR, XR0, XR1, ZZ, ZPLK0, ZPLK1,
     &                  DFDT, FLUP, FLDN, FLDIR, RFLDIR, RFLDN, UAVG,
     &                  U0C, UMU0L, DTAUCP, DTAUC, TAUC )
        ENDIF

        IF( ONLYFL ) THEN
          IF( MAXUMU.GE.NSTR ) THEN
c           ** Save azimuthally averaged intensities at quadrature angles
            DO 100 LU = 1, NTAU
              DO 90 IQ = 1, NSTR
                U0U( IQ, LU ) = U0C( IQ, LU )
   90         CONTINUE
  100       CONTINUE
          ENDIF
          GOTO 190
        ENDIF

        CALL ZEROIT( UUM, MAXUMU*MAXULV )

        IF( USRANG ) THEN
c         ** Compute azimuthal intensity components at user angles
          CALL USRINT( BPLANK, CMU, CWT, DELM0, DTAUCP, EMU, EXPBEA,
     &                 FBEAM, FISOT, GC, GU, KK, LAMBER, LAYRU, LL,
     &                 LYRCUT, MAZIM, MAXCMU, MAXULV,MAXUMU, NCUT, NLYR,
     &                 NN, NSTR, PLANK, NUMU, NTAU, PI, RMU, TAUCPR,
     &                 TPLANK, UMU, UMU0, UTAUPR, WK, ZBEAM, Z0U, Z1U,
     &                 ZZ, ZPLK0, ZPLK1, UUM, UMU0L )
        ELSE
c         ** Compute azimuthal intensity components at quadrature angles

          CALL CMPINT( FBEAM, GC, KK, LAYRU, LL, LYRCUT, MAZIM, MAXCMU,
     &                 MAXULV, MAXUMU, NCUT, NN, NSTR, PLANK, NTAU,
     &                 TAUCPR, UTAUPR, ZZ, ZPLK0, ZPLK1, UUM,
     &                 UMU0L, DTAUCP)
        ENDIF

        IF( MAZIM.EQ.0 ) THEN
c         ** Save azimuthally averaged intensities
          DO 130 LU = 1, NTAU
            DO 120 IU = 1, NUMU
              U0U( IU, LU ) = UUM( IU, LU )
              DO 110 J = 1, NPHI
                UU( IU, LU, J ) = UUM( IU, LU )
  110         CONTINUE
  120       CONTINUE
  130     CONTINUE

c         ** Print azimuthally averaged intensities at user angles
          IF( PRNTU0( 2 ) ) THEN
            CALL PRAVIN( UMU, NUMU, MAXUMU, UTAU, NTAU, U0U )
          ENDIF

          IF( NAZ.GT.0 ) THEN
            CALL ZEROIT( PHIRAD, MAXPHI )
            DO 140 J = 1, NPHI
              PHIRAD( J ) = RPD*( PHI( J ) - PHI0 )
  140       CONTINUE
          ENDIF

        ELSE
c         ** Increment intensity by current azimuthal component (Fourier cosine
c         ** series); Eq SD(2), STWL(6).

          AZERR  = 0.0

          DO 170 J = 1, NPHI

            COSPHI = COS( MAZIM*PHIRAD( J ) )

            DO 160 LU = 1, NTAU
              DO 150 IU = 1, NUMU
                AZTERM          = UUM( IU, LU )*COSPHI
                UU( IU, LU, J ) = UU( IU, LU, J ) + AZTERM
                AZERR  = MAX( AZERR,
     &                        RATIO( ABS(AZTERM), ABS(UU(IU,LU,J)) ) )
  150           CONTINUE
  160         CONTINUE
  170       CONTINUE

            IF( AZERR.LE.ACCUR ) THEN
              KCONV  = KCONV + 1
            ENDIF

            IF( KCONV.GE.2 ) THEN
              GOTO  190
            ENDIF

          ENDIF

  180   CONTINUE

  190 CONTINUE
c ===================  END LOOP ON AZIMUTHAL COMPONENTS  ===============

c     ** Version 3 debug block
c      PRINT*,
c      PRINT*, UMU0,NUMU,NSTR,NPHI
c      PRINT*, (RHOU(NUMU/2+1,0,IU),IU=0,NSTR-1)
c      PRINT*, (PHIRAD(IU)*180./3.141592653,IU=1,NPHI)
c      PRINT*, 
c      PRINT*,BDR_CORR(UMU0,UMU(NUMU/2+5),PHI(1)-PHI0,
c     &                NSTR,RHOU(NUMU/2+5,0,0:NSTR-1))

c     ** Apply Nakajima/Tanaka intensity corrections
      IF( CORINT )
     &  CALL INTCOR( DITHER, FBEAM, FLYR, LAYRU, LYRCUT, MAXMOM,
     &               MAXULV, MAXUMU, NMOM, NCUT, NPHI, NSTR, NTAU,
     &               NUMU, OPRIM, PHASA, PHAST, PHASM, PHIRAD, PI,
     &               RPD, PMOM, SSALB, DTAUC, TAUC, TAUCPR, UMU,
     &               UMU0, UTAU, UTAUPR, UU, DELTAMPLUS )

c     ** Version 3: add new SS correction of beam reflection
      IF(.NOT. LAMBER .AND. .NOT. LYRCUT .AND. FBEAM .NE. 0.0 ) THEN
        CALL INTCOR_BEAM_REFLEC( NUMU, UMU, NPHI, PHI, PHI0, UMU0,
     &                           MAXUMU, MAXPHI, MAXUMU, MAXULV,
     &                           MI, NAZZ, NSTR, NTAU, NCUT, FBEAM,
     &                           TAUCPR, UTAUPR, RHOU, RHO_ACCURATE,   
     &                           LAYRU, LYRCUT, PI, UU )
      ENDIF

c     ** Print intensities
      IF( PRNT( 3 ) .AND. .NOT.ONLYFL ) THEN
        CALL PRTINT( UU, UTAU, NTAU, UMU, NUMU, PHI, NPHI, MAXULV,
     &               MAXUMU )
      ENDIF

      IF( PASS1 ) THEN
c       ** Compare test case results with correct answers and abort if bad
         COMPAR = .TRUE.
         CALL SLFTST( CORINT, ACCUR, ALBEDO, BTEMP, DELTAM, DTAUC( 1 ),
     &                FBEAM, FISOT, IBCND, LAMBER, NLYR, PLANK, NPHI,
     &                NUMU, NSTR, NTAU, ONLYFL, PHI( 1 ), PHI0, NMOM,
     &                PMOM( 0,1 ), PRNT, PRNTU0, SSALB( 1 ), TEMIS,
     &                TEMPER( 0 ), TTEMP, UMU( 1 ), USRANG, USRTAU,
     &                UTAU( 1 ), UMU0, WVNMHI, WVNMLO, COMPAR,
     &                FLUP( 1 ), RFLDIR( 1 ), RFLDN( 1 ), UU( 1,1,1 ),
     &                DO_PSEUDO_SPHERE, DELTAMPLUS )

         PASS1 = .FALSE.
         GOTO 20
      ENDIF

      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE ASYMTX( AA, EVEC, EVAL, M, IA, IEVEC, IER, WKD, AAD,
     &                   EVECD, EVALD )

c    =======  D O U B L E    P R E C I S I O N    V E R S I O N  ======
c
c       Solves eigenfunction problem for real asymmetric matrix
c       for which it is known a priori that the eigenvalues are real.
c
c       This is an adaptation of a subroutine EIGRF in the IMSL
c       library to use real instead of complex arithmetic, accounting
c       for the known fact that the eigenvalues and eigenvectors in
c       the discrete ordinate solution are real.  Other changes include
c       putting all the called subroutines in-line, deleting the
c       performance index calculation, updating many DO-loops
c       to Fortran77, and in calculating the machine precision
c       TOL instead of specifying it in a data statement.
c
c       EIGRF is based primarily on EISPACK routines.  The matrix is
c       first balanced using the Parlett-Reinsch algorithm.  Then
c       the Martin-Wilkinson algorithm is applied.
c
c       There is a statement 'J  = WKD( I )' that converts a double
c       precision variable to an integer variable, that seems dangerous
c       to us in principle, but seems to work fine in practice.
c
c       References:
c          Dongarra, J. and C. Moler, EISPACK -- A Package for Solving
c             Matrix Eigenvalue Problems, in Cowell, ed., 1984:
c             Sources and Development of Mathematical Software,
c             Prentice-Hall, Englewood Cliffs, NJ
c         Parlett and Reinsch, 1969: Balancing a Matrix for Calculation
c             of Eigenvalues and Eigenvectors, Num. Math. 13, 293-304
c         Wilkinson, J., 1965: The Algebraic Eigenvalue Problem,
c             Clarendon Press, Oxford
c
c
c   I N P U T    V A R I A B L E S:
c
c       AA    :  input asymmetric matrix, destroyed after solved
c
c        M    :  order of  AA
c
c       IA    :  first dimension of  AA
c
c    IEVEC    :  first dimension of  EVEC
c
c
c   O U T P U T    V A R I A B L E S:
c
c       EVEC  :  (unnormalized) eigenvectors of  AA
c                   ( column J corresponds to EVAL(J) )
c
c       EVAL  :  (unordered) eigenvalues of AA ( dimension at least M )
c
c       IER   :  if .NE. 0, signals that EVAL(IER) failed to converge;
c                   in that case eigenvalues IER+1,IER+2,...,M  are
c                   correct but eigenvalues 1,...,IER are set to zero.
c
c
c   S C R A T C H   V A R I A B L E S:
c
c       WKD   :  work area ( dimension at least 2*M )
c       AAD   :  double precision stand-in for AA
c       EVECD :  double precision stand-in for EVEC
c       EVALD :  double precision stand-in for EVAL
c
c   Called by- SOLEIG
c   Calls- D1MACH, ERRMSG
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   IA, IER, IEVEC, M
c     ..
c     .. Array Arguments ..

      REAL      AA( IA, M ), EVAL( M ), EVEC( IEVEC, M )
      DOUBLE PRECISION AAD( IA, M ), EVALD( M ), EVECD( IA, M ),
     &                 WKD( * )
c     ..
c     .. Local Scalars ..

      LOGICAL   NOCONV, NOTLAS
      INTEGER   I, II, IN, J, K, KA, KKK, L, LB, LLL, N, N1, N2
      DOUBLE PRECISION C1, C2, C3, C4, C5, C6, COL, DISCRI, F, G, H,
     &                 ONE, P, Q, R, REPL, RNORM, ROW, S, SCALE, SGN, T,
     &                 TOL, UU, VV, W, X, Y, Z, ZERO
c     ..
c     .. External Functions ..

      DOUBLE PRECISION D1MACH
      EXTERNAL  D1MACH
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, MIN, SIGN, SQRT
c     ..
      DATA      C1 / 0.4375D0 / , C2 / 0.5D0 / , C3 / 0.75D0 / ,
     &          C4 / 0.95D0 / , C5 / 16.D0 / , C6 / 256.D0 / ,
     &          ZERO / 0.D0 / , ONE / 1.D0 /

      P = 0d0
      Q = 0d0
      R = 0d0

      IER  = 0
      TOL  = D1MACH( 4 )
      LB   = 0

      IF( M.LT.1 .OR. IA.LT.M .OR. IEVEC.LT.M )
     &    CALL ERRMSG( 'ASYMTX--bad input variable(s)', .TRUE. )


c                           ** Handle 1x1 and 2x2 special cases
      IF( M.EQ.1 ) THEN

         EVAL( 1 )   = AA( 1,1 )
         EVEC( 1,1 ) = 1.0
         RETURN

      ELSE IF( M.EQ.2 ) THEN

         DISCRI = ( AA( 1,1 ) - AA( 2,2 ) )**2 + 4.*AA( 1,2 )*AA( 2,1 )

         IF( DISCRI .LT. 0.0 )
     &       CALL ERRMSG( 'ASYMTX--complex evals in 2x2 case',.TRUE. )

         SGN  = ONE

         IF( AA( 1,1 ) .LT. AA( 2,2 ) ) SGN  = - ONE

         EVAL( 1 ) = REAL( 0.5*( AA( 1,1 ) + AA( 2,2 ) + 
     &                     SGN*SQRT( DISCRI ) ) )
         EVAL( 2 ) = REAL( 0.5*( AA( 1,1 ) + AA( 2,2 ) - 
     &                     SGN*SQRT( DISCRI ) ) )
         EVEC( 1,1 ) = 1.0
         EVEC( 2,2 ) = 1.0

         IF( AA( 1,1 ) .EQ. AA( 2,2 ) .AND.
     &       ( AA( 2,1 ).EQ.0.0 .OR. AA( 1,2 ).EQ.0.0 ) ) THEN

            RNORM = ABS( AA( 1,1 ) ) + ABS( AA( 1,2 ) ) +
     &              ABS( AA( 2,1 ) ) + ABS( AA( 2,2 ) )
            W     = TOL * RNORM
            EVEC( 2,1 ) =   REAL( AA( 2,1 ) / W )
            EVEC( 1,2 ) = - REAL( AA( 1,2 ) / W )

         ELSE

            EVEC( 2,1 ) = AA( 2,1 ) / ( EVAL( 1 ) - AA( 2,2 ) )
            EVEC( 1,2 ) = AA( 1,2 ) / ( EVAL( 2 ) - AA( 1,1 ) )

         END IF

         RETURN

      END IF

c                               ** Convert single-prec. matrix to double
      DO 20 J = 1, M

         DO 10 K = 1, M
            AAD( J,K ) = AA( J,K )
   10    CONTINUE

   20 CONTINUE

c                                ** Initialize output variables
      IER  = 0

      DO 40 I = 1, M

         EVALD( I ) = ZERO

         DO 30 J = 1, M
            EVECD( I, J ) = ZERO
   30    CONTINUE

         EVECD( I, I ) = ONE

   40 CONTINUE

c                  ** Balance the input matrix and reduce its norm by
c                  ** diagonal similarity transformation stored in WK;
c                  ** then search for rows isolating an eigenvalue
c                  ** and push them down
      RNORM  = ZERO
      L  = 1
      K  = M

   50 CONTINUE
      KKK  = K

      DO 90 J = KKK, 1, -1

         ROW  = ZERO

         DO 60 I = 1, K
            IF( I.NE.J ) ROW  = ROW + ABS( AAD( J,I ) )
   60    CONTINUE

         IF( ROW.EQ.ZERO ) THEN

            WKD( K ) = J

            IF( J.NE.K ) THEN

               DO 70 I = 1, K
                  REPL        = AAD( I, J )
                  AAD( I, J ) = AAD( I, K )
                  AAD( I, K ) = REPL
   70          CONTINUE

               DO 80 I = L, M
                  REPL        = AAD( J, I )
                  AAD( J, I ) = AAD( K, I )
                  AAD( K, I ) = REPL
   80          CONTINUE

            END IF

            K  = K - 1
            GO TO  50

         END IF

   90 CONTINUE
c                                ** Search for columns isolating an
c                                ** eigenvalue and push them left
  100 CONTINUE
      LLL  = L

      DO 140 J = LLL, K

         COL  = ZERO

         DO 110 I = L, K
            IF( I.NE.J ) COL  = COL + ABS( AAD( I,J ) )
  110    CONTINUE

         IF( COL.EQ.ZERO ) THEN

            WKD( L ) = J

            IF( J.NE.L ) THEN

               DO 120 I = 1, K
                  REPL        = AAD( I, J )
                  AAD( I, J ) = AAD( I, L )
                  AAD( I, L ) = REPL
  120          CONTINUE

               DO 130 I = L, M
                  REPL        = AAD( J, I )
                  AAD( J, I ) = AAD( L, I )
                  AAD( L, I ) = REPL
  130          CONTINUE

            END IF

            L  = L + 1
            GO TO  100

         END IF

  140 CONTINUE

c                           ** Balance the submatrix in rows L through K
      DO 150 I = L, K
         WKD( I ) = ONE
  150 CONTINUE

  160 CONTINUE
      NOCONV = .FALSE.

      DO 220 I = L, K

         COL  = ZERO
         ROW  = ZERO

         DO 170 J = L, K

            IF( J.NE.I ) THEN
               COL  = COL + ABS( AAD( J,I ) )
               ROW  = ROW + ABS( AAD( I,J ) )
            END IF

  170    CONTINUE

         F  = ONE
         G  = ROW / C5
         H  = COL + ROW

  180    CONTINUE
         IF( COL.LT.G ) THEN

            F    = F*C5
            COL  = COL*C6
            GO TO  180

         END IF

         G  = ROW*C5

  190    CONTINUE
         IF( COL.GE.G ) THEN

            F    = F / C5
            COL  = COL / C6
            GO TO  190

         END IF
c                                                ** Now balance
         IF( ( COL + ROW ) / F.LT.C4*H ) THEN

            WKD( I ) = WKD( I )*F
            NOCONV = .TRUE.

            DO 200 J = L, M
               AAD( I, J ) = AAD( I, J ) / F
  200       CONTINUE

            DO 210 J = 1, K
               AAD( J, I ) = AAD( J, I )*F
  210       CONTINUE

         END IF

  220 CONTINUE


      IF( NOCONV ) GO TO  160
c                                   ** Is A already in Hessenberg form?
      IF( K-1 .LT. L+1 ) GO TO  370

c                                   ** Transfer A to a Hessenberg form
      DO 310 N = L + 1, K - 1

         H  = ZERO
         WKD( N + M ) = ZERO
         SCALE  = ZERO
c                                                 ** Scale column
         DO 230 I = N, K
            SCALE  = SCALE + ABS( AAD( I,N - 1 ) )
  230    CONTINUE

         IF( SCALE.NE.ZERO ) THEN

            DO 240 I = K, N, -1
               WKD( I + M ) = AAD( I, N - 1 ) / SCALE
               H  = H + WKD( I + M )**2
  240       CONTINUE

            G    = - SIGN( SQRT( H ), WKD( N + M ) )
            H    = H - WKD( N + M )*G
            WKD( N + M ) = WKD( N + M ) - G
c                                            ** Form (I-(U*UT)/H)*A
            DO 270 J = N, M

               F  = ZERO

               DO 250 I = K, N, -1
                  F  = F + WKD( I + M )*AAD( I, J )
  250          CONTINUE

               DO 260 I = N, K
                  AAD( I, J ) = AAD( I, J ) - WKD( I + M )*F / H
  260          CONTINUE

  270       CONTINUE
c                                    ** Form (I-(U*UT)/H)*A*(I-(U*UT)/H)
            DO 300 I = 1, K

               F  = ZERO

               DO 280 J = K, N, -1
                  F  = F + WKD( J + M )*AAD( I, J )
  280          CONTINUE

               DO 290 J = N, K
                  AAD( I, J ) = AAD( I, J ) - WKD( J + M )*F / H
  290          CONTINUE

  300       CONTINUE

            WKD( N + M ) = SCALE*WKD( N + M )
            AAD( N, N - 1 ) = SCALE*G

         END IF

  310 CONTINUE


      DO 360 N = K - 2, L, -1

         N1   = N + 1
         N2   = N + 2
         F  = AAD( N + 1, N )

         IF( F.NE.ZERO ) THEN

            F  = F*WKD( N + 1 + M )

            DO 320 I = N + 2, K
               WKD( I + M ) = AAD( I, N )
  320       CONTINUE

            IF( N + 1.LE.K ) THEN

               DO 350 J = 1, M

                  G  = ZERO

                  DO 330 I = N + 1, K
                     G  = G + WKD( I + M )*EVECD( I, J )
  330             CONTINUE

                  G  = G / F

                  DO 340 I = N + 1, K
                     EVECD( I, J ) = EVECD( I, J ) + G*WKD( I + M )
  340             CONTINUE

  350          CONTINUE

            END IF

         END IF

  360 CONTINUE


  370 CONTINUE

      N  = 1

      DO 390 I = 1, M

         DO 380 J = N, M
            RNORM  = RNORM + ABS( AAD( I,J ) )
  380    CONTINUE

         N  = I

         IF( I.LT.L .OR. I.GT.K ) EVALD( I ) = AAD( I, I )

  390 CONTINUE

      N  = K
      T  = ZERO

c                                      ** Search for next eigenvalues
  400 CONTINUE
      IF( N.LT.L ) GO TO  550

      IN  = 0
      N1  = N - 1
      N2  = N - 2
c                          ** Look for single small sub-diagonal element
  410 CONTINUE

      DO 420 I = L, N

         LB  = N + L - I

         IF( LB.EQ.L ) GO TO  430

         S  = ABS( AAD( LB - 1,LB - 1 ) ) + ABS( AAD( LB,LB ) )

         IF( S.EQ.ZERO ) S  = RNORM

         IF( ABS( AAD( LB, LB-1 ) ).LE. TOL*S ) GO TO  430

  420 CONTINUE


  430 CONTINUE
      X  = AAD( N, N )

      IF( LB.EQ.N ) THEN
c                                        ** One eigenvalue found
         AAD( N, N ) = X + T
         EVALD( N ) = AAD( N, N )
         N  = N1
         GO TO  400

      END IF

      Y  = AAD( N1, N1 )
      W  = AAD( N, N1 )*AAD( N1, N )

      IF( LB.EQ.N1 ) THEN
c                                        ** Two eigenvalues found
         P  = ( Y - X )*C2
         Q  = P**2 + W
         Z  = SQRT( ABS( Q ) )
         AAD( N, N ) = X + T
         X  = AAD( N, N )
         AAD( N1, N1 ) = Y + T
c                                        ** Real pair
         Z  = P + SIGN( Z, P )
         EVALD( N1 ) = X + Z
         EVALD( N ) = EVALD( N1 )

         IF( Z.NE.ZERO ) EVALD( N ) = X - W / Z

         X  = AAD( N, N1 )
c                                  ** Employ scale factor in case
c                                  ** X and Z are very small
         R  = SQRT( X*X + Z*Z )
         P  = X / R
         Q  = Z / R
c                                             ** Row modification
         DO 440 J = N1, M
            Z  = AAD( N1, J )
            AAD( N1, J ) = Q*Z + P*AAD( N, J )
            AAD( N, J ) = Q*AAD( N, J ) - P*Z
  440    CONTINUE
c                                             ** Column modification
         DO 450 I = 1, N
            Z  = AAD( I, N1 )
            AAD( I, N1 ) = Q*Z + P*AAD( I, N )
            AAD( I, N ) = Q*AAD( I, N ) - P*Z
  450    CONTINUE
c                                          ** Accumulate transformations
         DO 460 I = L, K
            Z  = EVECD( I, N1 )
            EVECD( I, N1 ) = Q*Z + P*EVECD( I, N )
            EVECD( I, N ) = Q*EVECD( I, N ) - P*Z
  460    CONTINUE

         N  = N2
         GO TO  400

      END IF


      IF( IN.EQ.30 ) THEN

c                    ** No convergence after 30 iterations; set error
c                    ** indicator to the index of the current eigenvalue
         IER  = N
         GO TO  700

      END IF
c                                                  ** Form shift
      IF( IN.EQ.10 .OR. IN.EQ.20 ) THEN

         T  = T + X

         DO 470 I = L, N
            AAD( I, I ) = AAD( I, I ) - X
  470    CONTINUE

         S  = ABS( AAD( N,N1 ) ) + ABS( AAD( N1,N2 ) )
         X  = C3*S
         Y  = X
         W  = -C1*S**2

      END IF


      IN  = IN + 1

c                ** Look for two consecutive small sub-diagonal elements

      DO 480 J = LB, N2
         I  = N2 + LB - J
         Z  = AAD( I, I )
         R  = X - Z
         S  = Y - Z
         P  = ( R*S - W ) / AAD( I + 1, I ) + AAD( I, I + 1 )
         Q  = AAD( I + 1, I + 1 ) - Z - R - S
         R  = AAD( I + 2, I + 1 )
         S  = ABS( P ) + ABS( Q ) + ABS( R )
         P  = P / S
         Q  = Q / S
         R  = R / S

         IF( I.EQ.LB ) GO TO  490

         UU   = ABS( AAD( I, I-1 ) )*( ABS( Q ) + ABS( R ) )
         VV   = ABS( P ) * ( ABS( AAD( I-1, I-1 ) ) + ABS( Z ) +
     &                       ABS( AAD( I+1, I+1 ) ) )

         IF( UU .LE. TOL*VV ) GO TO  490

  480 CONTINUE

  490 CONTINUE
      AAD( I+2, I ) = ZERO

      DO 500 J = I + 3, N
         AAD( J, J - 2 ) = ZERO
         AAD( J, J - 3 ) = ZERO
  500 CONTINUE

c             ** Double QR step involving rows K to N and columns M to N

      DO 540 KA = I, N1

         NOTLAS = KA.NE.N1

         IF( KA.EQ.I ) THEN

            S  = SIGN( SQRT( P*P + Q*Q + R*R ), P )

            IF( LB.NE.I ) AAD( KA, KA - 1 ) = -AAD( KA, KA - 1 )

         ELSE

            P  = AAD( KA, KA - 1 )
            Q  = AAD( KA + 1, KA - 1 )
            R  = ZERO

            IF( NOTLAS ) R  = AAD( KA + 2, KA - 1 )

            X  = ABS( P ) + ABS( Q ) + ABS( R )

            IF( X.EQ.ZERO ) GO TO  540

            P  = P / X
            Q  = Q / X
            R  = R / X
            S  = SIGN( SQRT( P*P + Q*Q + R*R ), P )
            AAD( KA, KA - 1 ) = -S*X

         END IF

         P  = P + S
         X  = P / S
         Y  = Q / S
         Z  = R / S
         Q  = Q / P
         R  = R / P
c                                              ** Row modification
         DO 510 J = KA, M

            P  = AAD( KA, J ) + Q*AAD( KA + 1, J )

            IF( NOTLAS ) THEN

               P  = P + R*AAD( KA + 2, J )
               AAD( KA + 2, J ) = AAD( KA + 2, J ) - P*Z

            END IF

            AAD( KA + 1, J ) = AAD( KA + 1, J ) - P*Y
            AAD( KA, J ) = AAD( KA, J ) - P*X

  510    CONTINUE
c                                                 ** Column modification
         DO 520 II = 1, MIN( N, KA + 3 )

            P  = X*AAD( II, KA ) + Y*AAD( II, KA + 1 )

            IF( NOTLAS ) THEN

               P  = P + Z*AAD( II, KA + 2 )
               AAD( II, KA + 2 ) = AAD( II, KA + 2 ) - P*R

            END IF

            AAD( II, KA + 1 ) = AAD( II, KA + 1 ) - P*Q
            AAD( II, KA ) = AAD( II, KA ) - P

  520    CONTINUE
c                                          ** Accumulate transformations
         DO 530 II = L, K

            P  = X*EVECD( II, KA ) + Y*EVECD( II, KA + 1 )

            IF( NOTLAS ) THEN

               P  = P + Z*EVECD( II, KA + 2 )
               EVECD( II, KA + 2 ) = EVECD( II, KA + 2 ) - P*R

            END IF

            EVECD( II, KA + 1 ) = EVECD( II, KA + 1 ) - P*Q
            EVECD( II, KA ) = EVECD( II, KA ) - P

  530    CONTINUE

  540 CONTINUE

      GO TO  410
c                     ** All evals found, now backsubstitute real vector
  550 CONTINUE

      IF( RNORM.NE.ZERO ) THEN

         DO 580 N = M, 1, -1

            N2   = N
            AAD( N, N ) = ONE

            DO 570 I = N - 1, 1, -1

               W  = AAD( I, I ) - EVALD( N )

               IF( W.EQ.ZERO ) W  = TOL*RNORM

               R  = AAD( I, N )

               DO 560 J = N2, N - 1
                  R  = R + AAD( I, J )*AAD( J, N )
  560          CONTINUE

               AAD( I, N ) = -R / W
               N2   = I

  570       CONTINUE

  580    CONTINUE
c                      ** End backsubstitution vectors of isolated evals
         DO 600 I = 1, M

            IF( I.LT.L .OR. I.GT.K ) THEN

               DO 590 J = I, M
                  EVECD( I, J ) = AAD( I, J )
  590          CONTINUE

            END IF

  600    CONTINUE
c                                   ** Multiply by transformation matrix
         IF( K.NE.0 ) THEN

            DO 630 J = M, L, -1

               DO 620 I = L, K

                  Z  = ZERO

                  DO 610 N = L, MIN( J, K )
                     Z  = Z + EVECD( I, N )*AAD( N, J )
  610             CONTINUE

                  EVECD( I, J ) = Z

  620          CONTINUE

  630       CONTINUE

         END IF

      END IF


      DO 650 I = L, K

         DO 640 J = 1, M
            EVECD( I, J ) = EVECD( I, J ) * WKD( I )
  640    CONTINUE

  650 CONTINUE

c                           ** Interchange rows if permutations occurred
      DO 670 I = L-1, 1, -1

         J  =  INT( WKD( I ) )

         IF( I.NE.J ) THEN

            DO 660 N = 1, M
               REPL   = EVECD( I, N )
               EVECD( I, N ) = EVECD( J, N )
               EVECD( J, N ) = REPL
  660       CONTINUE

         END IF

  670 CONTINUE


      DO 690 I = K + 1, M

         J  = INT( WKD( I ) )

         IF( I.NE.J ) THEN

            DO 680 N = 1, M
               REPL   = EVECD( I, N )
               EVECD( I, N ) = EVECD( J, N )
               EVECD( J, N ) = REPL
  680       CONTINUE

         END IF

  690 CONTINUE

c                         ** Put results into output arrays
  700 CONTINUE

      DO 720 J = 1, M

         EVAL( J ) = REAL( EVALD( J ) )

         DO 710 K = 1, M
            EVEC( J, K ) = REAL( EVECD( J, K ) )
  710    CONTINUE

  720 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE CMPINT( FBEAM, GC, KK, LAYRU, LL, LYRCUT, MAZIM, MXCMU,
     &                   MXULV, MXUMU, NCUT, NN, NSTR, PLANK, NTAU,
     &                   TAUCPR, UTAUPR, ZZ, ZPLK0, ZPLK1, UUM,
     &                   UMU0L, DTAUCP )

c          Calculates the Fourier intensity components at the quadrature
c          angles for azimuthal expansion terms (MAZIM) in Eq. SD(2),
c          STWL(6)
c
c
c    I N P U T    V A R I A B L E S:
c
c       KK      :  Eigenvalues of coeff. matrix in Eq. SS(7), STWL(23b)
c
c       GC      :  Eigenvectors at polar quadrature angles in Eq. SC(1)
c
c       LL      :  Constants of integration in Eq. SC(1), obtained
c                  by solving scaled version of Eq. SC(5);
c                  exponential term of Eq. SC(12) not included
c
c       LYRCUT  :  Logical flag for truncation of computational layer
c
c       MAZIM   :  Order of azimuthal component
c
c       NCUT    :  Number of computational layer where absorption
c                  optical depth exceeds ABSCUT
c
c       NN      :  Order of double-Gauss quadrature (NSTR/2)
c
c       TAUCPR  :  Cumulative optical depth (delta-M-scaled)
c
c       UTAUPR  :  Optical depths of user output levels in delta-M
c                  coordinates;  equal to UTAU if no delta-M
c
c       ZZ      :  Beam source vectors in Eq. SS(19), STWL(24b)
c
c       ZPLK0   :  Thermal source vectors Z0, by solving Eq. SS(16),
c                  Y-sub-zero in STWL(26ab)
c
c       ZPLK1   :  Thermal source vectors Z1, by solving Eq. SS(16),
c                  Y-sub-one in STWL(26ab)
c
c       (Remainder are 'DISORT' input variables)
c
c
c    O U T P U T   V A R I A B L E S:
c
c       UUM     :  Fourier components of the intensity in Eq. SD(12)
c                    (at polar quadrature angles)
c
c
c    I N T E R N A L   V A R I A B L E S:
c
c       FACT    :  EXP( - UTAUPR / UMU0 )
c       ZINT    :  intensity of M=0 case, in Eq. SC(1)
c
c   Called by- DISORT
c +--------------------------------------------------------------------

c     .. Scalar Arguments ..

      LOGICAL   LYRCUT, PLANK
      INTEGER   MAZIM, MXCMU, MXULV, MXUMU, NCUT, NN, NSTR, NTAU
      REAL      FBEAM
      REAL      UMU0L(*), DTAUCP( * )
c     ..
c     .. Array Arguments ..

      INTEGER   LAYRU( * )
      REAL      GC( MXCMU, MXCMU, * ), KK( MXCMU, * ), LL( MXCMU, * ),
     &          TAUCPR( 0:* ), UTAUPR( MXULV ), UUM( MXUMU, MXULV ),
     &          ZPLK0( MXCMU, * ), ZPLK1( MXCMU, * ), ZZ( MXCMU, * )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, JQ, LU, LYU, LC
      REAL      ZINT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..

c                                       ** Loop over user levels
      DO 40 LU = 1, NTAU

         LYU  = LAYRU( LU )

         IF( LYRCUT .AND. LYU.GT.NCUT ) GO TO  40

         DO 30 IQ = 1, NSTR

            ZINT = 0.0

            DO 10 JQ = 1, NN
               ZINT = ZINT + GC( IQ, JQ, LYU ) * LL( JQ, LYU ) *
     &                       EXP( -KK( JQ,LYU )*
     &                     ( UTAUPR( LU ) - TAUCPR( LYU ) ) )
   10       CONTINUE

            DO 20 JQ = NN + 1, NSTR
               ZINT = ZINT + GC( IQ, JQ, LYU ) * LL( JQ, LYU ) *
     &                       EXP( -KK( JQ,LYU )*
     &                     ( UTAUPR( LU ) - TAUCPR( LYU-1 ) ) )
   20       CONTINUE

            UUM( IQ, LU ) = ZINT

c comment and upgrade pseudo spherical correction            
c            IF( FBEAM.GT.0.0 ) UUM( IQ, LU ) = ZINT +
c     &                         ZZ( IQ, LYU )*EXP( -UTAUPR( LU )/UMU0 )
            IF( FBEAM.GT.0.0 ) THEN
              UUM(IQ, LU) = ZZ( IQ, LYU )
              DO LC = 1, LYU-1
                UUM( IQ, LU ) = UUM( IQ, LU ) 
     &             * EXP(-DTAUCP(LC)/UMU0L(LC))
              ENDDO
              UUM(IQ, LU) = ZINT + UUM(IQ, LU)
     &             * EXP( ( TAUCPR(LYU-1) - UTAUPR(LU) ) / UMU0L(LYU) )
            ENDIF

            IF( PLANK .AND. MAZIM.EQ.0 )
     &          UUM( IQ, LU ) = UUM( IQ, LU ) + ZPLK0( IQ,LYU ) +
     &                          ZPLK1( IQ,LYU ) * UTAUPR( LU )
   30    CONTINUE

   40 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE FLUXES( CMU, CWT, FBEAM, GC, KK, LAYRU, LL, LYRCUT,
     &                   MAXULV, MXCMU, MXULV, NCUT, NN, NSTR, NTAU,
     &                   PI, PRNT, PRNTU0, SSALB, TAUCPR, UMU0, UTAU,
     &                   UTAUPR, XR0, XR1, ZZ, ZPLK0, ZPLK1, DFDT,
     &                   FLUP, FLDN, FLDIR, RFLDIR, RFLDN, UAVG, U0C,
     &                   UMU0L, DTAUCP, DTAUC, TAUC )

c       Calculates the radiative fluxes, mean intensity, and flux
c       derivative with respect to optical depth from the m=0 intensity
c       components (the azimuthally-averaged intensity)
c
c
c    I N P U T     V A R I A B L E S:
c
c       CMU      :  Abscissae for Gauss quadrature over angle cosine
c
c       CWT      :  Weights for Gauss quadrature over angle cosine
c
c       GC       :  Eigenvectors at polar quadrature angles, SC(1)
c
c       KK       :  Eigenvalues of coeff. matrix in Eq. SS(7), STWL(23b)
c
c       LAYRU    :  Layer number of user level UTAU
c
c       LL       :  Constants of integration in Eq. SC(1), obtained
c                   by solving scaled version of Eq. SC(5);
c                   exponential term of Eq. SC(12) not included
c
c       LYRCUT   :  Logical flag for truncation of comput. layer
c
c       NN       :  Order of double-Gauss quadrature (NSTR/2)
c
c       NCUT     :  Number of computational layer where absorption
c                   optical depth exceeds ABSCUT
c
c       PRNTU0   :  TRUE, print azimuthally-averaged intensity at
c                   quadrature angles
c
c       TAUCPR   :  Cumulative optical depth (delta-M-scaled)
c
c       UTAUPR   :  Optical depths of user output levels in delta-M
c                   coordinates;  equal to UTAU if no delta-M
c
c       XR0      :  Expansion of thermal source function in Eq. SS(14),
c                   STWL(24c)
c
c       XR1      :  Expansion of thermal source function Eq. SS(16),
c                   STWL(24c)
c
c       ZZ       :  Beam source vectors in Eq. SS(19), STWL(24b)
c
c       ZPLK0    :  Thermal source vectors Z0, by solving Eq. SS(16),
c                   Y0 in STWL(26b)
c
c       ZPLK1    :  Thermal source vectors Z1, by solving Eq. SS(16),
c                   Y1 in STWL(26a)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T     V A R I A B L E S:
c
c       U0C      :  Azimuthally averaged intensities
c                   ( at polar quadrature angles )
c
c       (RFLDIR, RFLDN, FLUP, DFDT, UAVG are DISORT output variables)
c
c
c    I N T E R N A L       V A R I A B L E S:
c
c       DIRINT   :  Direct intensity attenuated
c       FDNTOT   :  Total downward flux (direct + diffuse)
c       FLDIR    :  Direct-beam flux (delta-M scaled)
c       FLDN     :  Diffuse down-flux (delta-M scaled)
c       FNET     :  Net flux (total-down - diffuse-up)
c       FACT     :  EXP( - UTAUPR / UMU0 )
c       PLSORC   :  Planck source function (thermal)
c       ZINT     :  Intensity of m = 0 case, in Eq. SC(1)
c
c   Called by- DISORT
c   Calls- ZEROIT
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   LYRCUT, PRNTU0
      INTEGER   MAXULV, MXCMU, MXULV, NCUT, NN, NSTR, NTAU
      REAL      FBEAM, PI, UMU0, UMU0L( * )
c     ..
c     .. Array Arguments ..

      LOGICAL   PRNT( * )
      INTEGER   LAYRU( MXULV )
      REAL      CMU( MXCMU ), CWT( MXCMU ), DFDT( MAXULV ),
     &          FLDIR( MXULV ), FLDN( MXULV ), FLUP( MAXULV ),
     &          GC( MXCMU, MXCMU, * ), KK( MXCMU, * ), LL( MXCMU, * ),
     &          RFLDIR( MAXULV ), RFLDN( MAXULV ), SSALB( * ),
     &          TAUCPR( 0:* ), U0C( MXCMU, MXULV ), UAVG( MAXULV ),
     &          UTAU( MAXULV ), UTAUPR( MXULV ), XR0( * ), XR1( * ),
     &          ZPLK0( MXCMU, * ), ZPLK1( MXCMU, * ), ZZ( MXCMU, * ),
     &          DTAUCP( * ), DTAUC( * ), TAUC( 0:* ) 
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, JQ, LU, LYU, LC
      REAL      ANG1, ANG2, DIRINT, FACT, FDNTOT, FNET, PLSORC, ZINT
      REAL      FACT2
c     ..
c     .. External Subroutines ..

      EXTERNAL  ZEROIT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..


      IF( PRNT( 2 ) ) WRITE ( *, '(//,21X,A,/,2A,/,2A,/)' )
     &    '<----------------------- FLUXES ----------------------->',
     &    '   Optical  Compu    Downward    Downward    Downward     ',
     &    ' Upward                    Mean      Planck   d(Net Flux)',
     &    '     Depth  Layer      Direct     Diffuse       Total     ',
     &    'Diffuse         Net   Intensity      Source   / d(Op Dep)'

c                                        ** Zero DISORT output arrays
      CALL ZEROIT( U0C, MXULV*MXCMU )
      CALL ZEROIT( FLDIR, MXULV )
      CALL ZEROIT( FLDN, MXULV )
      !CALL ZEROIT( FACT, 1 )
      FACT = 0.0

c                                        ** Loop over user levels
      DO 80 LU = 1, NTAU

         LYU  = LAYRU( LU )

         IF( LYRCUT .AND. LYU.GT.NCUT ) THEN
c                                                ** No radiation reaches
c                                                ** this level
            FDNTOT = 0.0
            FNET   = 0.0
            PLSORC = 0.0
            GO TO  70

         END IF


         IF( FBEAM.GT.0.0 ) THEN

c  comment code and add spherical correction
c            FACT         = EXP( -UTAUPR( LU ) / UMU0 )
c            DIRINT       = FBEAM*FACT
c            FLDIR( LU )  = UMU0*( FBEAM*FACT )
c            RFLDIR( LU ) = UMU0*FBEAM * EXP( -UTAU( LU ) / UMU0 )

c  condiser pseudo spherical correction            
            FACT  = 0.0
            FACT2 = 0.0
            DO LC = 1, LYU-1
              FACT = FACT - DTAUCP( LC ) / UMU0L(LC)
              FACT2 = FACT2 - DTAUC( LC ) / UMU0L(LC)
            ENDDO
            FACT = FACT - ( UTAUPR(LU) - TAUCPR(LYU-1) ) / UMU0L(LYU)
            FACT = EXP(FACT)
            FACT2= FACT2- ( UTAU(LU) - TAUC(LYU-1) ) / UMU0L(LYU)
            FACT2= EXP(FACT2)
            DIRINT       = FBEAM*FACT
            FLDIR( LU )  = UMU0 * ( FBEAM * FACT )
            RFLDIR( LU ) = UMU0 * ( FBEAM * FACT2 )

         ELSE

            DIRINT       = 0.0
            FLDIR( LU )  = 0.0
            RFLDIR( LU ) = 0.0

         END IF


         DO 30 IQ = 1, NN

            ZINT = 0.0

            DO 10 JQ = 1, NN
               ZINT = ZINT + GC( IQ, JQ, LYU )*LL( JQ, LYU )*
     &                EXP( -KK( JQ,LYU )*( UTAUPR( LU ) -
     &                TAUCPR( LYU ) ) )
   10       CONTINUE

            DO 20 JQ = NN + 1, NSTR
               ZINT = ZINT + GC( IQ, JQ, LYU )*LL( JQ, LYU )*
     &                EXP( -KK( JQ,LYU )*( UTAUPR( LU ) -
     &                TAUCPR( LYU-1 ) ) )
   20       CONTINUE

            U0C( IQ, LU ) = ZINT

            IF( FBEAM.GT.0.0 ) U0C( IQ, LU ) = ZINT + ZZ( IQ, LYU )*FACT

            U0C( IQ, LU ) = U0C( IQ, LU ) + ZPLK0( IQ,LYU ) +
     &                      ZPLK1( IQ,LYU )*UTAUPR( LU )
            UAVG( LU ) = UAVG( LU ) + CWT( NN + 1 - IQ )*U0C( IQ, LU )
            FLDN( LU ) = FLDN( LU ) + CWT( NN + 1 - IQ )*
     &                   CMU( NN + 1 - IQ )*U0C( IQ, LU )
   30    CONTINUE


         DO 60 IQ = NN + 1, NSTR

            ZINT = 0.0

            DO 40 JQ = 1, NN
               ZINT = ZINT + GC( IQ, JQ, LYU )*LL( JQ, LYU )*
     &                EXP( -KK( JQ,LYU )*( UTAUPR( LU ) -
     &                TAUCPR( LYU ) ) )
   40       CONTINUE

            DO 50 JQ = NN + 1, NSTR
               ZINT = ZINT + GC( IQ, JQ, LYU )*LL( JQ, LYU )*
     &                EXP( -KK( JQ,LYU )*( UTAUPR( LU ) -
     &                TAUCPR( LYU-1 ) ) )
   50       CONTINUE

            U0C( IQ, LU ) = ZINT

            IF( FBEAM.GT.0.0 ) U0C( IQ, LU ) = ZINT + ZZ( IQ, LYU )*FACT

            U0C( IQ, LU ) = U0C( IQ, LU ) + ZPLK0( IQ,LYU ) +
     &                      ZPLK1( IQ,LYU )*UTAUPR( LU )
            UAVG( LU ) = UAVG( LU ) + CWT( IQ - NN )*U0C( IQ, LU )
            FLUP( LU ) = FLUP( LU ) + CWT( IQ - NN )*CMU( IQ - NN )*
     &                   U0C( IQ, LU )
   60    CONTINUE


         FLUP( LU )  = 2.*PI*FLUP( LU )
         FLDN( LU )  = 2.*PI*FLDN( LU )
         FDNTOT      = FLDN( LU ) + FLDIR( LU )
         FNET        = FDNTOT - FLUP( LU )
         RFLDN( LU ) = FDNTOT - RFLDIR( LU )
         UAVG( LU )  = ( 2.*PI*UAVG( LU ) + DIRINT ) / ( 4.*PI )
         PLSORC      = XR0( LYU ) + XR1( LYU )*UTAUPR( LU )
         DFDT( LU )  = ( 1. - SSALB( LYU ) ) * 4.*PI *
     &                 ( UAVG( LU ) - PLSORC )

   70    CONTINUE
         IF( PRNT( 2 ) ) WRITE ( *, '(F10.4,I7,1P,7E12.3,E14.3)' )
     &       UTAU( LU ), LYU, RFLDIR( LU ), RFLDN( LU ), FDNTOT,
     &       FLUP( LU ), FNET, UAVG( LU ), PLSORC, DFDT( LU )

   80 CONTINUE


      IF( PRNTU0 ) THEN

         WRITE ( *, '(//,2A)' ) ' ******** AZIMUTHALLY AVERAGED ',
     &     'INTENSITIES ( at polar quadrature angles ) *******'

         DO 100 LU = 1, NTAU

            WRITE ( *, '(/,A,F10.4,//,2A)' )
     &        ' Optical depth =', UTAU( LU ),
     &        '     Angle (deg)   cos(Angle)     Intensity',
     &        '     Angle (deg)   cos(Angle)     Intensity'

            DO 90 IQ = 1, NN
               ANG1 = ( 180./PI )*ACOS( CMU( 2 *NN-IQ+1 ) )
               ANG2 = ( 180./PI )*ACOS( CMU( IQ ) )
               WRITE ( *, '(2(0P,F16.4,F13.5,1P,E14.3))' )
     &           ANG1, CMU(2*NN-IQ+1), U0C(IQ,LU),
     &           ANG2, CMU(IQ),        U0C(IQ+NN,LU)
   90       CONTINUE

  100    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE INTCOR( DITHER, FBEAM, FLYR, LAYRU, LYRCUT, MAXMOM,
     &                   MAXULV, MAXUMU, NMOM, NCUT, NPHI, NSTR, NTAU,
     &                   NUMU, OPRIM, PHASA, PHAST, PHASM, PHIRAD, PI,
     &                   RPD, PMOM, SSALB, DTAUC, TAUC, TAUCPR, UMU,
     &                   UMU0, UTAU, UTAUPR, UU, DELTAMPLUS )

c       Corrects intensity field by using Nakajima-Tanaka algorithm
c       (1988). For more details, see Section 3.6 of STWL NASA report.
c
c                I N P U T   V A R I A B L E S
c
c       DITHER  10 times machine precision
c
c       DTAUC   computational-layer optical depths
c
c       FBEAM   incident beam radiation at top
c
c       FLYR    separated fraction in delta-M method
c
c       LAYRU   index of UTAU in multi-layered system
c
c       LYRCUT  logical flag for truncation of computational layer
c
c       NMOM    number of phase function Legendre coefficients supplied
c
c       NCUT    total number of computational layers considered
c
c       NPHI    number of user azimuthal angles
c
c       NSTR    number of polar quadrature angles
c
c       NTAU    number of user-defined optical depths
c
c       NUMU    number of user polar angles
c
c       OPRIM   delta-M-scaled single-scatter albedo
c
c       PHIRAD  azimuthal angles in radians
c
c       PMOM    phase function Legendre coefficients (K, LC)
c                   K = 0 to NMOM, LC = 1 to NLYR with PMOM(0,LC)=1
c
c       RPD     PI/180
c
c       SSALB   single scattering albedo at computational layers
c
c       TAUC    optical thickness at computational levels
c
c       TAUCPR  delta-M-scaled optical thickness
c
c       UMU     cosine of emergent angle
c
c       UMU0    cosine of incident zenith angle
c
c       UTAU    user defined optical depths
c
c       UTAUPR  delta-M-scaled version of UTAU
c
c                O U T P U T   V A R I A B L E S
c
c       UU      corrected intensity field; UU(IU,LU,J)
c                         IU=1,NUMU; LU=1,NTAU; J=1,NPHI
c
c                I N T E R N A L   V A R I A B L E S
c
c       CTHETA  cosine of scattering angle
c       DTHETA  angle (degrees) to define aureole region as
c                    direction of beam source +/- DTHETA
c       PHASA   actual (exact) phase function
c       PHASM   delta-M-scaled phase function
c       PHAST   phase function used in TMS correction; actual phase
c                    function divided by (1-FLYR*SSALB)
c       PL      ordinary Legendre polynomial of degree l, P-sub-l
c       PLM1    ordinary Legendre polynomial of degree l-1, P-sub-(l-1)
c       PLM2    ordinary Legendre polynomial of degree l-2, P-sub-(l-2)
c       THETA0  incident zenith angle (degrees)
c       THETAP  emergent angle (degrees)
c       USSNDM  single-scattered intensity computed by using exact
c                   phase function and scaled optical depth
c                   (first term in STWL(68a))
c       USSP    single-scattered intensity from delta-M method
c                   (second term in STWL(68a))
c       DUIMS   intensity correction term from IMS method
c                   (delta-I-sub-IMS in STWL(A.19))
c
c   Called by- DISORT
c   Calls- SINSCA, SECSCA
c
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   LYRCUT, DELTAMPLUS
      INTEGER   MAXMOM, MAXULV, MAXUMU, NCUT, NMOM, NPHI, NSTR, NTAU,
     &          NUMU
      REAL      DITHER, FBEAM, PI, RPD, UMU0
c     ..
c     .. Array Arguments ..

      INTEGER   LAYRU( * )
      REAL      DTAUC( * ), FLYR( * ), OPRIM( * ), PHASA( * ),
     &          PHAST( * ), PHASM( * ), PHIRAD( * ),
     &          PMOM( 0:MAXMOM, * ), SSALB( * ), TAUC( 0:* ),
     &          TAUCPR( 0:* ), UMU( * ), UTAU( * ), UTAUPR( * ),
     &          UU( MAXUMU, MAXULV, * )

c     ..
c     .. Local Scalars ..

      INTEGER   IU, JP, K, LC, LTAU, LU
      REAL      CTHETA, DTHETA, DUIMS, PL, PLM1, PLM2, THETA0, THETAP,
     &          USSNDM, USSP
      REAL      f, sigma_sq, c 
c     ..
c     .. External Functions ..

      REAL      SECSCA, SINSCA
      EXTERNAL  SECSCA, SINSCA

c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, ACOS, COS, SQRT
c     ..

      THETA0 = 0.0
      THETAP = 0.0

      DTHETA = 10.





c                                ** Start loop over zenith angles

      DO 110 IU = 1, NUMU

         IF( UMU( IU ).LT.0. ) THEN

c                                ** Calculate zenith angles of icident
c                                ** and emerging directions


            THETA0 = ACOS( -UMU0 ) / RPD
            THETAP = ACOS( UMU( IU ) ) / RPD

         END IF

c                                ** Start loop over azimuth angles

         DO 100 JP = 1, NPHI

c                                ** Calculate cosine of scattering
c                                ** angle, Eq. STWL(4)

            CTHETA = -UMU0*UMU( IU ) + SQRT( ( 1.-UMU0**2 )*
     &               ( 1.-UMU( IU )**2 ) )*COS( PHIRAD( JP ) )

c                                ** Initialize phase function
            DO 10 LC = 1, NCUT

               PHASA( LC ) = 1.
               PHASM( LC ) = 1.

   10       CONTINUE
c                                ** Initialize Legendre poly. recurrence
            PLM1 = 1.
            PLM2 = 0.

            DO 40 K = 1, NMOM
c                                ** Calculate Legendre polynomial of
c                                ** P-sub-l by upward recurrence

               PL   = ( ( 2 *K-1 )*CTHETA*PLM1 - ( K-1 )*PLM2 ) / K
               PLM2 = PLM1
               PLM1 = PL
c                                ** Calculate actual phase function
               DO 20 LC = 1, NCUT

                  PHASA( LC ) = PHASA( LC ) +
     &                          ( 2*K + 1 )*PL*PMOM( K, LC )

   20          CONTINUE

c                                ** Calculate delta-M transformed
c                                ** phase function
               DO 30 LC = 1, NCUT
c              if( deltamPlus .and. k .le. nstr-1 .and.
c     &               pmom(nstr,lc).ne.pmom(nstr+1,lc) ) then                               
                 if( deltamPlus ) then 
                   if( k .le. nstr-1 .and.
     &               pmom(nstr,lc).ne.pmom(nstr+1,lc) ) then
c                  ** Calculate new delta-m plus
                   f = pmom(nstr, lc)
                   sigma_sq = ( (nstr+1)**2 - (nstr)**2 ) / 
     &             ( log(pmom(nstr,lc)**2) - log(pmom(nstr+1,lc)**2) )
                   c = exp(nstr**2/(2*sigma_sq))
                   f = c*f
                   phasm(lc) = phasm(lc) + (2*k+1) * PL *(pmom(k,lc) 
     &                         - f*exp(-k**2/(2*sigma_sq)))/(1.0-f)
                   endif
                 else IF( K.LE.NSTR - 1 ) THEN
                   PHASM( LC ) = PHASM( LC ) + ( 2*K + 1 ) * PL *
     &                             ( PMOM( K,LC ) - FLYR( LC ) ) /
     &                             ( 1. - FLYR( LC ) )
                 end if
   30          CONTINUE

                 

   40       CONTINUE


c                                ** Apply TMS method, Eq. STWL(68)
            DO 70 LC = 1, NCUT

               PHAST( LC ) = PHASA(LC) / ( 1. - FLYR(LC) * SSALB(LC) )

   70       CONTINUE

            DO 80 LU = 1, NTAU

               IF( .NOT.LYRCUT .OR. LAYRU( LU ).LT.NCUT ) THEN

                   USSNDM  = SINSCA( DITHER, LAYRU( LU ), NCUT, PHAST,
     &                               SSALB, TAUCPR, UMU( IU ), UMU0,
     &                               UTAUPR( LU ), FBEAM, PI )

                   USSP    = SINSCA( DITHER, LAYRU( LU ), NCUT, PHASM,
     &                               OPRIM, TAUCPR, UMU( IU ), UMU0,
     &                               UTAUPR( LU ), FBEAM, PI )

                   UU( IU, LU, JP ) = UU( IU, LU, JP ) + USSNDM - USSP

               END IF

   80       CONTINUE

            IF( UMU(IU).LT.0. .AND. ABS( THETA0-THETAP ).LE.DTHETA) THEN

c                                ** Emerging direction is in the aureole
c                                ** (theta0 +/- dtheta). Apply IMS
c                                ** method for correction of secondary
c                                ** scattering below top level.

               LTAU = 1

               IF( UTAU( 1 ).LE.DITHER ) LTAU = 2

               DO 90 LU = LTAU, NTAU

                  IF( .NOT.LYRCUT .OR. LAYRU( LU ).LT.NCUT ) THEN
                    if (.not. deltamPlus) then
                      DUIMS = SECSCA( CTHETA, FLYR, LAYRU( LU ), MAXMOM,
     &                                NMOM, NSTR, PMOM, SSALB, DTAUC,
     &                                TAUC, UMU( IU ), UMU0, UTAU( LU ),
     &                                FBEAM, PI )

                      UU( IU, LU, JP ) = UU( IU, LU, JP ) - DUIMS
                    endif

                  END IF

   90          CONTINUE

            END IF
c                                ** End loop over azimuth angles
  100    CONTINUE

c                                ** End loop over zenith angles
  110 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------

      SUBROUTINE INTCOR_BEAM_REFLEC( NUMU, UMU, NPHI, PHI, PHI0, UMU0,
     &                               MAXUMU, MAXPHI, MXUMU, MAXULV, MI,
     &                               NAZZ, NSTR, NTAU, NCUT, FBEAM,
     &                               TAUCPR, UTAUPR, RHOU, RHO_ACCURATE,
     &                               LAYRU, LYRCUT, PI, UU  )
c
c          
c  ** Version 3 subroutine **    
c          
c     Corrections of intensity field (reflected from lower boundary)
c     by using an improved Nakajima-Tanaka algorithm 
c     called after the original intensity correction subroutine INTCOR
c          
c     For more details, see DISORT3 paper section 3.5.2 and Eq.(43)        



c                I N P U T   V A R I A B L E S
c          
c       NUMU    number of user polar angles
c          
c       UMU     cosine of emergent angle
c
c       NPHI    number of user azimuthal angles
c          
c       PHI     azimuthal angles in degree
c          
c       PHI0    azimuthal incident angles in degree
c          
c       UMU0    cosine of incident zenith angle
c       
c       NSTR    number of polar quadrature angles
c
c       NTAU    number of user-defined optical depths
c
c       NCUT    total number of computational layers considered
c
c       FBEAM   incident beam radiation at top
c          
c       TAUCPR  delta-M-scaled optical thickness
c
c       UTAUPR  delta-M-scaled version of UTAU
c
c       RHOU    BRDF fourier components matrix
c
c       RHO_ACCURATE    analytic brdf results
c
c       LAYRU   index of UTAU in multi-layered system
c
c       LYRCUT  logical flag for truncation of computational layer
c
c       PI      3.141592653.... constant
c
c
c
c                O U T P U T   V A R I A B L E S
c
c       UU      corrected intensity field;  UU(IU,LU,J)
c                         IU=1,NUMU; LU=1,NTAU; J=1,NPHI
c
c
c          
c                I N T E R N A L   V A R I A B L E S
c      
c       RHO_APPROX    fourier expanded brdf results
c
c       DPHO          difference between analytic and fourier expanded
c                     brdf results 
c
c       USS           intensity correction term from improved N/T method
c                     (DISORT3 paper Eq. (43), third term on the righ hand side)
c          
c     Called by- DISORT after the original N/T correction
c
c +-------------------------------------------------------------------+
          
      REAL      RHO_APPROX(NUMU,NPHI), RHO_ACCURATE(MAXUMU, MAXPHI)
      LOGICAL   LYRCUT
      INTEGER   NUMU, NPHI, MAXUMU, MAXPHI, MXUMU, MI, NAZZ, NSTR
      REAL      UMU(MAXUMU), PHI(MAXPHI), PHI0, FBEAM
      INTEGER   IU, J, MAXULV, LAYRU( * )
      REAL      TAUCPR( 0:* ), UTAUPR( * )
      REAL      RHOU(MXUMU,0:MI, 0:NAZZ)
      REAL      DRHO, USS, PI, UU( MAXUMU, MAXULV, * )
      

       DO IU = 1, NUMU

         DO J = 1, NPHI

           IF(UMU(IU) .GT. 0.0 .AND. RHO_ACCURATE(IU,J) .GT. 0.0 )THEN

             RHO_APPROX(IU,J) = BDR_APPROX( PHI(J)-PHI0, NSTR, PI,
     &                                      RHOU(IU,0,0:NSTR-1) )


             DO LU = 1, NTAU

               IF( .NOT.LYRCUT .OR. LAYRU( LU ) .LE. NCUT ) THEN

                 DRHO = RHO_ACCURATE(IU,J) - RHO_APPROX(IU,J) 

                 USS = UMU0 * FBEAM * DRHO
     &                  * EXP( -TAUCPR(NCUT) / UMU0 )
     &                  * EXP( (UTAUPR(LU) - TAUCPR(NCUT))/UMU(IU) )

                 UU( IU, LU, J ) = UU( IU, LU, J) + USS
               END IF

             ENDDO
            END IF
        END DO
       END DO


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      REAL FUNCTION  SECSCA( CTHETA, FLYR, LAYRU, MAXMOM, NMOM, NSTR,
     &                       PMOM, SSALB, DTAUC, TAUC, UMU, UMU0, UTAU,
     &                       FBEAM, PI )

c          Calculates secondary scattered intensity of EQ. STWL (A7)
c
c                I N P U T   V A R I A B L E S
c
c        CTHETA  cosine of scattering angle
c
c        DTAUC   computational-layer optical depths
c
c        FLYR    separated fraction f in Delta-M method
c
c        LAYRU   index of UTAU in multi-layered system
c
c        MAXMOM  maximum number of phase function moment coefficients
c
c        NMOM    number of phase function Legendre coefficients supplied
c
c        NSTR    number of polar quadrature angles
c
c        PMOM    phase function Legendre coefficients (K, LC)
c                K = 0 to NMOM, LC = 1 to NLYR, with PMOM(0,LC)=1
c
c        SSALB   single scattering albedo of computational layers
c
c        TAUC    cumulative optical depth at computational layers
c
c        UMU     cosine of emergent angle
c
c        UMU0    cosine of incident zenith angle
c
c        UTAU    user defined optical depth for output intensity
c
c        FBEAM   incident beam radiation at top
c
c        PI       3.1415...
c
c   LOCAL VARIABLES
c
c        PSPIKE  2*P"-P"**2, where P" is the residual phase function
c        WBAR    mean value of single scattering albedo
c        FBAR    mean value of separated fraction f
c        DTAU    layer optical depth
c        STAU    sum of layer optical depths between top of atmopshere
c                and layer LAYRU
c
c   Called by- INTCOR
c   Calls- XIFUNC
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..
      INTEGER   LAYRU, MAXMOM, NMOM, NSTR
      REAL      CTHETA, FBEAM, PI, UMU, UMU0, UTAU
c     ..
c     .. Array Arguments ..
      REAL      DTAUC( * ), FLYR( * ), PMOM( 0:MAXMOM, * ), SSALB( * ),
     &          TAUC( 0:* )
c     ..
c     .. Local Scalars ..
      INTEGER   K, LYR
      REAL      DTAU, FBAR, GBAR, PL, PLM1, PLM2, PSPIKE, STAU, UMU0P,
     &          WBAR, ZERO
c     ..
c     .. External Functions ..
      REAL      XIFUNC
      EXTERNAL  XIFUNC
c     ..

      ZERO = 1E-4

c                          ** Calculate vertically averaged value of
c                          ** single scattering albedo and separated
c                          ** fraction f, Eq. STWL (A.15)

      DTAU = UTAU - TAUC( LAYRU - 1 )
      WBAR = SSALB( LAYRU ) * DTAU
      FBAR = FLYR( LAYRU ) * WBAR
      STAU = DTAU

      DO 10 LYR = 1, LAYRU - 1

         WBAR = WBAR + SSALB( LYR ) * DTAUC( LYR )
         FBAR = FBAR + SSALB( LYR ) * DTAUC( LYR ) * FLYR( LYR )
         STAU = STAU + DTAUC( LYR )

   10 CONTINUE

      IF( WBAR.LE.ZERO .OR.
     &    FBAR.LE.ZERO .OR. STAU.LE.ZERO .OR.FBEAM.LE.ZERO ) THEN

          SECSCA = 0.0
          RETURN

      END IF

      FBAR  = FBAR / WBAR
      WBAR  = WBAR / STAU


c                          ** Calculate PSPIKE=(2P"-P"**2)
      PSPIKE = 1.
      GBAR   = 1.
      PLM1    = 1.
      PLM2    = 0.
c                                   ** PSPIKE for L<=2N-1
      DO 20 K = 1, NSTR - 1

         PL   = ( ( 2 *K-1 )*CTHETA*PLM1 - ( K-1 )*PLM2 ) / K
         PLM2  = PLM1
         PLM1  = PL

         PSPIKE = PSPIKE + ( 2.*GBAR - GBAR**2 )*( 2*K + 1 )*PL

   20 CONTINUE
c                                   ** PSPIKE for L>2N-1
      DO 40 K = NSTR, NMOM

         PL   = ( ( 2 *K-1 )*CTHETA*PLM1 - ( K-1 )*PLM2 ) / K
         PLM2  = PLM1
         PLM1  = PL

         DTAU = UTAU - TAUC( LAYRU - 1 )

         GBAR = PMOM( K, LAYRU ) * SSALB( LAYRU ) * DTAU

         DO 30 LYR = 1, LAYRU - 1
            GBAR = GBAR + PMOM( K, LYR ) * SSALB( LYR ) * DTAUC( LYR )
   30    CONTINUE

         IF( FBAR*WBAR*STAU .LE. ZERO ) THEN
            GBAR   = 0.0
         ELSE
            GBAR   = GBAR / ( FBAR*WBAR*STAU )
         END IF

         PSPIKE = PSPIKE + ( 2.*GBAR - GBAR**2 )*( 2*K + 1 )*PL

   40 CONTINUE

      UMU0P = UMU0 / ( 1. - FBAR*WBAR )

c                              ** Calculate IMS correction term,
c                              ** Eq. STWL (A.13)

      SECSCA = FBEAM / ( 4.*PI ) * ( FBAR*WBAR )**2 / ( 1.-FBAR*WBAR ) *
     &         PSPIKE * XIFUNC( -UMU, UMU0P, UMU0P, UTAU )


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SETDIS( CMU, CWT, DELTAM, DTAUC, DTAUCP, EXPBEA, FBEAM,
     &                   FLYR, GL, IBCND, LAYRU, LYRCUT, MAXMOM, MAXUMU,
     &                   MXCMU, NCUT, NLYR, NTAU, NN, NSTR, PLANK, NUMU,
     &                   ONLYFL, CORINT, OPRIM, PMOM, SSALB, TAUC,
     &                   TAUCPR, UTAU, UTAUPR, UMU, UMU0, USRTAU,
     &                   USRANG, NAZZ, MI, SQT,
     &                   DO_PSEUDO_SPHERE, EARTH_RADIUS, H_LYR, UMU0L,
     &                   DELTAMPLUS)

c          Perform miscellaneous setting-up operations
c
c    INPUT :  all are DISORT input variables (see DOC file)
c
c
c    O U T P U T     V A R I A B L E S:
c
c       NTAU,UTAU   if USRTAU = FALSE (defined in DISORT.doc)
c       NUMU,UMU    if USRANG = FALSE (defined in DISORT.doc)
c
c       CMU,CWT     computational polar angles and
c                   corresponding quadrature weights
c
c       EXPBEA      transmission of direct beam
c
c       FLYR        separated fraction in delta-M method
c
c       GL          phase function Legendre coefficients multiplied
c                   by (2L+1) and single-scatter albedo
c
c       LAYRU       Computational layer in which UTAU falls
c
c       LYRCUT      flag as to whether radiation will be zeroed
c                   below layer NCUT
c
c       NCUT        computational layer where absorption
c                   optical depth first exceeds  ABSCUT
c
c       NN          NSTR / 2
c
c       OPRIM       delta-M-scaled single-scatter albedo
c
c       TAUCPR      delta-M-scaled optical depth
c
c       UTAUPR      delta-M-scaled version of  UTAU
c
c   Called by- DISORT
c   Calls- QGAUSN, ERRMSG
c ---------------------------------------------------------------------

c     .. Scalar Arguments ..

      LOGICAL   CORINT, DELTAM, LYRCUT, ONLYFL, PLANK, USRANG, USRTAU
      INTEGER   IBCND, MAXMOM, MAXUMU, MXCMU, NCUT, NLYR, NN, NSTR,
     &          NTAU, NUMU
      REAL      FBEAM
      logical   DELTAMPLUS
c     ..
c     .. Array Arguments ..

      INTEGER   LAYRU( * )
      REAL      CMU( MXCMU ), CWT( MXCMU ), DTAUC( * ), DTAUCP( * ),
     &          EXPBEA( 0:* ), FLYR( * ), GL( 0:NSTR, * ), OPRIM( * ),
     &          PMOM( 0:MAXMOM, * ), SSALB( * ), TAUC( 0:* ),
     &          TAUCPR( 0:* ), UMU( MAXUMU ), UTAU( * ), UTAUPR( * ),
     &          SQT(2*NSTR)

      REAL      EARTH_RADIUS, H_LYR(0:NLYR),UMU0P(NLYR,NLYR),UMU0L(NLYR)
      LOGICAL   DO_PSEUDO_SPHERE
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, IU, K, LC, LU, n
      REAL      ABSCUT, ABSTAU, F, YESSCT, TAU_SLANT(0:NLYR )
      real      sigma_sq, c 
      INTEGER   NAZZ, MI, NS
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG, QGAUSN, R1MACH
      REAL      R1MACH
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, EXP
c     ..
      DATA      ABSCUT / 10. /

 

      IF( .NOT.USRTAU ) THEN
c                              ** Set output levels at computational
c                              ** layer boundaries
         NTAU  = NLYR + 1

         DO 10 LC = 0, NTAU - 1
            UTAU( LC + 1 ) = TAUC( LC )
   10    CONTINUE

      END IF
c                        ** Apply delta-M scaling and move description
c                        ** of computational layers to local variables
      EXPBEA( 0 ) = 1.0
      TAUCPR( 0 ) = 0.0
      ABSTAU      = 0.0
      YESSCT      = 0.0
      TAU_SLANT( 0 ) = 0.0

c                        ** Call Chapman function
      IF( DO_PSEUDO_SPHERE ) THEN
          CALL CHAPMAN( NLYR, UMU0, EARTH_RADIUS, H_LYR,
     &                  UMU0P(1:NLYR,1:NLYR) ) 
      ENDIF


      DO 40 LC = 1, NLYR

         YESSCT = YESSCT + SSALB( LC )

         PMOM( 0, LC ) = 1.0

         IF( ABSTAU.LT.ABSCUT ) NCUT  = LC

         ABSTAU = ABSTAU + ( 1. - SSALB( LC ) )*DTAUC( LC )

         IF( .NOT.DELTAM .and. .not. DELTAMPLUS ) THEN
            OPRIM( LC )  = SSALB( LC )
            DTAUCP( LC ) = DTAUC( LC )
            TAUCPR( LC ) = TAUC( LC )

            DO 20 K = 0, NSTR - 1
               GL( K, LC ) = ( 2*K + 1 )*OPRIM( LC )*PMOM( K, LC )
   20       CONTINUE

            F  = 0.0
            sigma_sq = 0.0


c         ELSE IF ( DELTAM .OR. PMOM(NSTR,LC) .EQ. PMOM(NSTR+1,LC) ) THEN
         ELSE IF ( DELTAM .OR. PMOM(NSTR-1,LC) .EQ. PMOM(NSTR,LC) ) THEN
c                                    ** Do delta-M transformation

            F  = PMOM( NSTR, LC )
            OPRIM( LC )  = SSALB( LC )*( 1. - F ) / ( 1. - F*SSALB(LC) )
            DTAUCP( LC ) = ( 1. - F*SSALB( LC ) )*DTAUC( LC )
            TAUCPR( LC ) = TAUCPR( LC - 1 ) + DTAUCP( LC )

            DO 30 K = 0, NSTR - 1
               GL( K, LC ) = ( 2*K + 1 )*OPRIM( LC )*
     &                       ( PMOM( K,LC ) - F ) / ( 1. - F )
   30       CONTINUE

         ELSE IF (PMOM(NSTR,LC).NE.PMOM(NSTR+1,LC).AND.DELTAMPLUS) THEN
c        ** do new delta-M plus transformation
C          if(PMOM(NSTR,LC).gt.PMOM(NSTR+1,LC)) then ! fix to avoid negative sigma_sq
             f = pmom(nstr, lc)
!           print*, pmom(nstr,lc),pmom(nstr+1,lc)
             sigma_sq = ( (nstr+1)**2 - (nstr)**2 ) / 
     &           ( log(pmom(nstr,lc)**2) - log(pmom(nstr+1,lc)**2) )
             c = exp( nstr**2/(2*sigma_sq) )
!           PRINT*, F, C, sigma_sq
             f = c*f
!           PRINT*, F, C, sigma_sq
C          else
C            f = PMOM( NSTR, LC ) ! go back to delta-M when PMOM(NSTR,LC) < PMOM(NSTR+1,LC)
C          endif  
             oprim(lc) = ssalb(lc)*(1.0-f) / (1.0-f*ssalb(lc))
             dtaucp(lc) = (1.0-f*ssalb(lc)) * dtauc(lc)
             taucpr(lc) = taucpr(lc-1) + dtaucp(lc)
             do k = 0, nstr-1
               gl(k, lc) = (2*k+1)*oprim(lc)*
     &          (pmom(k,lc) -  f*exp(-k**2/(2*sigma_sq))) / (1.0-f) 
             enddo
         else
           print*, "error: can't do both deltaM and deltaM-Plus"
           exit
         END IF

         FLYR( LC ) = F
         EXPBEA( LC ) = 0.0

!         IF( FBEAM.GT.0.0 ) EXPBEA( LC ) = EXP( -TAUCPR( LC )/UMU0 )

c        ** Pseudo spherical correction
c        ** correct beam attenuation term 
c        ** correct solar zenith angle
         IF( FBEAM.GT.0.0 ) THEN
            IF( .NOT. DO_PSEUDO_SPHERE ) THEN
!                print*, LC, TAUCPR(LC), -TAUCPR( LC )/UMU0
                IF( TAUCPR( LC )/UMU0 .LT. -LOG(R1MACH(1)) ) THEN 
                    EXPBEA( LC ) = EXP( -TAUCPR( LC )/UMU0 )
                ELSE
                    EXPBEA( LC ) = 0.0;
                ENDIF
!                print*, EXPBEA(LC)
                UMU0L( LC ) = UMU0
            ELSE
                TAU_SLANT(LC) = 0.0
                DO N = 1, LC
                  TAU_SLANT(LC) = TAU_SLANT(LC) + DTAUCP(N)/UMU0P(LC,N)
                ENDDO
                EXPBEA( LC ) = EXP( - TAU_SLANT(LC) )
                IF ( DTAUCP( LC ) .NE. 0.0 ) THEN
                  UMU0L( LC ) = DTAUCP( LC ) /
     &                  ( TAU_SLANT(LC) - TAU_SLANT(LC-1) )
                ELSE IF (DTAUCP( LC ) .EQ. 0.0 .AND. LC .GT. 1) THEN
                  UMU0L( LC ) = UMU0L(LC-1)
                ELSE
                  UMU0L( LC ) = UMU0
                END IF
            ENDIF
            !print*, DO_PSEUDO_SPHERE, umu0l(lc)

           
         ENDIF


   40 CONTINUE
c                      ** If no thermal emission, cut off medium below
c                      ** absorption optical depth = ABSCUT ( note that
c                      ** delta-M transformation leaves absorption
c                      ** optical depth invariant ).  Not worth the
c                      ** trouble for one-layer problems, though.
      LYRCUT = .FALSE.

      IF( ABSTAU.GE.ABSCUT .AND. .NOT.PLANK .AND. IBCND.NE.1 .AND.
     &    NLYR.GT.1 ) LYRCUT = .TRUE.

      IF( .NOT.LYRCUT ) NCUT = NLYR

c                             ** Set arrays defining location of user
c                             ** output levels within delta-M-scaled
c                             ** computational mesh
      DO 70 LU = 1, NTAU

         DO 50 LC = 1, NLYR

            IF( UTAU( LU ).GE.TAUC( LC-1 ) .AND.
     &          UTAU( LU ).LE.TAUC( LC ) ) GO TO  60

   50    CONTINUE
         LC   = NLYR

   60    CONTINUE
         UTAUPR( LU ) = UTAU( LU )
         IF( DELTAM .or. deltamPlus ) THEN
           UTAUPR( LU ) = TAUCPR( LC - 1 ) +
     &                               ( 1. - SSALB( LC )*FLYR( LC ) )*
     &                               ( UTAU( LU ) - TAUC( LC-1 ) )
         endif
         LAYRU( LU ) = LC

   70 CONTINUE
c                      ** Calculate computational polar angle cosines
c                      ** and associated quadrature weights for Gaussian
c                      ** quadrature on the interval (0,1) (upward)
      NN   = NSTR / 2

      CALL QGAUSN( NN, CMU, CWT )
c                                  ** Downward (neg) angles and weights
      DO 80 IQ = 1, NN
         CMU( IQ + NN ) = -CMU( IQ )
         CWT( IQ + NN ) = CWT( IQ )
   80 CONTINUE


      IF( FBEAM.GT.0.0 ) THEN
c                               ** Compare beam angle to comput. angles
         DO 90 IQ = 1, NN

C           IF( ABS( UMU0-CMU( IQ ) )/UMU0.LT.1.E-4 ) CALL ERRMSG(
C    &          'SETDIS--beam angle=computational angle; change NSTR',
C    &          .True. )
       IF( ABS( UMU0-CMU( IQ ) )/UMU0.LT.10.0*R1MACH(4) ) 
     &     UMU0 = UMU0+10.0*R1MACH(4)


   90    CONTINUE

      END IF


      IF( .NOT.USRANG .OR. ( ONLYFL.AND.MAXUMU.GE.NSTR ) ) THEN

c                                   ** Set output polar angles to
c                                   ** computational polar angles
         NUMU = NSTR

         DO 100 IU = 1, NN
            UMU( IU ) = -CMU( NN + 1 - IU )
  100    CONTINUE

         DO 110 IU = NN + 1, NSTR
            UMU( IU ) = CMU( IU - NN )
  110    CONTINUE

      END IF

CC    COMMENTED BLOCK FOR DYNAMIC ALLOCATION 2017-11-27
CC     IF( USRANG .AND. IBCND.EQ.1 ) THEN
CC
CCc                               ** Shift positive user angle cosines to
CCc                               ** upper locations and put negatives
CCc                               ** in lower locations
CC        DO 120 IU = 1, NUMU
CC           UMU( IU + NUMU ) = UMU( IU )
CC 120    CONTINUE
CC
CC        DO 130 IU = 1, NUMU
CC           UMU( IU ) = -UMU( 2*NUMU + 1 - IU )
CC 130    CONTINUE
CC
CC        NUMU = 2*NUMU
CC
CC     END IF

c                               ** Turn off intensity correction when
c                               ** only fluxes are calculated, there
c                               ** is no beam source, no scattering,
c                               ** or delta-M transformation is not
c                               ** applied
c
      IF( ONLYFL .OR. FBEAM.EQ.0.0 .OR. YESSCT.EQ.0.0 .OR.
     &   .NOT.(DELTAM .OR. DELTAMPLUS) .OR. DO_PSEUDO_SPHERE ) THEN  
        CORINT = .FALSE.
      END IF

c     ** Version 3: Added NAZZ = MXCMU - 1
      NAZZ = MXCMU-1
      MI = MXCMU/2

      DO NS = 1, 2*NSTR  
         SQT( NS ) = SQRT( REAL( NS ) )
      ENDDO

      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SETMTX( BDR, CBAND, CMU, CWT, DELM0, DTAUCP, GC, KK,
     &                   LAMBER, LYRCUT, MXCMU, NCOL, NCUT,
     &                   NLYR, NN, NSTR, TAUCPR, WK )

c        Calculate coefficient matrix for the set of equations
c        obtained from the boundary conditions and the continuity-
c        of-intensity-at-layer-interface equations;  store in the
c        special banded-matrix format required by LAPACK/LINPACK routines
c
c
c    I N P U T      V A R I A B L E S:
c
c       BDR      :  surface bidirectional reflectivity
c
c       CMU,CWT     abscissae, weights for Gauss quadrature
c                   over angle cosine
c
c       DELM0    :  Kronecker delta, delta-sub-m0
c
c       GC       :  Eigenvectors at polar quadrature angles, SC(1)
c
c       KK       :  Eigenvalues of coeff. matrix in Eq. SS(7), STWL(23b)
c
c       LYRCUT   :  Logical flag for truncation of computational layers
c
c       NN       :  Number of streams in a hemisphere (NSTR/2)
c
c       NCUT     :  Total number of computational layers considered
c
c       TAUCPR   :  Cumulative optical depth (delta-M-scaled)
c
c       (remainder are DISORT input variables)
c
c
c   O U T P U T     V A R I A B L E S:
c
c       CBAND    :  Left-hand side matrix of linear system Eq. SC(5),
c                   scaled by Eq. SC(12); in banded form required
c                   by LAPACK/LINPACK solution routines
c
c       NCOL     :  Number of columns in CBAND
c
c
c   I N T E R N A L    V A R I A B L E S:
c
c       IROW     :  Points to row in CBAND
c       JCOL     :  Points to position in layer block
c       LDA      :  Row dimension of CBAND
c       NCD      :  Number of diagonals below or above main diagonal
c       NSHIFT   :  For positioning number of rows in band storage
c       WK       :  Temporary storage for EXP evaluations
c
c
c   BAND STORAGE
c
c      LAPACK/LINPACK requires band matrices to be input in a special
c      form where the elements of each diagonal are moved up or
c      down (in their column) so that each diagonal becomes a row.
c      (The column locations of diagonal elements are unchanged.)
c
c      Example:  if the original matrix is
c
c          11 12 13  0  0  0
c          21 22 23 24  0  0
c           0 32 33 34 35  0
c           0  0 43 44 45 46
c           0  0  0 54 55 56
c           0  0  0  0 65 66
c
c      then its LAPACK input form would be:
c
c           *  *  *  +  +  +  , * = not used
c           *  * 13 24 35 46  , + = used for pivoting
c           * 12 23 34 45 56
c          11 22 33 44 55 66
c          21 32 43 54 65  *
c
c      If A is a band matrix, the following program segment
c      will convert it to the form (ABD) required by LAPACK/LINPACK
c      band-matrix routines:
c
c               N  = (column dimension of A, ABD)
c               ML = (band width below the diagonal)
c               MU = (band width above the diagonal)
c               M = ML + MU + 1
c               DO J = 1, N
c                  I1 = MAX(1, J-MU)
c                  I2 = MIN(N, J+ML)
c                  DO I = I1, I2
c                     K = I - J + M
c                     ABD(K,J) = A(I,J)
c                  END DO
c               END DO
c
c      This uses rows  ML+1  through  2*ML+MU+1  of ABD.
c      The total number of rows needed in ABD is  2*ML+MU+1 .
c      In the example above, N = 6, ML = 1, MU = 2, and the
c      row dimension of ABD must be >= 5.
c
c
c   Called by- DISORT, ALBTRN
c   Calls- ZEROIT
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   LAMBER, LYRCUT
      INTEGER   MXCMU, NCOL, NCUT, NN, NLYR, NSTR
      REAL      DELM0
c     ..
c     .. Array Arguments ..

      REAL      BDR( NN, 0:NN ), CBAND(9*NN-2,NSTR*NLYR), CMU( MXCMU ),
     &          CWT( MXCMU ), DTAUCP( * ), GC( MXCMU, MXCMU, * ),
     &          KK( MXCMU, * ), TAUCPR( 0:* ), WK( MXCMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, IROW, JCOL, JQ, K, LC, LDA, NCD, NNCOL, NSHIFT
      REAL      EXPA, SUM
c     ..
c     .. External Subroutines ..

      EXTERNAL  ZEROIT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..


      CALL ZEROIT( CBAND, (9*NN-2)*NSTR*NLYR )

      NCD    = 3*NN - 1
      LDA    = 3*NCD + 1
      NSHIFT = LDA - 2*NSTR + 1
      NCOL   = 0
c                         ** Use continuity conditions of Eq. STWJ(17)
c                         ** to form coefficient matrix in STWJ(20);
c                         ** employ scaling transformation STWJ(22)
      DO 60 LC = 1, NCUT

         DO 10 IQ = 1, NN
            WK( IQ ) = EXP( KK( IQ,LC )*DTAUCP( LC ) )
   10    CONTINUE

         JCOL  = 0

         DO 30 IQ = 1, NN

            NCOL  = NCOL + 1
            IROW  = NSHIFT - JCOL

            DO 20 JQ = 1, NSTR
               CBAND( IROW + NSTR, NCOL ) =   GC( JQ, IQ, LC )
               CBAND( IROW, NCOL )        = - GC( JQ, IQ, LC )*WK( IQ )
               IROW  = IROW + 1
   20       CONTINUE

            JCOL  = JCOL + 1

   30    CONTINUE


         DO 50 IQ = NN + 1, NSTR

            NCOL  = NCOL + 1
            IROW  = NSHIFT - JCOL

            DO 40 JQ = 1, NSTR
               CBAND( IROW + NSTR, NCOL ) =   GC( JQ, IQ, LC )*
     &                                          WK( NSTR + 1 - IQ )
               CBAND( IROW, NCOL )        = - GC( JQ, IQ, LC )
               IROW  = IROW + 1
   40       CONTINUE

            JCOL  = JCOL + 1

   50    CONTINUE

   60 CONTINUE
c                  ** Use top boundary condition of STWJ(20a) for
c                  ** first layer
      JCOL  = 0

      DO 80 IQ = 1, NN

         EXPA  = EXP( KK( IQ,1 )*TAUCPR( 1 ) )
         IROW  = NSHIFT - JCOL + NN

         DO 70 JQ = NN, 1, -1
            CBAND( IROW, JCOL + 1 ) = GC( JQ, IQ, 1 )*EXPA
            IROW  = IROW + 1
   70    CONTINUE

         JCOL  = JCOL + 1

   80 CONTINUE


      DO 100 IQ = NN + 1, NSTR

         IROW  = NSHIFT - JCOL + NN

         DO 90 JQ = NN, 1, -1
            CBAND( IROW, JCOL + 1 ) = GC( JQ, IQ, 1 )
            IROW  = IROW + 1
   90    CONTINUE

         JCOL  = JCOL + 1

  100 CONTINUE
c                           ** Use bottom boundary condition of
c                           ** STWJ(20c) for last layer

      NNCOL = NCOL - NSTR
      JCOL  = 0

      DO 130 IQ = 1, NN

         NNCOL  = NNCOL + 1
         IROW   = NSHIFT - JCOL + NSTR

         DO 120 JQ = NN + 1, NSTR

            IF( LYRCUT .OR. ( LAMBER .AND. DELM0.EQ.0 ) ) THEN

c                          ** No azimuthal-dependent intensity if Lam-
c                          ** bert surface; no intensity component if
c                          ** truncated bottom layer

               CBAND( IROW, NNCOL ) = GC( JQ, IQ, NCUT )

            ELSE

               SUM  = 0.0

               DO 110 K = 1, NN
                  SUM  = SUM + CWT( K )*CMU( K )*BDR( JQ - NN, K )*
     &                     GC( NN + 1 - K, IQ, NCUT )
  110          CONTINUE

               CBAND( IROW, NNCOL ) = GC( JQ, IQ, NCUT ) -
     &                                ( 1.+ DELM0 )*SUM
            END IF

            IROW  = IROW + 1

  120    CONTINUE

         JCOL  = JCOL + 1

  130 CONTINUE


      DO 160 IQ = NN + 1, NSTR

         NNCOL  = NNCOL + 1
         IROW   = NSHIFT - JCOL + NSTR
         EXPA   = WK( NSTR + 1 - IQ )

         DO 150 JQ = NN + 1, NSTR

            IF( LYRCUT .OR. ( LAMBER .AND. DELM0.EQ.0 ) ) THEN

               CBAND( IROW, NNCOL ) = GC( JQ, IQ, NCUT )*EXPA

            ELSE

               SUM  = 0.0

               DO 140 K = 1, NN
                  SUM  = SUM + CWT( K )*CMU( K )*BDR( JQ - NN, K )*
     &                         GC( NN + 1 - K, IQ, NCUT )
  140          CONTINUE

               CBAND( IROW, NNCOL ) = ( GC( JQ,IQ,NCUT ) -
     &                                ( 1.+ DELM0 )*SUM )*EXPA
            END IF

            IROW  = IROW + 1

  150    CONTINUE

         JCOL  = JCOL + 1

  160 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      REAL FUNCTION  SINSCA( DITHER, LAYRU, NLYR, PHASE, OMEGA, TAU,
     &                       UMU, UMU0, UTAU, FBEAM, PI )

c        Calculates single-scattered intensity from EQS. STWL (65b,d,e)
c
c                I N P U T   V A R I A B L E S
c
c        DITHER   10 times machine precision
c
c        LAYRU    index of UTAU in multi-layered system
c
c        NLYR     number of sublayers
c
c        PHASE    phase functions of sublayers
c
c        OMEGA    single scattering albedos of sublayers
c
c        TAU      optical thicknesses of sublayers
c
c        UMU      cosine of emergent angle
c
c        UMU0     cosine of incident zenith angle
c
c        UTAU     user defined optical depth for output intensity
c
c        FBEAM   incident beam radiation at top
c
c        PI       3.1415...
c
c   Called by- INTCOR
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   LAYRU, NLYR
      REAL      DITHER, FBEAM, PI, UMU, UMU0, UTAU
c     ..
c     .. Array Arguments ..

      REAL      OMEGA( * ), PHASE( * ), TAU( 0:* )
!      real      rho
c     ..
c     .. Local Scalars ..

      INTEGER   LYR
      REAL      EXP0, EXP1

c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, EXP
c     ..


      SINSCA = 0.
      EXP0 = EXP( -UTAU/UMU0 )

      IF( ABS( UMU+UMU0 ).LE.DITHER ) THEN

c                                 ** Calculate downward intensity when
c                                 ** UMU=UMU0, Eq. STWL (65e)

         DO 10 LYR = 1, LAYRU - 1
            SINSCA = SINSCA + OMEGA( LYR ) * PHASE( LYR ) *
     &               ( TAU( LYR ) - TAU( LYR-1 ) )
   10    CONTINUE

         SINSCA = FBEAM / ( 4.*PI * UMU0 ) * EXP0 * ( SINSCA +
     &            OMEGA( LAYRU )*PHASE( LAYRU )*( UTAU-TAU(LAYRU-1) ) )

         RETURN

      END IF


      IF( UMU.GT.0. ) THEN
c                                 ** Upward intensity, Eq. STWL (65b)

 
         DO 20 LYR = LAYRU, NLYR
            EXP1 = EXP( -( ( TAU( LYR )-UTAU )/UMU + TAU( LYR )/UMU0 ) )
            SINSCA = SINSCA + OMEGA( LYR )*PHASE( LYR )*( EXP0 - EXP1 )
            EXP0 = EXP1
   20    CONTINUE

      ELSE
c                                 ** Downward intensity, Eq. STWL (65d)
         DO 30 LYR = LAYRU, 1, -1

            EXP1 = EXP( -( ( TAU(LYR-1)-UTAU )/UMU + TAU(LYR-1)/UMU0 ) )
            SINSCA = SINSCA + OMEGA( LYR )*PHASE( LYR )*( EXP0 - EXP1 )
            EXP0 = EXP1

   30    CONTINUE

      END IF

      SINSCA = FBEAM / ( 4.*PI * ( 1. + UMU/UMU0 ) ) * SINSCA

!c     ** Version 3 old
!      IF( .NOT. PASS1) THEN
!        IF(UMU .GT. 0.) THEN
!          INTENSITY_BOT_UP = UMU0*FBEAM*RHO*EXP(-TAU(NLYR)/UMU0) 
!c          PRINT*,  RHO, SINSCA, SINSCA
!c     &      + INTENSITY_BOT_UP * EXP((TAU(LAYRU)-TAU(NLYR))/UMU)
!
!          SINSCA = SINSCA 
!     &      + INTENSITY_BOT_UP * EXP((UTAU-TAU(NLYR))/UMU)
!        END IF
!      END IF
!c     ** Version 3 old


      RETURN
      END
c- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c-----------------------------------------------------------------------
      REAL FUNCTION BDR_APPROX( DPHI, NSTR, PI, RHOU )
c     ** Version 3 function

      REAL     DPHI
      INTEGER  NSTR
      INTEGER  M
      REAL     RHOU(0:*)
      REAL     RHO_APPROX, RHO_FLOURIER(0:NSTR)

      REAL     BDREF
      EXTERNAL BDREF

      REAL     COSMPHI
      REAL     PI
      INTEGER  NAZ


      NAZ = NSTR-1

      RHO_APPROX = 0.0

      DO M = 0, NAZ
        RHO_FLOURIER(M) = RHOU(M)/PI
        COSMPHI = COS(M*DPHI*PI/180.)
        RHO_APPROX = RHO_APPROX + RHO_FLOURIER(M)*COSMPHI 
      END DO

      BDR_APPROX = RHO_APPROX
      RETURN
      END
c     ** Version 3 function end
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


c ---------------------------------------------------------------------
      SUBROUTINE SOLEIG( AMB, APB, ARRAY, CMU, CWT, GL, MAZIM,
     &                   MXCMU, NN, NSTR, YLMC, CC, EVECC, EVAL, KK, GC,
     &                   AAD, EVECCD, EVALD, WKD )

c         ** Version 3 subroutine 
c         Solves eigenvalue/vector problem necessary to construct
c         homogeneous part of discrete ordinate solution; STWJ(8b),
c         STWL(23f)
c         ** NOTE ** Eigenvalue problem is degenerate when single
c                    scattering albedo = 1;  present way of doing it
c                    seems numerically more stable than alternative
c                    methods that we tried
c
c
c   I N P U T     V A R I A B L E S:
c
c       GL     :  Delta-M scaled Legendre coefficients of phase function
c                 (including factors 2l+1 and single-scatter albedo)
c
c       CMU    :  Computational polar angle cosines
c
c       CWT    :  Weights for quadrature over polar angle cosine
c
c       MAZIM  :  Order of azimuthal component
c
c       NN     :  Half the total number of streams
c
c       YLMC   :  Normalized associated Legendre polynomial
c                 at the quadrature angles CMU
c
c       (remainder are DISORT input variables)
c
c
c   O U T P U T    V A R I A B L E S:
c
c       CC     :  C-sub-ij in Eq. SS(5); needed in SS(15&18)
c
c       EVAL   :  NN eigenvalues of Eq. SS(12), STWL(23f) on return
c                 from ASYMTX but then square roots taken
c
c       EVECC  :  NN eigenvectors  (G+) - (G-)  on return
c                 from ASYMTX ( column j corresponds to EVAL(j) )
c                 but then  (G+) + (G-)  is calculated from SS(10),
c                 G+  and  G-  are separated, and  G+  is stacked on
c                 top of  G-  to form NSTR eigenvectors of SS(7)
c
c       GC     :  Permanent storage for all NSTR eigenvectors, but
c                 in an order corresponding to KK
c
c       KK     :  Permanent storage for all NSTR eigenvalues of SS(7),
c                 but re-ordered with negative values first ( square
c                 roots of EVAL taken and negatives added )
c
c
c   I N T E R N A L   V A R I A B L E S:
c
c       AMB,APB :  Matrices (alpha-beta), (alpha+beta) in reduced
c                    eigenvalue problem
c       ARRAY   :  Complete coefficient matrix of reduced eigenvalue
c                    problem: (alfa+beta)*(alfa-beta)
c       GPPLGM  :  (G+) + (G-) (cf. Eqs. SS(10-11))
c       GPMIGM  :  (G+) - (G-) (cf. Eqs. SS(10-11))
c       WKD     :  Scratch array required by ASYMTX
c
c   Called by- DISORT, ALBTRN
c   Calls- ASYMTX, ERRMSG
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAZIM, MXCMU, NN, NSTR
c     ..
c     .. Array Arguments ..

      REAL      AMB( NN, NN ), APB( NN, NN ), ARRAY( NN, NN ),
     &          CC(NSTR,NSTR ), CMU( MXCMU ), CWT( MXCMU ),
     &          EVAL( NN ), EVECC( NSTR,NSTR ), GC( MXCMU, MXCMU ),
     &          GL( 0:NSTR ), KK( MXCMU ), YLMC( 0:MXCMU, MXCMU )
      DOUBLE PRECISION AAD( NN, NN ), EVALD( NN ), EVECCD(NN,NN),
     &                 WKD( MXCMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IER, IQ, JQ, KQ, L
      REAL      ALPHA, BETA, GPMIGM, GPPLGM, SUM

c      .. Local Array ..
      REAL      TMP(NN, NN)
     
c     ..
c     .. External Subroutines ..

      EXTERNAL  ASYMTX, ERRMSG
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, SQRT
c     ..

c                             ** Calculate quantities in Eqs. SS(5-6),
c                             ** STWL(8b,15,23f)
      DO 40 IQ = 1, NN

         DO 20 JQ = 1, NSTR

            SUM  = 0.0
            DO 10 L = MAZIM, NSTR - 1
               SUM  = SUM + GL( L )*YLMC( L, IQ )*YLMC( L, JQ )
   10       CONTINUE

            CC( IQ, JQ ) = 0.5*SUM*CWT( JQ )

   20    CONTINUE

         DO 30 JQ = 1, NN
c                             ** Fill remainder of array using symmetry
c                             ** relations  C(-mui,muj) = C(mui,-muj)
c                             ** and        C(-mui,-muj) = C(mui,muj)

            CC( IQ + NN, JQ ) = CC( IQ, JQ + NN )
            CC( IQ + NN, JQ + NN ) = CC( IQ, JQ )

c                                       ** Get factors of coeff. matrix
c                                       ** of reduced eigenvalue problem

            ALPHA  = CC( IQ, JQ ) / CMU( IQ )
            BETA   = CC( IQ, JQ + NN ) / CMU( IQ )
            AMB( IQ, JQ ) = ALPHA - BETA
            APB( IQ, JQ ) = ALPHA + BETA

   30    CONTINUE

         AMB( IQ, IQ ) = AMB( IQ, IQ ) - 1.0 / CMU( IQ )
         APB( IQ, IQ ) = APB( IQ, IQ ) - 1.0 / CMU( IQ )

   40 CONTINUE
c                      ** Finish calculation of coefficient matrix of
c                      ** reduced eigenvalue problem:  get matrix
c                      ** product (alfa+beta)*(alfa-beta); SS(12),
c                      ** STWL(23f)
      DO 70 IQ = 1, NN

         DO 60 JQ = 1, NN

            SUM  = 0.
            DO 50 KQ = 1, NN
               SUM  = SUM + APB( IQ, KQ )*AMB( KQ, JQ )
   50       CONTINUE

            ARRAY( IQ, JQ ) = SUM

   60    CONTINUE


   70 CONTINUE
c                      ** Find (real) eigenvalues and eigenvectors

      CALL ASYMTX( ARRAY, EVECC, EVAL, NN, NN, NSTR, IER, WKD, AAD,
     &             EVECCD, EVALD )

      IF( IER.GT.0 ) THEN

         WRITE( *, '(//,A,I4,A)' ) ' ASYMTX--eigenvalue no. ',
     &      IER, '  didnt converge.  Lower-numbered eigenvalues wrong.'

         CALL ERRMSG( 'ASYMTX--convergence problems',.True.)

      END IF


      DO 80 IQ = 1, NN
         EVAL( IQ )    = SQRT( ABS( EVAL( IQ ) ) )
         KK( IQ + NN ) = EVAL( IQ )
c                                      ** Add negative eigenvalue
         KK( NN + 1 - IQ ) = -EVAL( IQ )
   80 CONTINUE

c                          ** Find eigenvectors (G+) + (G-) from SS(10)
c                          ** and store temporarily in array
      DO 110 JQ = 1, NN

         DO 100 IQ = 1, NN

            SUM  = 0.
            DO 90 KQ = 1, NN
               SUM  = SUM + AMB( IQ, KQ )*EVECC( KQ, JQ )
   90       CONTINUE

            TMP( IQ, JQ ) = SUM / EVAL( JQ )

  100    CONTINUE

  110 CONTINUE


      DO 130 JQ = 1, NN

         DO 120 IQ = 1, NN

            GPPLGM = TMP( IQ, JQ )
            GPMIGM = EVECC( IQ, JQ )
c                                ** Recover eigenvectors G+,G- from
c                                ** their sum and difference; stack them
c                                ** to get eigenvectors of full system
c                                ** SS(7) (JQ = eigenvector number)

            EVECC( IQ,      JQ ) = 0.5*( GPPLGM + GPMIGM )
            EVECC( IQ + NN, JQ ) = 0.5*( GPPLGM - GPMIGM )

c                                ** Eigenvectors corresponding to
c                                ** negative eigenvalues (corresp. to
c                                ** reversing sign of 'k' in SS(10) )
            GPPLGM = - GPPLGM
            EVECC(IQ,   JQ+NN) = 0.5 * ( GPPLGM + GPMIGM )
            EVECC(IQ+NN,JQ+NN) = 0.5 * ( GPPLGM - GPMIGM )
            GC( IQ+NN,   JQ+NN )   = EVECC( IQ,    JQ )
            GC( NN+1-IQ, JQ+NN )   = EVECC( IQ+NN, JQ )
            GC( IQ+NN,   NN+1-JQ ) = EVECC( IQ,    JQ+NN )
            GC( NN+1-IQ, NN+1-JQ ) = EVECC( IQ+NN, JQ+NN )

  120    CONTINUE

  130 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SOLVE0( B, BDR, BEM, BPLANK, CBAND, CMU, CWT, EXPBEA,
     &                   FBEAM, FISOT, IPVT, LAMBER, LL, LYRCUT, MAZIM,
     &                   MXCMU, NCOL, NCUT, NN, NSTR, NLYR,
     &                   PI, TPLANK, TAUCPR, UMU0, ZZ, ZPLK0, ZPLK1 )

c        Construct right-hand side vector B for general boundary
c        conditions STWJ(17) and solve system of equations obtained
c        from the boundary conditions and the continuity-of-
c        intensity-at-layer-interface equations.
c        Thermal emission contributes only in azimuthal independence.
c
c **    Version 3 upgrade: replace LINPAK by LAPACK 3.5.0   **
c          
c
c    I N P U T      V A R I A B L E S:
c
c       BDR      :  Surface bidirectional reflectivity
c
c       BEM      :  Surface bidirectional emissivity
c
c       BPLANK   :  Bottom boundary thermal emission
c
c       CBAND    :  Left-hand side matrix of linear system Eq. SC(5),
c                   scaled by Eq. SC(12); in banded form required
c                   by LAPACK solution routines
c
c       CMU,CWT  :  Abscissae, weights for Gauss quadrature
c                   over angle cosine
c
c       EXPBEA   :  Transmission of incident beam, EXP(-TAUCPR/UMU0)
c
c       LYRCUT   :  Logical flag for truncation of computational layers
c
c       MAZIM    :  Order of azimuthal component
c
c       NCOL     :  Number of columns in CBAND
c
c       NN       :  Order of double-Gauss quadrature (NSTR/2)
c
c       NCUT     :  Total number of computational layers considered
c
c       TPLANK   :  Top boundary thermal emission
c
c       TAUCPR   :  Cumulative optical depth (delta-M-scaled)
c
c       ZZ       :  Beam source vectors in Eq. SS(19), STWL(24b)
c
c       ZPLK0    :  Thermal source vectors Z0, by solving Eq. SS(16),
c                   Y0 in STWL(26b)
c
c       ZPLK1    :  Thermal source vectors Z1, by solving Eq. SS(16),
c                   Y1 in STWL(26a)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T     V A R I A B L E S:
c
c       B        :  Right-hand side vector of Eq. SC(5) going into
c                   SGBSL; returns as solution vector of Eq. SC(12),
c                   constants of integration without exponential term
c
c      LL        :  Permanent storage for B, but re-ordered
c
c
c   I N T E R N A L    V A R I A B L E S:
c
c       IPVT     :  Integer vector of pivot indices
c       IT       :  Pointer for position in  B
c       NCD      :  Number of diagonals below or above main diagonal
c       RCOND    :  Indicator of singularity for CBAND
c       Z        :  Scratch array required by SGBCO
c
c   Called by- DISORT
c   Calls- ZEROIT, SGBCO, ERRMSG, SGBSL
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   LAMBER, LYRCUT
      INTEGER   MAZIM, MXCMU, NCOL, NCUT, NN, NLYR, NSTR
      REAL      BPLANK, FBEAM, FISOT, PI, TPLANK, UMU0
c     ..
c     .. Array Arguments ..

      INTEGER   IPVT( * )
      REAL      B( NSTR*NLYR ), BDR( NN, 0:NN ), BEM( NN ),
     &          CBAND(9*NN-2,NSTR*NLYR),CMU( MXCMU ),CWT( MXCMU ),
     &          EXPBEA( 0:* ), LL( MXCMU, * ), TAUCPR( 0:* ),
     &          ZPLK0( MXCMU, * ), ZPLK1( MXCMU, * ),
     &          ZZ( MXCMU, * )

          
!      DOUBLE PRECISION LEFT_MAT(9*NN-2,NSTR*NLYR )
!      DOUBLE PRECISION RIGHT_COL(NSTR*NLYR)
c     ..
c     .. Local Scalars ..

      INTEGER   IPNT, IQ, IT, JQ, LC, NCD, INFO
      REAL      SUM
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG, SGBCO, SGBSL, ZEROIT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..


      CALL ZEROIT( B, NSTR*NLYR )
c                              ** Construct B,  STWJ(20a,c) for
c                              ** parallel beam + bottom reflection +
c                              ** thermal emission at top and/or bottom
      IF( MAZIM.GT.0 .AND. FBEAM.GT.0.0 ) THEN 
c                                         ** Azimuth-dependent case
c                                         ** (never called if FBEAM = 0)
         IF( LYRCUT .OR. LAMBER ) THEN

c               ** No azimuthal-dependent intensity for Lambert surface;
c               ** no intensity component for truncated bottom layer

            DO 10 IQ = 1, NN
c                                                  ** Top boundary
               B( IQ ) = -ZZ( NN + 1 - IQ, 1 )
c                                                  ** Bottom boundary

               B( NCOL - NN + IQ ) = -ZZ( IQ + NN, NCUT )*EXPBEA( NCUT )

   10       CONTINUE


         ELSE

            DO 30 IQ = 1, NN

               B( IQ ) = -ZZ( NN + 1 - IQ, 1 )

               SUM  = 0.
               DO 20 JQ = 1, NN
                  SUM  = SUM + CWT( JQ )*CMU( JQ )*BDR( IQ, JQ )*
     &                         ZZ( NN + 1 - JQ, NCUT )*EXPBEA( NCUT )
   20          CONTINUE

               B( NCOL - NN + IQ ) = SUM
               IF( FBEAM.GT.0.0 ) B( NCOL - NN + IQ ) = SUM +
     &             ( BDR( IQ,0 )*UMU0*FBEAM/PI
     &             - ZZ( IQ+NN,NCUT ) )*EXPBEA( NCUT )

   30       CONTINUE

         END IF
c                             ** Continuity condition for layer
c                             ** interfaces of Eq. STWJ(20b)
         IT  = NN

         DO 50 LC = 1, NCUT - 1

            DO 40 IQ = 1, NSTR
               IT  = IT + 1
               B( IT ) = ( ZZ( IQ,LC+1 ) - ZZ( IQ,LC ) )*EXPBEA( LC )
   40       CONTINUE

   50    CONTINUE


      ELSE
c                                   ** Azimuth-independent case

         IF( FBEAM.EQ.0.0 ) THEN

            DO 60 IQ = 1, NN
c                                      ** Top boundary

               B( IQ ) = -ZPLK0( NN + 1 - IQ, 1 ) + FISOT + TPLANK

   60       CONTINUE


            IF( LYRCUT ) THEN
c                               ** No intensity component for truncated
c                               ** bottom layer
               DO 70 IQ = 1, NN
c                                      ** Bottom boundary

                  B( NCOL - NN + IQ ) = - ZPLK0( IQ + NN, NCUT ) -
     &                                    ZPLK1( IQ + NN, NCUT ) *
     &                                    TAUCPR( NCUT )
   70          CONTINUE


            ELSE

               DO 90 IQ = 1, NN

                  SUM  = 0.
                  DO 80 JQ = 1, NN
                     SUM  = SUM + CWT( JQ )*CMU( JQ )*BDR( IQ, JQ )*
     &                        ( ZPLK0( NN+1-JQ, NCUT ) +
     &                          ZPLK1( NN+1-JQ, NCUT ) *TAUCPR( NCUT ) )
   80             CONTINUE

                  B( NCOL - NN + IQ ) = 2.*SUM + BEM( IQ )*BPLANK -
     &                                  ZPLK0( IQ + NN, NCUT ) -
     &                                  ZPLK1( IQ + NN, NCUT ) *
     &                                  TAUCPR( NCUT )
   90          CONTINUE

            END IF
c                             ** Continuity condition for layer
c                             ** interfaces, STWJ(20b)
            IT  = NN
            DO 110 LC = 1, NCUT - 1

               DO 100 IQ = 1, NSTR
                  IT  = IT + 1
                  B( IT ) =   ZPLK0( IQ, LC + 1 ) - ZPLK0( IQ, LC ) +
     &                      ( ZPLK1( IQ, LC + 1 ) - ZPLK1( IQ, LC ) )*
     &                      TAUCPR( LC )
  100          CONTINUE

  110       CONTINUE


         ELSE

            DO 120 IQ = 1, NN
               B( IQ ) = -ZZ( NN + 1 - IQ, 1 ) -
     &                   ZPLK0( NN + 1 - IQ, 1 ) + FISOT + TPLANK
  120       CONTINUE

            IF( LYRCUT ) THEN

               DO 130 IQ = 1, NN
                  B( NCOL-NN+IQ ) = - ZZ(IQ+NN, NCUT) * EXPBEA(NCUT)
     &                              - ZPLK0(IQ+NN, NCUT)
     &                              - ZPLK1(IQ+NN, NCUT) * TAUCPR(NCUT)
  130          CONTINUE


            ELSE

               DO 150 IQ = 1, NN

                  SUM  = 0.
                  DO 140 JQ = 1, NN
                     SUM = SUM + CWT(JQ) * CMU(JQ) * BDR(IQ,JQ)
     &                          * ( ZZ(NN+1-JQ, NCUT) * EXPBEA(NCUT)
     &                            + ZPLK0(NN+1-JQ, NCUT)
     &                            + ZPLK1(NN+1-JQ, NCUT) * TAUCPR(NCUT))
  140             CONTINUE

                  B(NCOL-NN+IQ) = 2.*SUM + ( BDR(IQ,0) * UMU0*FBEAM/PI
     &                            - ZZ(IQ+NN, NCUT) ) * EXPBEA(NCUT)
     &                            + BEM(IQ) * BPLANK
     &                            - ZPLK0(IQ+NN, NCUT)
     &                            - ZPLK1(IQ+NN, NCUT) * TAUCPR(NCUT)
  150          CONTINUE

            END IF


            IT  = NN

            DO 170 LC = 1, NCUT - 1

               DO 160 IQ = 1, NSTR

                  IT  = IT + 1
                  B(IT) = ( ZZ(IQ,LC+1) - ZZ(IQ,LC) ) * EXPBEA(LC)
     &                    + ZPLK0(IQ,LC+1) - ZPLK0(IQ,LC) +
     &                    ( ZPLK1(IQ,LC+1) - ZPLK1(IQ,LC) ) * TAUCPR(LC)
  160          CONTINUE

  170       CONTINUE

         END IF

      END IF

      NCD    = 3*NN - 1

c     ** version 3: LAPACK with single precision **
c     L-U decomposition:  SGBTRF      
c     Solve linear system: SGBTRS
c      
c
c                     ** Find L-U (lower/upper triangular) decomposition
c                     ** of band matrix CBAND and test if it is nearly
c                     ** singular (note: CBAND is destroyed)
c                     ** (CBAND is in LAPACK packed format)

      CALL SGBTRF( NCOL, NCOL, NCD, NCD, CBAND, 9*NN-2, IPVT, INFO )
 
      IF( INFO .NE. 0 ) 
     &   CALL ERRMSG('SOLVE0--SGBTRF says matrix near singular',.FALSE.)
 
c                   ** Solve linear system with coeff matrix CBAND
c                   ** and R.H. side(s) B after CBAND has been L-U
c                   ** decomposed.  Solution is returned in B.
 
      CALL SGBTRS( 'N', NCOL, NCD, NCD, 1, CBAND, 9*NN-2, IPVT,
     &              B, NSTR*NLYR, INFO   )




c     ** code prior to Version 3: LINPACK with single precision
c
C      RCOND  = 0.0
C      CALL SGBCO( CBAND, 9*NN-2, NCOL, NCD, NCD, IPVT, RCOND, Z )
C      IF( 1.0 + RCOND.EQ.1.0 )
C    &    CALL ERRMSG('SOLVE0--SGBCO says matrix near singular',.FALSE.)
C      CALL SGBSL( CBAND, 9*NN-2, NCOL, NCD, NCD, IPVT, B, 0 )
 




c                   ** Zero CBAND (it may contain 'foreign'
c                   ** elements upon returning from LAPACK/LINPACK);
c                   ** necessary to prevent errors

      CALL ZEROIT( CBAND, (9*NN-2)*NSTR*NLYR )

      DO 190 LC = 1, NCUT

         IPNT  = LC*NSTR - NN

         DO 180 IQ = 1, NN
            LL( NN + 1 - IQ, LC ) = B( IPNT + 1 - IQ )
            LL( IQ + NN,     LC ) = B( IQ + IPNT )
  180    CONTINUE

  190 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SURFAC( ALBEDO, FBEAM, LAMBER, MI, MAZIM,
     &                   MXUMU, NN, NUMU, ONLYFL, UMU, 
     &                   USRANG, BDR, EMU, BEM, RMU, 
     &                   RHOQ, RHOU, EMUST, BEMST, NAZZ )

c     ** Version 3 has new added variables after RMU
c
c       Computes user's surface bidirectional properties, STWL(41)
c
c   I N P U T     V A R I A B L E S:
c
c       CMU    :  Computational polar angle cosines (Gaussian)
c
c       DELM0  :  Kronecker delta, delta-sub-m0
c
c       MAZIM  :  Order of azimuthal component
c
c       NN     :  Order of Double-Gauss quadrature (NSTR/2)
c
c       (Remainder are 'DISORT' input variables)
c
c    O U T P U T     V A R I A B L E S:
c
c       BDR :  Fourier expansion coefficient of surface bidirectional
c                 reflectivity (computational angles)
c
c       RMU :  Surface bidirectional reflectivity (user angles)
c
c       BEM :  Surface directional emissivity (computational angles)
c
c       EMU :  Surface directional emissivity (user angles)

c    I N T E R N A L     V A R I A B L E S:

c       DREF   :  Directional reflectivity
c
c       NMUG   :  Number of angle cosine quadrature points on (-1,1)
c                 for integrating bidirectional reflectivity to get
c                 directional emissivity (it is necessary to use a
c                 quadrature set distinct from the computational angles,
c                 because the computational angles may not be dense
c                 enough -- i.e. 'NSTR' may be too small-- to give an
c                 accurate approximation for the integration).
c
c       GMU    :  The 'NMUG' angle cosine quadrature points on (0,1)
c
c       GWT    :  The 'NMUG' angle cosine quadrature weights on (0,1)
c
c   Called by- DISORT
c   Calls- QGAUSN, BDREF, ZEROIT
c+---------------------------------------------------------------------+

c     .. Parameters ..

      INTEGER   NMUG
      PARAMETER ( NMUG = 50 )
c     ..
c     .. Scalar Arguments ..

      LOGICAL   LAMBER, ONLYFL, USRANG
      INTEGER   MAZIM, MI, MXUMU, NN, NUMU
      REAL      ALBEDO, FBEAM 
!      REAL      WVNMHI, WVNMLO, UMU0

c     ..
c     .. Array Arguments ..
      REAL      BDR( NN, 0:NN ), BEM( NN ), EMU( NUMU ),
     &          RMU( NUMU, 0:NN ), UMU( * )
c    ..
      REAL RHOQ( MI, 0:MI, 0:NAZZ ), RHOU( MXUMU, 0:MI, 0:NAZZ ),
     &     EMUST( NUMU ), BEMST( MI )
c    ..

c     ..
c     .. Local Scalars ..

      LOGICAL   PASS1
      INTEGER   IQ, IU, JQ

c     ..
c     .. External Functions ..

      REAL      BDREF
      EXTERNAL  BDREF
c     ..
c     .. External Subroutines ..

      EXTERNAL  QGAUSN, ZEROIT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC COS
c     ..
      SAVE      PASS1
      DATA      PASS1 / .True. /

c      IF( PASS1 ) THEN

c         PASS1  = .FALSE.

c         CALL QGAUSN( NMUG/2, GMU, GWT )

c         DO 10 K = 1, NMUG / 2
c            GMU( K + NMUG/2 ) = -GMU( K )
c            GWT( K + NMUG/2 ) = GWT( K )
c   10    CONTINUE

c      END IF

      CALL ZEROIT( BDR, NN*( NN+1 ) )
      CALL ZEROIT( BEM, NN )

c                             ** Compute Fourier expansion coefficient
c                             ** of surface bidirectional reflectance
c                             ** at computational angles Eq. STWL (41)

      IF( LAMBER .AND. MAZIM.EQ.0 ) THEN

         DO 30 IQ = 1, NN
            BEM( IQ ) = 1.0 - ALBEDO

            DO 20 JQ = 0, NN
               BDR( IQ, JQ ) = ALBEDO
   
   20       CONTINUE

   30    CONTINUE

      ELSE IF( .NOT.LAMBER ) THEN

         DO 70 IQ = 1, NN

            DO 50 JQ = 1, NN

               BDR(IQ,JQ) = RHOQ(IQ,JQ,MAZIM)

   50       CONTINUE

            IF( FBEAM.GT.0.0 ) THEN
       
                BDR(IQ,0) = RHOQ(IQ,0,MAZIM)

            END IF

   70    CONTINUE


         IF( MAZIM.EQ.0 ) THEN

c                             ** Integrate bidirectional reflectivity
c                             ** at reflection polar angle cosines -CMU-
c                             ** and incident angle cosines -GMU- to get
c                             ** directional emissivity at computational
c                             ** angle cosines -CMU-.
            DO 100 IQ = 1, NN
                BEM(IQ) = BEMST(IQ)

  100       CONTINUE

         END IF

      END IF
c                             ** Compute Fourier expansion coefficient
c                             ** of surface bidirectional reflectance
c                             ** at user angles Eq. STWL (41)

      IF( .NOT.ONLYFL .AND. USRANG ) THEN

         CALL ZEROIT( EMU, NUMU )
         CALL ZEROIT( RMU, NUMU*( NN+1 ) )

         DO 170 IU = 1, NUMU

            IF( UMU(IU).GT.0.0 ) THEN

               IF( LAMBER .AND. MAZIM.EQ.0 ) THEN

                  DO 110 IQ = 0, NN
                     RMU(IU,IQ) = ALBEDO
  110             CONTINUE

                  EMU(IU) = 1.0 - ALBEDO

               ELSE IF( .NOT.LAMBER ) THEN

                  DO 130 IQ = 1, NN
                  RMU(IU,IQ) = RHOU(IU,IQ,MAZIM)

  130             CONTINUE

                  IF( FBEAM.GT.0.0 ) THEN

                   RMU(IU,0) = RHOU(IU,0,MAZIM)

                  END IF


                  IF( MAZIM.EQ.0 ) THEN

c                               ** Integrate bidirectional reflectivity
c                               ** at reflection angle cosines -UMU- and
c                               ** incident angle cosines -GMU- to get
c                               ** directional emissivity at
c                               ** user angle cosines -UMU-.

                      EMU(IU) = EMUST(IU)

                  END IF

               END IF

            END IF

  170    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE TERPEV( CWT, EVECC, GL, GU, MAZIM, MXCMU, MXUMU, NN,
     &                   NSTR, NUMU, WK, YLMC, YLMU )

c         Interpolate eigenvectors to user angles; Eq SD(8)
c
c   Called by- DISORT, ALBTRN
c --------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAZIM, MXCMU, MXUMU, NN, NSTR, NUMU
c     ..
c     .. Array Arguments ..

      REAL      CWT( MXCMU ), EVECC( NSTR, NSTR ), GL( 0:NSTR ),
     &          GU( MXUMU, MXCMU ), WK( MXCMU ), YLMC( 0:MXCMU, MXCMU ),
     &          YLMU( 0:MXCMU, MXUMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, IU, JQ, L
      REAL      SUM
c     ..


      DO 50 IQ = 1, NSTR

         DO 20 L = MAZIM, NSTR - 1
c                                   ** Inner sum in SD(8) times all
c                                   ** factors in outer sum but PLM(mu)
            SUM  = 0.0
            DO 10 JQ = 1, NSTR
               SUM  = SUM + CWT( JQ )*YLMC( L, JQ )*EVECC( JQ, IQ )
   10       CONTINUE

            WK( L + 1 ) = 0.5*GL( L )*SUM

   20    CONTINUE
c                                    ** Finish outer sum in SD(8)
c                                    ** and store eigenvectors
         DO 40 IU = 1, NUMU

            SUM  = 0.
            DO 30 L = MAZIM, NSTR - 1
               SUM  = SUM + WK( L + 1 )*YLMU( L, IU )
   30       CONTINUE

            IF( IQ.LE.NN ) GU( IU, IQ + NN )       = SUM
            IF( IQ.GT.NN ) GU( IU, NSTR + 1 - IQ ) = SUM

   40    CONTINUE

   50 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE TERPSO( CWT, DELM0, FBEAM, GL, MAZIM, MXCMU, PLANK,
     &                   NUMU, NSTR, OPRIM, PI, YLM0, YLMC, YLMU, PSI0,
     &                   PSI1, XR0, XR1, Z0, Z1, ZJ, ZBEAM, Z0U, Z1U )

c         Interpolates source functions to user angles, Eq. STWL(30)
c
c
c    I N P U T      V A R I A B L E S:
c
c       CWT    :  Weights for Gauss quadrature over angle cosine
c
c       DELM0  :  Kronecker delta, delta-sub-m0
c
c       GL     :  Delta-M scaled Legendre coefficients of phase function
c                 (including factors 2L+1 and single-scatter albedo)
c
c       MAZIM  :  Order of azimuthal component
c
c       OPRIM  :  Single scattering albedo
c
c       XR0    :  Expansion of thermal source function, Eq. STWL(24d)
c
c       XR1    :  Expansion of thermal source function Eq. STWL(24d)
c
c       YLM0   :  Normalized associated Legendre polynomial
c                 at the beam angle
c
c       YLMC   :  Normalized associated Legendre polynomial
c                 at the quadrature angles
c
c       YLMU   :  Normalized associated Legendre polynomial
c                 at the user angles
c
c       Z0     :  Solution vectors Z-sub-zero of Eq. SS(16), STWL(26a)
c
c       Z1     :  Solution vectors Z-sub-one  of Eq. SS(16), STWL(26b)
c
c       ZJ     :  Solution vector Z-sub-zero after solving Eq. SS(19),
c                 STWL(24b)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T     V A R I A B L E S:
c
c       ZBEAM  :  Incident-beam source function at user angles
c
c       Z0U,Z1U:  Components of a linear-in-optical-depth-dependent
c                 source (approximating the Planck emission source)
c
c
c   I N T E R N A L    V A R I A B L E S:
c
c       PSI0  :  Sum just after square bracket in  Eq. SD(9)
c       PSI1  :  Sum in Eq. STWL(31d)
c
c   Called by- DISORT
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   PLANK
      INTEGER   MAZIM, MXCMU, NSTR, NUMU
      REAL      DELM0, FBEAM, OPRIM, PI, XR0, XR1
c     ..
c     .. Array Arguments ..

      REAL      CWT( MXCMU ), GL( 0:NSTR ), PSI0( MXCMU ),
     &          PSI1( MXCMU ), YLM0(0:MXCMU,1), YLMC( 0:MXCMU, MXCMU ),
     &          YLMU( 0:MXCMU, * ), Z0( MXCMU ), Z0U( * ), Z1( MXCMU ),
     &          Z1U( * ), ZBEAM( * ), ZJ( MXCMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, IU, JQ
      REAL      FACT, PSUM, PSUM0, PSUM1, SUM, SUM0, SUM1
c     ..


      IF( FBEAM.GT.0.0 ) THEN
c                                  ** Beam source terms; Eq. SD(9)

         DO 20 IQ = MAZIM, NSTR - 1

            PSUM   = 0.
            DO 10 JQ = 1, NSTR
               PSUM  = PSUM + CWT( JQ )*YLMC( IQ, JQ )*ZJ( JQ )
   10       CONTINUE

            PSI0( IQ + 1 ) = 0.5*GL( IQ )*PSUM

   20    CONTINUE

         FACT   = ( 2. - DELM0 )*FBEAM / ( 4.0*PI )

         DO 40 IU = 1, NUMU

            SUM    = 0.
            DO 30 IQ = MAZIM, NSTR - 1
               SUM  = SUM + YLMU( IQ, IU )*
     &                    ( PSI0( IQ+1 ) + FACT*GL( IQ )*YLM0(IQ,1) )
   30       CONTINUE

            ZBEAM( IU ) = SUM

   40    CONTINUE

      END IF


      IF( PLANK .AND. MAZIM.EQ.0 ) THEN

c                          ** Thermal source terms, STWJ(27c), STWL(31c)
c
         DO 60 IQ = MAZIM, NSTR - 1

            PSUM0  = 0.0
            PSUM1  = 0.0
            DO 50 JQ = 1, NSTR
               PSUM0  = PSUM0 + CWT( JQ )*YLMC( IQ, JQ )*Z0( JQ )
               PSUM1  = PSUM1 + CWT( JQ )*YLMC( IQ, JQ )*Z1( JQ )
   50       CONTINUE

            PSI0( IQ + 1 ) = 0.5*GL( IQ ) * PSUM0
            PSI1( IQ + 1 ) = 0.5*GL( IQ ) * PSUM1

   60    CONTINUE

         DO 80 IU = 1, NUMU

            SUM0   = 0.0
            SUM1   = 0.0
            DO 70 IQ = MAZIM, NSTR - 1
               SUM0  = SUM0 + YLMU( IQ, IU ) * PSI0( IQ + 1 )
               SUM1  = SUM1 + YLMU( IQ, IU ) * PSI1( IQ + 1 )
   70       CONTINUE

            Z0U( IU ) = SUM0 + ( 1. - OPRIM ) * XR0
            Z1U( IU ) = SUM1 + ( 1. - OPRIM ) * XR1

   80    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -




c ---------------------------------------------------------------------
      SUBROUTINE UPBEAM( ARRAY, APB, AMB,
     &                   NN, MAZIM, MXCMU,
     &                   CMU, DELM0, FBEAM, GL, YLM0, YLMC,
     &                   PI, UMU0, 
     &                   ZJ, ZZ )
 
c         Finds the incident-beam particular solution of SS(18),
c         STWL(24a)
c          
c **   Version 3 upgrade:                                      **
c **      1)  new algorithm: order of reduction                **
c **      2)  double precision LAPACK 3.5.0                    **
c **      3)  other change: shrink ARRAY dimension             **
c          
c          
c
c   I N P U T    V A R I A B L E S:
c
c       NN     :  half the total number of streams
c
c       CMU    :  Abscissae for Gauss quadrature over angle cosine
c
c       DELM0  :  Kronecker delta, delta-sub-m0
c
c       GL     :  Delta-M scaled Legendre coefficients of phase function
c                 (including factors 2L+1 and single-scatter albedo)
c
c       MAZIM  :  Order of azimuthal component
c
c       YLM0   :  Normalized associated Legendre polynomial
c                 at the beam angle
c
c       YLMC   :  Normalized associated Legendre polynomial
c                 at the quadrature angles
c
c       (remainder are DISORT input variables)
c
c
c   I N T E R N A L   V A R I A B L E S:
c
c       AMB,APB :  Matrices (alpha-beta), (alpha+beta) in reduced
c                    eigenvalue problem
c       ARRAY   :  Complete coefficient matrix of reduced particular
c                  solution problem: (alfa+beta)*(alfa-beta)-1/umu0**
c       IPVT   :  Integer vector of pivot indices required by LAPACK
c
c   Called by- DISORT
c   Calls- SGECO, ERRMSG, SGESL
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAZIM, MXCMU, NN
      REAL      DELM0, FBEAM, PI, UMU0
c     ..
c     .. Array Arguments ..

      REAL      AMB( NN, NN ), APB( NN, NN ), 
     &          ARRAY(NN,NN), 
     &          CMU( MXCMU ),
     &          YLM0( 0:MXCMU,1 ),
     &          YLMC( 0:MXCMU, * ), GL( 0:MXCMU ), ZJ(MXCMU),
     &          ZZ(MXCMU) 
      INTEGER   IPVT ( NN  )
c     ..      
c     .. Local Scalars ..

      INTEGER   IQ, KQ, JOB

      REAL*8    LEFT_MAT(NN,NN),  ZZP(NN), ZZM(NN) 
      REAL*8    SUM, SUM1, SUM2, ZJM(NN), ZJP(NN), FACTOR
      INTEGER   INFO

c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG, DGETRS, DGETRF 
c     ..

c     ** Pass argument, avoid contamination to array

c      LEFT_MAT = ARRAY*UMU0**2
      LEFT_MAT = REAL(ARRAY,8)

      DO 50 IQ = 1, NN

c     .. Left Matrix

c         LEFT_MAT(IQ,IQ) = LEFT_MAT(IQ,IQ) - 1.
         LEFT_MAT(IQ,IQ) = LEFT_MAT(IQ,IQ) - 1d0/REAL(UMU0,8)**2

c     .. Right Vector
       
         SUM1 = 0d0
         SUM2 = 0d0
         DO 60 K = MAZIM, 2*NN-1
            SUM1  = SUM1 +
     &        REAL(GL(K),8)*REAL(YLMC(K,IQ),8)   *REAL(YLM0(K,1),8)
            SUM2  = SUM2 + 
     &        REAL(GL(K),8)*REAL(YLMC(K,IQ+NN),8)*REAL(YLM0(K,1),8)
   60    CONTINUE

         FACTOR = ( 2d0-REAL(DELM0,8) )*REAL(FBEAM,8)/( 4d0*REAL(PI,8) )

         ZJP(IQ) = FACTOR*(SUM1+SUM2)/REAL(CMU(IQ),8)
         ZJM(IQ) = FACTOR*(SUM1-SUM2)/REAL(CMU(IQ),8)

   50 CONTINUE
         

      DO 70 IQ = 1, NN
         SUM = 0d0
         DO 80 KQ = 1, NN
            SUM = SUM + REAL(APB(IQ,KQ),8)*ZJM(KQ)
   80    CONTINUE
c         ZZM(IQ) = -SUM*UMU0**2- ZJP(IQ)*UMU0
         ZZM(IQ) = -SUM - ZJP(IQ)/REAL(UMU0,8)

   70 CONTINUE

c                  ** Find L-U (lower/upper triangular) decomposition
c                  ** of ARRAY and see if it is nearly singular
c                  ** (NOTE: LEFT_MAT is altered)

      CALL DGETRF( NN, NN, LEFT_MAT, NN, IPVT, INFO )


      IF(INFO .NE. 0 ) THEN
         PRINT*, 'BEAM MATRIX LU DECOMPOSITION (DGETRF) FAIL'
      END IF

c                ** Solve linear system with coeff matrix ARRAY
c                ** (assumed already L-U decomposed) and R.H. side(s)
c                ** ZJ;  return solution(s) in ZJ
        JOB  = 0

        CALL DGETRS('N', NN, 1,  LEFT_MAT, NN, IPVT, ZZM, NN, INFO )

      IF(INFO .NE. 0 ) THEN
         PRINT*, 'BEAM SOLUTION (DGETRS) FAIL'
      END IF

        DO 90 IQ = 1, NN
          SUM = 0.
          DO 100 KQ = 1,NN
            SUM = SUM + AMB(IQ,KQ)*ZZM(KQ)
  100     CONTINUE 
          ZZP(IQ) = ( SUM + ZJM(IQ) )*UMU0   
   90   CONTINUE

        DO 110 IQ = 1,NN
           ZJ( IQ )     = 0.5*REAL(ZZP(IQ)+ZZM(IQ),4)
           ZJ( NN+IQ )  = 0.5*REAL(ZZP(IQ)-ZZM(IQ),4)
           ZZ( IQ+NN )  = ZJ( IQ )
           ZZ( NN+1-IQ) = ZJ( IQ + NN )
  110   CONTINUE

      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE UPISOT( ARRAY, CC, CMU, IPVT, MXCMU, NN, NSTR, OPRIM,
     &                   WK, XR0, XR1, Z0, Z1, ZPLK0, ZPLK1 )

c       Finds the particular solution of thermal radiation of STWL(25)
c
c
c
c    I N P U T     V A R I A B L E S:
c
c       CC     :  C-sub-ij in Eq. SS(5), STWL(8b)
c
c       CMU    :  Abscissae for Gauss quadrature over angle cosine
c
c       OPRIM  :  Delta-M scaled single scattering albedo
c
c       XR0    :  Expansion coefficient b-sub-zero of thermal source
c                   function, Eq. STWL(24c)
c
c       XR1    :  Expansion coefficient b-sub-one of thermal source
c                   function Eq. STWL(24c)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T    V A R I A B L E S:
c
c       Z0     :  Solution vectors Z-sub-zero of Eq. SS(16), STWL(26a)
c
c       Z1     :  Solution vectors Z-sub-one  of Eq. SS(16), STWL(26b)
c
c       ZPLK0, :  Permanent storage for Z0,Z1, but re-ordered
c        ZPLK1
c
c
c   I N T E R N A L    V A R I A B L E S:
c
c       ARRAY  :  Coefficient matrix in left-hand side of EQ. SS(16)
c       IPVT   :  Integer vector of pivot indices required by LINPACK
c       WK     :  Scratch array required by LINPACK
c
c   Called by- DISORT
c   Calls- SGECO, ERRMSG, SGESL
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MXCMU, NN, NSTR
      REAL      OPRIM, XR0, XR1
c     ..
c     .. Array Arguments ..

      INTEGER   IPVT( * )
      REAL      ARRAY(NSTR,NSTR ), CC(NSTR,NSTR ), CMU( MXCMU ),
     &          WK( MXCMU ), Z0( MXCMU ), Z1( MXCMU ), ZPLK0( MXCMU ),
     &          ZPLK1( MXCMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, JQ
      REAL      RCOND
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG, SGECO, SGESL
c     ..


      DO 20 IQ = 1, NSTR

         DO 10 JQ = 1, NSTR
            ARRAY( IQ, JQ ) = -CC( IQ, JQ )
   10    CONTINUE

         ARRAY( IQ, IQ ) = 1.0 + ARRAY( IQ, IQ )

         Z1( IQ ) = ( 1. - OPRIM ) * XR1

   20 CONTINUE
c                       ** Solve linear equations: same as in UPBEAM,
c                       ** except ZJ replaced by Z1 and Z0
      RCOND  = 0.0

      CALL SGECO( ARRAY, NSTR, NSTR, IPVT, RCOND, WK )

      IF( 1.0 + RCOND.EQ.1.0 )
     &    CALL ERRMSG('UPISOT--SGECO says matrix near singular',.False.)

      CALL SGESL( ARRAY, NSTR, NSTR, IPVT, Z1, 0 )

      DO 30 IQ = 1, NSTR
         Z0( IQ ) = ( 1. - OPRIM ) * XR0 + CMU( IQ ) * Z1( IQ )
   30 CONTINUE

      CALL SGESL( ARRAY, NSTR, NSTR, IPVT, Z0, 0 )

      DO 40 IQ = 1, NN
         ZPLK0( IQ + NN ) = Z0( IQ )
         ZPLK1( IQ + NN ) = Z1( IQ )
         ZPLK0( NN + 1 - IQ ) = Z0( IQ + NN )
         ZPLK1( NN + 1 - IQ ) = Z1( IQ + NN )
   40 CONTINUE

      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE USRINT( BPLANK, CMU, CWT, DELM0, DTAUCP, EMU, EXPBEA,
     &                   FBEAM, FISOT, GC, GU, KK, LAMBER, LAYRU, LL,
     &                   LYRCUT, MAZIM, MXCMU, MXULV, MXUMU, NCUT, NLYR,
     &                   NN, NSTR, PLANK, NUMU, NTAU, PI, RMU, TAUCPR,
     &                   TPLANK, UMU, UMU0, UTAUPR, WK, ZBEAM, Z0U, Z1U,
     &                   ZZ, ZPLK0, ZPLK1, UUM, UMU0L )

c       Computes intensity components at user output angles
c       for azimuthal expansion terms in Eq. SD(2), STWL(6)
c
c
c   I N P U T    V A R I A B L E S:
c
c       BPLANK :  Integrated Planck function for emission from
c                 bottom boundary
c
c       CMU    :  Abscissae for Gauss quadrature over angle cosine
c
c       CWT    :  Weights for Gauss quadrature over angle cosine
c
c       DELM0  :  Kronecker delta, delta-sub-M0
c
c       EMU    :  Surface directional emissivity (user angles)
c
c       EXPBEA :  Transmission of incident beam, EXP(-TAUCPR/UMU0)
c
c       GC     :  Eigenvectors at polar quadrature angles, SC(1)
c
c       GU     :  Eigenvectors interpolated to user polar angles
c                    (i.e., G in Eq. SC(1) )
c
c       KK     :  Eigenvalues of coeff. matrix in Eq. SS(7), STWL(23b)
c
c       LAYRU  :  Layer number of user level UTAU
c
c       LL     :  Constants of integration in Eq. SC(1), obtained
c                 by solving scaled version of Eq. SC(5);
c                 exponential term of Eq. SC(12) not included
c
c       LYRCUT :  Logical flag for truncation of computational layer
c
c       MAZIM  :  Order of azimuthal component
c
c       NCUT   :  Total number of computational layers considered
c
c       NN     :  Order of double-Gauss quadrature (NSTR/2)
c
c       RMU    :  Surface bidirectional reflectivity (user angles)
c
c       TAUCPR :  Cumulative optical depth (delta-M-Scaled)
c
c       TPLANK :  Integrated Planck function for emission from
c                 top boundary
c
c       UTAUPR :  Optical depths of user output levels in delta-M
c                 coordinates;  equal to UTAU if no delta-M
c
c       Z0U    :  Z-sub-zero in Eq. SS(16) interpolated to user
c                 angles from an equation derived from SS(16),
c                 Y-sub-zero on STWL(26b)
c
c       Z1U    :  Z-sub-one in Eq. SS(16) interpolated to user
c                 angles from an equation derived from SS(16),
c                 Y-sub-one in STWL(26a)
c
c       ZZ     :  Beam source vectors in Eq. SS(19), STWL(24b)
c
c       ZPLK0  :  Thermal source vectors Z0, by solving Eq. SS(16),
c                 Y-sub-zero in STWL(26)
c
c       ZPLK1  :  Thermal source vectors Z1, by solving Eq. SS(16),
c                 Y-sub-one in STWL(26)
c
c       ZBEAM  :  Incident-beam source vectors
c
c       (Remainder are DISORT input variables)
c
c
c    O U T P U T    V A R I A B L E S:
c
c       UUM    :  Azimuthal components of the intensity in EQ. STWJ(5),
c                 STWL(6)
c
c
c    I N T E R N A L    V A R I A B L E S:
c
c       BNDDIR :  Direct intensity down at the bottom boundary
c       BNDDFU :  Diffuse intensity down at the bottom boundary
c       BNDINT :  Intensity attenuated at both boundaries, STWJ(25-6)
c       DTAU   :  Optical depth of a computational layer
c       LYREND :  End layer of integration
c       LYRSTR :  Start layer of integration
c       PALINT :  Intensity component from parallel beam
c       PLKINT :  Intensity component from planck source
c       WK     :  Scratch vector for saving EXP evaluations
c
c       All the exponential factors ( EXP1, EXPN,... etc.)
c       come from the substitution of constants of integration in
c       Eq. SC(12) into Eqs. S1(8-9).  They all have negative
c       arguments so there should never be overflow problems.
c
c   Called by- DISORT
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   LAMBER, LYRCUT, PLANK
      INTEGER   MAZIM, MXCMU, MXULV, MXUMU, NCUT, NLYR, NN, NSTR, NTAU,
     &          NUMU
      REAL      BPLANK, DELM0, FBEAM, FISOT, PI, TPLANK, UMU0, UMU0L(*)
c     ..
c     .. Array Arguments ..

      INTEGER   LAYRU( * )
      REAL      CMU( MXCMU ), CWT( MXCMU ), DTAUCP( * ), EMU( NUMU ),
     &          EXPBEA( 0:* ), GC( MXCMU, MXCMU, * ),
     &          GU( MXUMU, MXCMU, * ), KK( MXCMU, * ), LL( MXCMU, * ),
     &          RMU( NUMU, 0:* ), TAUCPR( 0:* ), UMU( * ),
     &          UTAUPR( MXULV ), UUM( MXUMU, MXULV ), WK( MXCMU ),
     &          Z0U( MXUMU, * ), Z1U( MXUMU, * ), ZBEAM( MXUMU, * ),
     &          ZPLK0( MXCMU, * ), ZPLK1( MXCMU, * ), ZZ( MXCMU, * )
c     ..
c     .. Local Scalars ..

      LOGICAL   NEGUMU
      INTEGER   IQ, IU, JQ, LC, LU, LYREND, LYRSTR, LYU
      REAL      BNDDFU, BNDDIR, BNDINT, DENOM, DFUINT, DTAU, DTAU1,
     &          DTAU2, EXP0, EXP1, EXP2, EXPN, F0N, F1N, FACT, PALINT,
     &          PLKINT, SGN
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, EXP
c     ..

      EXP0 = 0.0
      EXP1 = 0.0
      EXP2 = 0.0

c                          ** Incorporate constants of integration into
c                          ** interpolated eigenvectors
      DO 30 LC = 1, NCUT

         DO 20 IQ = 1, NSTR

            DO 10 IU = 1, NUMU
               GU( IU, IQ, LC ) = GU( IU, IQ, LC ) * LL( IQ, LC )
   10       CONTINUE

   20    CONTINUE

   30 CONTINUE
c                           ** Loop over levels at which intensities
c                           ** are desired ('user output levels')
      DO 160 LU = 1, NTAU

c  comment code      
c        IF( FBEAM.GT.0.0 ) EXP0  = EXP( -UTAUPR( LU ) / UMU0 )

         LYU  = LAYRU( LU )
c  update exp0 with pseudo spherical correction
         EXP0 = 0.0
         IF( FBEAM .GT. 0.0 ) THEN
             DO LC = 1, LYU-1
               EXP0 = EXP0 - DTAUCP(LC) / UMU0L( LC )
             ENDDO
             EXP0 = EXP0 - ( UTAUPR(LU) - TAUCPR(LYU-1) ) / UMU0L(LYU)
             EXP0 = EXP(EXP0)
         ENDIF


c                              ** Loop over polar angles at which
c                              ** intensities are desired
         DO 150 IU = 1, NUMU

            IF( LYRCUT .AND. LYU.GT.NCUT ) GO TO  150

            NEGUMU = UMU( IU ) .LT. 0.0

            IF( NEGUMU ) THEN

               LYRSTR = 1
               LYREND = LYU - 1
               SGN    = -1.0

            ELSE

               LYRSTR = LYU + 1
               LYREND = NCUT
               SGN    = 1.0

            END IF
c                          ** For downward intensity, integrate from top
c                          ** to LYU-1 in Eq. S1(8); for upward,
c                          ** integrate from bottom to LYU+1 in S1(9)
            PALINT = 0.0
            PLKINT = 0.0

            DO 60 LC = LYRSTR, LYREND

               DTAU = DTAUCP( LC )
               EXP1 = EXP( ( UTAUPR(LU) - TAUCPR(LC-1) ) / UMU( IU ) )
               EXP2 = EXP( ( UTAUPR(LU) - TAUCPR(LC)   ) / UMU( IU ) )

               IF( PLANK .AND. MAZIM.EQ.0 ) THEN

c                          ** Eqs. STWL(36b,c, 37b,c)
c
                  F0N = SGN * ( EXP1 - EXP2 )

                  F1N = SGN * ( ( TAUCPR( LC-1 ) + UMU( IU ) ) * EXP1 -
     &                          ( TAUCPR( LC )   + UMU( IU ) ) * EXP2 )

                  PLKINT = PLKINT + Z0U( IU,LC )*F0N + Z1U( IU,LC )*F1N

               END IF


               IF( FBEAM.GT.0.0 ) THEN

                  DENOM  = 1. + UMU( IU ) / UMU0L(LC)

                  IF( ABS( DENOM ).LT.0.0001 ) THEN
c                                                   ** L'Hospital limit
                     EXPN   = ( DTAU / UMU0L(LC) )*EXP0

                  ELSE

                     EXPN   = ( EXP1*EXPBEA( LC-1 ) -
     &                          EXP2*EXPBEA( LC ) ) * SGN / DENOM

                  END IF

                  PALINT = PALINT + ZBEAM( IU, LC )*EXPN

               END IF

c                                                   ** KK is negative
               DO 40 IQ = 1, NN

                  WK( IQ ) = EXP( KK( IQ,LC )*DTAU )
                  DENOM  = 1.0 + UMU( IU )*KK( IQ, LC )

                  IF( ABS( DENOM ).LT.0.0001 ) THEN
c                                                   ** L'Hospital limit
                     EXPN   = DTAU / UMU( IU )*EXP2

                  ELSE

                     EXPN   = SGN*( EXP1*WK( IQ ) - EXP2 ) / DENOM

                  END IF

                  PALINT = PALINT + GU( IU, IQ, LC )*EXPN

   40          CONTINUE

c                                                   ** KK is positive
               DO 50 IQ = NN + 1, NSTR

                  DENOM  = 1.0 + UMU( IU )*KK( IQ, LC )

                  IF( ABS( DENOM ).LT.0.0001 ) THEN
c                                                   ** L'Hospital limit
                     EXPN  = -DTAU / UMU( IU )*EXP1

                  ELSE

                     EXPN  = SGN*( EXP1 - EXP2*WK( NSTR+1-IQ ) ) / DENOM

                  END IF

                  PALINT = PALINT + GU( IU, IQ, LC )*EXPN

   50          CONTINUE


   60       CONTINUE
c                           ** Calculate contribution from user
c                           ** output level to next computational level

            DTAU1  = UTAUPR( LU ) - TAUCPR( LYU - 1 )
            DTAU2  = UTAUPR( LU ) - TAUCPR( LYU )

            IF( ABS( DTAU1 ).LT.1.E-6 .AND. NEGUMU ) GO TO  90
            IF( ABS( DTAU2 ).LT.1.E-6 .AND. (.NOT.NEGUMU ) ) GO TO  90

            IF( NEGUMU )      EXP1  = EXP( DTAU1/UMU( IU ) )
            IF( .NOT.NEGUMU ) EXP2  = EXP( DTAU2/UMU( IU ) )

            IF( FBEAM.GT.0.0 ) THEN

               DENOM  = 1. + UMU( IU ) / UMU0L(LYU)

               IF( ABS( DENOM ).LT.0.0001 ) THEN

                  EXPN   = ( DTAU1 / UMU0L(LYU) )*EXP0

               ELSE IF( NEGUMU ) THEN

                  EXPN  = ( EXP0 - EXPBEA( LYU-1 )*EXP1 ) / DENOM

               ELSE

                  EXPN  = ( EXP0 - EXPBEA( LYU )*EXP2 ) / DENOM

               END IF

               PALINT = PALINT + ZBEAM( IU, LYU )*EXPN

            END IF

c                                                   ** KK is negative
            DTAU  = DTAUCP( LYU )

            DO 70 IQ = 1, NN

               DENOM  = 1. + UMU( IU )*KK( IQ, LYU )

               IF( ABS( DENOM ).LT.0.0001 ) THEN

                  EXPN = -DTAU2 / UMU( IU )*EXP2

               ELSE IF( NEGUMU ) THEN

                  EXPN = ( EXP( -KK( IQ,LYU ) * DTAU2 ) -
     &                     EXP(  KK( IQ,LYU ) * DTAU  ) * EXP1 ) / DENOM

               ELSE

                  EXPN = ( EXP( -KK( IQ,LYU ) * DTAU2 ) - EXP2 ) / DENOM

               END IF

               PALINT = PALINT + GU( IU, IQ, LYU )*EXPN

   70       CONTINUE

c                                                   ** KK is positive
            DO 80 IQ = NN + 1, NSTR

               DENOM  = 1. + UMU( IU )*KK( IQ, LYU )

               IF( ABS( DENOM ).LT.0.0001 ) THEN

                  EXPN   = -DTAU1 / UMU( IU )*EXP1

               ELSE IF( NEGUMU ) THEN

                  EXPN = ( EXP( -KK( IQ,LYU ) * DTAU1 ) - EXP1 ) / DENOM

               ELSE

                  EXPN = ( EXP( -KK( IQ,LYU ) * DTAU1 ) -
     &                     EXP( -KK( IQ,LYU ) * DTAU  ) * EXP2 ) / DENOM

               END IF

               PALINT = PALINT + GU( IU, IQ, LYU )*EXPN

   80       CONTINUE


            IF( PLANK .AND. MAZIM.EQ.0 ) THEN

c                            ** Eqs. STWL (35-37) with tau-sub-n-1
c                            ** replaced by tau for upward, and
c                            ** tau-sub-n replaced by tau for downward
c                            ** directions

               IF( NEGUMU ) THEN

                  EXPN  = EXP1
                  FACT  = TAUCPR( LYU - 1 ) + UMU( IU )

               ELSE

                  EXPN  = EXP2
                  FACT  = TAUCPR( LYU ) + UMU( IU )

               END IF

               F0N  = 1. - EXPN
               F1N  = UTAUPR( LU ) + UMU( IU ) - FACT * EXPN

               PLKINT = PLKINT + Z0U( IU, LYU )*F0N + Z1U( IU, LYU )*F1N

            END IF

c                            ** Calculate intensity components
c                            ** attenuated at both boundaries.
c                            ** NOTE: no azimuthal intensity
c                            ** component for isotropic surface
   90       CONTINUE
            BNDINT = 0.0

            IF( NEGUMU .AND. MAZIM.EQ.0 ) THEN

               BNDINT = (FISOT + TPLANK) * EXP( UTAUPR(LU ) / UMU(IU) )


            ELSE IF( .NOT.NEGUMU ) THEN

               IF( LYRCUT .OR. ( LAMBER.AND.MAZIM.GT.0 ) ) GO TO  140

               DO 100 JQ = NN + 1, NSTR
                  WK( JQ ) = EXP( -KK( JQ,NLYR )*DTAUCP( NLYR ) )
  100          CONTINUE

               BNDDFU = 0.0

               DO 130 IQ = NN, 1, -1

                  DFUINT = 0.0
                  DO 110 JQ = 1, NN
                     DFUINT = DFUINT + GC( IQ, JQ, NLYR )*LL( JQ, NLYR )
  110             CONTINUE

                  DO 120 JQ = NN + 1, NSTR
                     DFUINT = DFUINT + GC( IQ, JQ, NLYR )*
     &                                 LL( JQ, NLYR )*WK( JQ )
  120             CONTINUE

                  IF( FBEAM.GT.0.0 ) DFUINT = DFUINT +
     &                                     ZZ( IQ, NLYR )*EXPBEA( NLYR )

                  DFUINT = DFUINT + DELM0 * ( ZPLK0( IQ, NLYR ) +
     &                              ZPLK1( IQ,NLYR ) *TAUCPR( NLYR ) )
                  BNDDFU = BNDDFU + ( 1.+DELM0 ) * RMU(IU,NN+1-IQ)
     &                            * CMU(NN+1-IQ) * CWT(NN+1-IQ)* DFUINT
  130          CONTINUE

               BNDDIR = 0.0
               IF( FBEAM.GT.0.0 ) BNDDIR = UMU0 * FBEAM 
     &                               / PI*RMU( IU, 0 ) * EXPBEA( NLYR )

               BNDINT = ( BNDDFU + BNDDIR + DELM0 * EMU(IU) * BPLANK )
     &                  * EXP( (UTAUPR(LU)-TAUCPR(NLYR)) / UMU(IU) )

            END IF

  140       CONTINUE

            UUM( IU, LU ) = PALINT + PLKINT + BNDINT

  150    CONTINUE

  160 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      REAL FUNCTION  XIFUNC( UMU1, UMU2, UMU3, TAU )

c          Calculates Xi function of EQ. STWL (72)
c
c                    I N P U T   V A R I A B L E S
c
c        TAU         optical thickness of the layer
c
c        UMU1,2,3    cosine of zenith angle_1, _2, _3
c
c   Called by- SECSCA
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      REAL      TAU, UMU1, UMU2, UMU3
c     ..
c     .. Local Scalars ..

      REAL      EXP1, X1, X2
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..


      X1     = 1. / UMU1 - 1. / UMU2
      X2     = 1. / UMU1 - 1. / UMU3

      EXP1 = EXP( -TAU/UMU1 )

      IF( UMU2.EQ.UMU3 .AND. UMU1.EQ.UMU2 ) THEN

         XIFUNC = TAU*TAU * EXP1 / ( 2.*UMU1*UMU2 )

      ELSE IF( UMU2.EQ.UMU3 .AND. UMU1.NE.UMU2 ) THEN

         XIFUNC = ( ( TAU - 1./X1 ) * EXP( -TAU/UMU2 ) + EXP1 / X1 )
     &            / ( X1*UMU1*UMU2 )

      ELSE IF( UMU2.NE.UMU3 .AND. UMU1.EQ.UMU2 ) THEN

         XIFUNC = ( ( EXP( -TAU/UMU3 ) - EXP1 ) / X2 - TAU * EXP1 )
     &            / ( X2*UMU1*UMU2 )

      ELSE IF( UMU2.NE.UMU3 .AND. UMU1.EQ.UMU3 ) THEN

         XIFUNC = ( ( EXP( -TAU/UMU2 ) - EXP1 ) / X1 - TAU * EXP1 )
     &            / ( X1*UMU1*UMU2 )

      ELSE

         XIFUNC = ( ( EXP( -TAU/UMU3 ) - EXP1 ) / X2 -
     &            (   EXP( -TAU/UMU2 ) - EXP1 ) / X1 ) /
     &            ( X2*UMU1*UMU2 )

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ******************************************************************
c ***************** DISORT service routines ************************
c ******************************************************************

c ---------------------------------------------------------------------
      SUBROUTINE CHEKIN( NLYR, DTAUC, SSALB, NMOM, PMOM, TEMPER, WVNMLO,
     &                   WVNMHI, USRTAU, NTAU, UTAU, NSTR, USRANG,
     &                   NUMU, UMU, NPHI, PHI, IBCND, FBEAM, UMU0,
     &                   PHI0, FISOT, LAMBER, ALBEDO, BTEMP, TTEMP,
     &                   TEMIS, PLANK, ONLYFL, DELTAM, CORINT, ACCUR,
     &                   TAUC, MAXCLY, MAXULV, MAXUMU, MAXPHI, MAXMOM,
     &                   MAXCMU )

c           Checks the input dimensions and variables
c
c   Calls- WRTBAD, WRTDIM, DREF, ERRMSG
c   Called by- DISORT
c +------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   CORINT, DELTAM, LAMBER, ONLYFL, PLANK, USRANG, USRTAU
      INTEGER   IBCND, MAXCLY, MAXMOM, MAXPHI, MAXULV, MAXUMU,
     &          MAXCMU, NLYR, NMOM, NPHI,
     &          NSTR, NTAU, NUMU
      REAL      ACCUR, ALBEDO, BTEMP, FBEAM, FISOT, PHI0, TEMIS, TTEMP,
     &          UMU0, WVNMHI, WVNMLO
c     ..
c     .. Array Arguments ..

      REAL      DTAUC( MAXCLY ), PHI( MAXPHI ),
     &          PMOM( 0:MAXMOM, MAXCLY ), SSALB( MAXCLY ),
     &          TAUC( 0:MAXCLY ), TEMPER( 0:MAXCLY ), UMU( MAXUMU ),
     &          UTAU( MAXULV )
c     ..
c     .. Local Scalars ..

      LOGICAL   INPERR
      INTEGER   IRMU, IU, J, K, LC, LU
      REAL      FLXALB, RMU, YESSCT
c     ..
c     .. External Functions ..

      LOGICAL   WRTBAD, WRTDIM
      REAL      DREF
      EXTERNAL  WRTBAD, WRTDIM, DREF
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, MAX, MOD
c     ..


      INPERR = .FALSE.

      IF( NSTR.LT.2 .OR. MOD( NSTR,2 ).NE.0 ) INPERR = WRTBAD( 'NSTR' )

      IF( NSTR.EQ.2 )
     &    CALL ERRMSG( 'CHEKIN--2 streams not recommended; '//
     &                 'use specialized 2-stream code TWOSTR instead',
     &                 .True.)

      IF( NLYR.LT.1 ) INPERR = WRTBAD( 'NLYR' )

      IF( NLYR.GT.MAXCLY ) INPERR = WRTBAD( 'MAXCLY' )

      YESSCT = 0.0

      DO 10 LC = 1, NLYR

         IF( DTAUC( LC ).LT.0.0 ) INPERR = WRTBAD( 'DTAUC' )

         IF( SSALB( LC ).LT.0.0 .OR. SSALB( LC ).GT.1.0 )
     &       INPERR = WRTBAD( 'SSALB' )

         YESSCT = YESSCT + SSALB( LC )

         IF( PLANK .AND. IBCND.NE.1 ) THEN

            IF( LC.EQ.1 .AND. TEMPER( 0 ).LT.0.0 )
     &          INPERR = WRTBAD( 'TEMPER' )

            IF( TEMPER( LC ).LT.0.0 ) INPERR = WRTBAD( 'TEMPER' )

         END IF

   10 CONTINUE

      IF( NMOM.LT.0 .OR. ( YESSCT.GT.0.0 .AND. NMOM.LT.NSTR ) )
     &    INPERR = WRTBAD( 'NMOM' )

      IF( MAXMOM.LT.NMOM ) INPERR = WRTBAD( 'MAXMOM' )


      DO 30 LC = 1, NLYR

         DO 20 K = 0, NMOM

            IF( PMOM( K,LC ).LT.-1.0 .OR. PMOM( K,LC ).GT.1.0 )
     &          INPERR = WRTBAD( 'PMOM' )

   20    CONTINUE

   30 CONTINUE

      IF( IBCND.EQ.1 ) THEN

         IF( MAXULV.LT.2 ) INPERR = WRTBAD( 'MAXULV' )

      ELSE IF( USRTAU ) THEN

         IF( NTAU.LT.1 ) INPERR = WRTBAD( 'NTAU' )

         IF( MAXULV.LT.NTAU ) INPERR = WRTBAD( 'MAXULV' )

         DO 40 LU = 1, NTAU

            IF( ABS( UTAU( LU )-TAUC( NLYR ) ).LE.1.E-4 )
     &          UTAU( LU ) = TAUC( NLYR )

            IF( UTAU( LU ).LT.0.0 .OR. UTAU( LU ).GT.TAUC( NLYR ) )
     &          INPERR = WRTBAD( 'UTAU' )

   40    CONTINUE

      ELSE

         IF( MAXULV.LT.NLYR + 1 ) INPERR = WRTBAD( 'MAXULV' )

      END IF
       
! TOP LAYER IF ADDED 2017-11-27 TO AVOID CHECKING UMU AND OTHER VARS WHEN ONLYFL = .TRUE.  
      IF( .NOT. ONLYFL ) THEN   
      IF( USRANG ) THEN

         IF( NUMU.LT.0 ) INPERR = WRTBAD( 'NUMU' )

         IF( .NOT.ONLYFL .AND. NUMU.EQ.0 ) INPERR = WRTBAD( 'NUMU' )

         IF( NUMU.GT.MAXUMU ) INPERR = WRTBAD( 'MAXUMU' )

         IF( IBCND.EQ.1 .AND. NUMU.GT.MAXUMU )
     &       INPERR = WRTBAD( 'MAXUMU' )
C    MODIFIED FOR DYNAMIC ALLOCATION 2017-11-27
C        IF( IBCND.EQ.1 .AND. 2*NUMU.GT.MAXUMU )
C    &       INPERR = WRTBAD( 'MAXUMU' )


         DO 50 IU = 1, NUMU

            IF( UMU( IU ).LT.-1.0 .OR. UMU( IU ).GT.1.0 .OR.
     &          UMU( IU ).EQ.0.0 ) INPERR = WRTBAD( 'UMU' )

C    COMMENTED FOR DYNAMIC ALLOCATION 2017-11-27
C           IF( IBCND.EQ.1 .AND. UMU( IU ).LT.0.0 )
C    &          INPERR = WRTBAD( 'UMU' )

            IF( IU.GT.1 ) THEN

               IF( UMU( IU ).LT.UMU( IU-1 ) ) INPERR = WRTBAD( 'UMU' )

            END IF

   50    CONTINUE

      ELSE

         IF( MAXUMU.LT.NSTR ) INPERR = WRTBAD( 'MAXUMU' )

      END IF
      ENDIF 

      IF( .NOT.ONLYFL .AND. IBCND.NE.1 ) THEN

         IF( NPHI.LE.0 ) INPERR = WRTBAD( 'NPHI' )

         IF( NPHI.GT.MAXPHI ) INPERR = WRTBAD( 'MAXPHI' )

         DO 60 J = 1, NPHI

            IF( PHI( J ).LT.0.0 .OR. PHI( J ).GT.360.0 )
     &          INPERR = WRTBAD( 'PHI' )

   60    CONTINUE

      END IF
       


      IF( IBCND.LT.0 .OR. IBCND.GT.1 ) INPERR = WRTBAD( 'IBCND' )

      IF( IBCND.EQ.0 ) THEN

         IF( FBEAM.LT.0.0 ) INPERR = WRTBAD( 'FBEAM' )

         IF( FBEAM.GT.0.0 .AND. ( UMU0.LE.0.0 .OR. UMU0.GT.1.0 ) )
     &       INPERR = WRTBAD( 'UMU0' )

         IF( FBEAM.GT.0.0 .AND. ( PHI0.LT.0.0 .OR. PHI0.GT.360.0 ) )
     &       INPERR = WRTBAD( 'PHI0' )

         IF( FISOT.LT.0.0 ) INPERR = WRTBAD( 'FISOT' )

         IF( LAMBER ) THEN

            IF( ALBEDO.LT.0.0 .OR. ALBEDO.GT.1.0 )
     &          INPERR = WRTBAD( 'ALBEDO' )

         ELSE
c                    ** Make sure flux albedo at dense mesh of incident
c                    ** angles does not assume unphysical values
c                    ** NOTE: We could save some time if we check only 
c                    ** 10 angles as opposed to 100. In which case you
c                    ** can uncomment the two lines below.
c            DO 70 IRMU = 0, 10

            DO 70 IRMU = 0, 100

c               RMU  = IRMU*0.1
               RMU  = IRMU*0.01
c               FLXALB = DREF( WVNMLO, WVNMHI, RMU )
               FLXALB = DREF( RMU )

               IF( FLXALB.LT.0.0 .OR. FLXALB.GT.1.0 )
     &             INPERR = WRTBAD( 'FUNCTION BDREF' )

   70       CONTINUE

         END IF


      ELSE IF( IBCND.EQ.1 ) THEN

         IF( ALBEDO.LT.0.0 .OR. ALBEDO.GT.1.0 )
     &       INPERR = WRTBAD( 'ALBEDO' )

      END IF


      IF( PLANK .AND. IBCND.NE.1 ) THEN

         IF( WVNMLO.LT.0.0 .OR. WVNMHI.LE.WVNMLO )
     &       INPERR = WRTBAD( 'WVNMLO,HI' )

         IF( TEMIS.LT.0.0 .OR. TEMIS.GT.1.0 ) INPERR = WRTBAD( 'TEMIS' )

         IF( BTEMP.LT.0.0 ) INPERR = WRTBAD( 'BTEMP' )

         IF( TTEMP.LT.0.0 ) INPERR = WRTBAD( 'TTEMP' )

      END IF


      IF( ACCUR.LT.0.0 .OR. ACCUR.GT.1.E-2 ) INPERR = WRTBAD( 'ACCUR' )

      IF( MAXCLY.LT.NLYR ) INPERR = WRTDIM( 'MAXCLY', NLYR )

      IF( IBCND.NE.1 ) THEN

         IF( USRTAU .AND. MAXULV.LT.NTAU )
     &       INPERR = WRTDIM( 'MAXULV', NTAU )

         IF( .NOT.USRTAU .AND. MAXULV.LT.NLYR + 1 )
     &       INPERR = WRTDIM( 'MAXULV', NLYR + 1 )

      ELSE

         IF( MAXULV.LT.2 ) INPERR = WRTDIM( 'MAXULV', 2 )

      END IF

      IF( MAXCMU.LT.NSTR ) INPERR = WRTDIM( 'MAXCMU', NSTR )

      IF( USRANG .AND. MAXUMU.LT.NUMU ) INPERR = WRTDIM('MAXUMU',NUMU)


      IF( .NOT.ONLYFL .AND. IBCND.NE.1 .AND. MAXPHI.LT.NPHI )
     &    INPERR = WRTDIM( 'MAXPHI', NPHI )


      IF( INPERR )
     &    CALL ERRMSG( 'DISORT--input and/or dimension errors', .True. )

      IF( PLANK ) THEN

         DO 80 LC = 1, NLYR

            IF( ABS( TEMPER( LC )-TEMPER( LC-1 ) ).GT.10.0 )
     &          CALL ERRMSG('CHEKIN--vertical temperature step may'
     &                      //' be too large for good accuracy',
     &                      .False. )
   80    CONTINUE

      END IF

      IF( .NOT.CORINT .AND. .NOT.ONLYFL .AND. FBEAM.GT.0.0 .AND.
     &    YESSCT.GT.0.0 .AND. DELTAM )
     &     CALL ERRMSG( 'CHEKIN--intensity correction is off; '//
     &                  'intensities may be less accurate', .False. )


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      REAL FUNCTION  DREF( MU )

c      REAL FUNCTION  DREF( WVNMLO, WVNMHI, MU )
c      ** Version 3 removes unused WVNMLO, WVNMHI variables
c
c        Flux albedo for given angle of incidence, given
c        a bidirectional reflectivity.
c
c  INPUT :   MU      Cosine of incidence angle
c
c            WVNMLO  Lower wavenumber (inv-cm) of spectral interval
c
c            WVNMHI  Upper wavenumber (inv-cm) of spectral interval
c
c
c  INTERNAL VARIABLES :
c
c       NMUG   :  Number of angle cosine quadrature points on (-1,1)
c                 for integrating bidirectional reflectivity to get
c                 directional emissivity (it is necessary to use a
c                 quadrature set distinct from the computational angles,
c                 because the computational angles may not be dense
c                 enough -- i.e. 'NSTR' may be too small -- to give an
c                 accurate approximation for the integration).
c
c       GMU    :  The 'NMUG' angle cosine quadrature points on (0,1)
c
c       GWT    :  The 'NMUG' angle cosine quadrature weights on (0,1)
c
c   Called by- CHEKIN
c   Calls- QGAUSN, ERRMSG, BDREF
c +--------------------------------------------------------------------+

c     .. Parameters ..

      INTEGER   NMUG
      PARAMETER ( NMUG = 50 )
c     ..
c     .. Scalar Arguments ..

!      REAL      MU, WVNMHI, WVNMLO
      REAL      MU
c     ..
c     .. Local Scalars ..

      LOGICAL   PASS1
      INTEGER   K
      REAL      PI
c     ..
c     .. Local Arrays ..

      REAL      GMU( NMUG ), GWT( NMUG )
c     ..
c     .. External Functions ..

      REAL      BDREF
      EXTERNAL  BDREF
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG, QGAUSN
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, ASIN
c     ..
      SAVE      PASS1, GMU, GWT, PI
      DATA      PASS1 / .True. /


      IF( PASS1 ) THEN

         PASS1 = .FALSE.
         PI   = 2.*ASIN( 1.0 )

         CALL QGAUSN( NMUG/2, GMU, GWT )

         DO 10 K = 1, NMUG / 2
            GMU( K + NMUG/2 ) = -GMU( K )
            GWT( K + NMUG/2 ) = GWT( K )
   10    CONTINUE

      END IF

      IF( ABS( MU ).GT.1.0 )
     &    CALL ERRMSG( 'DREF--input argument error(s)',.True. )

      DREF = 0.0

c                       ** Loop over azimuth angle difference
c      DO 30 JG = 1, NMUG
c
c         SUM  = 0.0
c                       ** Loop over angle of reflection
c         DO 20 K = 1, NMUG / 2
c            SUM  = SUM + GWT( K )*GMU( K )*
c     &             BDREF( WVNMLO, WVNMHI, GMU( K ), MU, PI*GMU( JG ) )
c   20    CONTINUE
c
c         DREF = DREF + GWT( JG )*SUM
c
c   30 CONTINUE
c
c      IF( DREF.LT.0.0 .OR. DREF.GT.1.0 )
c     &    CALL ERRMSG( 'DREF--albedo value not in (0,1)',.False. )

      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE LEPOLY( NMU, M, MAXMU, TWONM1, MU, SQT, YLM )

c       Computes the normalized associated Legendre polynomial,
c       defined in terms of the associated Legendre polynomial
c       Plm = P-sub-l-super-m as
c
c             Ylm(MU) = sqrt( (l-m)!/(l+m)! ) * Plm(MU)
c
c       for fixed order m and all degrees from l = m to TWONM1.
c       When m.GT.0, assumes that Y-sub(m-1)-super(m-1) is available
c       from a prior call to the routine.
c
c       REFERENCE: Dave, J.V. and B.H. Armstrong, Computations of
c                  High-Order Associated Legendre Polynomials,
c                  J. Quant. Spectrosc. Radiat. Transfer 10,
c                  557-562, 1970.  (hereafter D/A)
c
c       METHOD: Varying degree recurrence relationship.
c
c       NOTES:
c       (1) The D/A formulas are transformed by setting M=n-1; L=k-1.
c       (2) Assumes that routine is called first with  M = 0, then with
c           M = 1, etc. up to  M = TWONM1.
c
c
c  I N P U T     V A R I A B L E S:
c
c       NMU    :  Number of arguments of YLM
c
c       M      :  Order of YLM
c
c       MAXMU  :  First dimension of YLM
c
c       TWONM1 :  Max degree of YLM
c
c       MU(i)  :  Arguments of YLM (i = 1 to NMU)
c
c       SQT(k) :  Square root of k
c
c       If M.GT.0, YLM(M-1,i) for i = 1 to NMU is assumed to exist
c       from a prior call.
c
c
c  O U T P U T     V A R I A B L E:
c
c       YLM(l,i) :  l = M to TWONM1, normalized associated Legendre
c                   polynomials evaluated at argument MU(i)
c
c   Called by- DISORT, ALBTRN
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   M, MAXMU, NMU, TWONM1
c     ..
c     .. Array Arguments ..

      REAL      MU( * ), YLM( 0:MAXMU, * ), SQT( * )
c     ..
c     .. Local Scalars ..

      INTEGER   I, L
      REAL      TMP1, TMP2
c     ..


      IF( M.EQ.0 ) THEN
c                             ** Upward recurrence for ordinary
c                             ** Legendre polynomials
         DO 20 I = 1, NMU
            YLM( 0, I ) = 1.0
            YLM( 1, I ) = MU( I )
   20    CONTINUE


         DO 40 L = 2, TWONM1

            DO 30 I = 1, NMU
               YLM( L, I ) = ( ( 2*L - 1 )*MU( I )*YLM( L-1, I ) -
     &                         ( L - 1 )*YLM( L-2, I ) ) / L
   30       CONTINUE

   40    CONTINUE


      ELSE

         DO 50 I = 1, NMU
c                               ** Y-sub-m-super-m; derived from
c                               ** D/A Eqs. (11,12), STWL(58c)

            YLM( M, I ) = - SQT( 2*M - 1 ) / SQT( 2*M )*
     &                      SQRT( 1.- MU(I)**2 )*YLM( M-1, I )

c                              ** Y-sub-(m+1)-super-m; derived from
c                              ** D/A Eqs.(13,14) using Eqs.(11,12),
c                              ** STWL(58f)

            YLM( M+1, I ) = SQT( 2*M + 1 )*MU( I )*YLM( M, I )

   50    CONTINUE

c                                   ** Upward recurrence; D/A EQ.(10),
c                                   ** STWL(58a)
         DO 70 L = M + 2, TWONM1

            TMP1  = SQT( L - M )*SQT( L + M )
            TMP2  = SQT( L - M - 1 )*SQT( L + M - 1 )

            DO 60 I = 1, NMU
               YLM( L, I ) = ( ( 2*L - 1 )*MU( I )*YLM( L-1, I ) -
     &                         TMP2*YLM( L-2, I ) ) / TMP1
   60       CONTINUE

   70    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      SUBROUTINE LEPOLY0(  M, MAXMU, TWONM1, MU, SQT, YLM )
C       Special version of LEPOLY, now MU is a scalar
c       Computes the normalized associated Legendre polynomial,
c       defined in terms of the associated Legendre polynomial
c       Plm = P-sub-l-super-m as
c
c             Ylm(MU) = sqrt( (l-m)!/(l+m)! ) * Plm(MU)
c
c       for fixed order m and all degrees from l = m to TWONM1.
c       When m.GT.0, assumes that Y-sub(m-1)-super(m-1) is available
c       from a prior call to the routine.
c
c       REFERENCE: Dave, J.V. and B.H. Armstrong, Computations of
c                  High-Order Associated Legendre Polynomials,
c                  J. Quant. Spectrosc. Radiat. Transfer 10,
c                  557-562, 1970.  (hereafter D/A)
c
c       METHOD: Varying degree recurrence relationship.
c
c       NOTES:
c       (1) The D/A formulas are transformed by setting M=n-1; L=k-1.
c       (2) Assumes that routine is called first with  M = 0, then with
c           M = 1, etc. up to  M = TWONM1.
c
c
c  I N P U T     V A R I A B L E S:
c
c       NMU    :  Number of arguments of YLM
c
c       M      :  Order of YLM
c
c       MAXMU  :  First dimension of YLM
c
c       TWONM1 :  Max degree of YLM
c
c       MU  :  Arguments of YLM (i = 1 to NMU)
c
c       SQT(k) :  Square root of k
c
c       If M.GT.0, YLM(M-1,i) for i = 1 to NMU is assumed to exist
c       from a prior call.
c
c
c  O U T P U T     V A R I A B L E:
c
c       YLM(l) :  l = M to TWONM1, normalized associated Legendre
c                   polynomials evaluated at argument MU
c   Called by- DISORT, ALBTRN
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   M, MAXMU, TWONM1
c     ..
c     .. Array Arguments ..

      REAL      MU, YLM( 0:MAXMU ), SQT( * )
c     ..
c     .. Local Scalars ..

      INTEGER   L
      REAL      TMP1, TMP2
c     ..


      IF( M.EQ.0 ) THEN
c                             ** Upward recurrence for ordinary
c                             ** Legendre polynomials
c         DO 20 I = 1, NMU
            YLM( 0 ) = 1.0
            YLM( 1 ) = MU
c   20    CONTINUE


         DO 40 L = 2, TWONM1

c            DO 30 I = 1, NMU
               YLM( L) = ( ( 2*L - 1 )*MU*YLM( L-1 ) -
     &                         ( L - 1 )*YLM( L-2 ) ) / L
c   30       CONTINUE

   40    CONTINUE


      ELSE

c         DO 50 I = 1, NMU
c                               ** Y-sub-m-super-m; derived from
c                               ** D/A Eqs. (11,12), STWL(58c)

            YLM( M ) = - SQT( 2*M - 1 ) / SQT( 2*M )*
     &                      SQRT( 1.- MU**2 )*YLM( M-1 )

c                              ** Y-sub-(m+1)-super-m; derived from
c                              ** D/A Eqs.(13,14) using Eqs.(11,12),
c                              ** STWL(58f)

            YLM( M+1 ) = SQT( 2*M + 1 )*MU*YLM( M )

c   50    CONTINUE

c                                   ** Upward recurrence; D/A EQ.(10),
c                                   ** STWL(58a)
         DO 70 L = M + 2, TWONM1

            TMP1  = SQT( L - M )*SQT( L + M )
            TMP2  = SQT( L - M - 1 )*SQT( L + M - 1 )

c            DO 60 I = 1, NMU
               YLM( L ) = ( ( 2*L - 1 )*MU*YLM( L-1 ) -
     &                         TMP2*YLM( L-2 ) ) / TMP1
c   60       CONTINUE

   70    CONTINUE

      END IF


      RETURN
      END

c ---------------------------------------------------------------------
      REAL FUNCTION PLKAVG( WNUMLO, WNUMHI, T )

c        Computes Planck function integrated between two wavenumbers
c
c  INPUT :  WNUMLO : Lower wavenumber (inv cm) of spectral interval
c
c           WNUMHI : Upper wavenumber
c
c           T      : Temperature (K)
c
c  OUTPUT : PLKAVG : Integrated Planck function ( Watts/sq m )
c                      = Integral (WNUMLO to WNUMHI) of
c                        2h c**2  nu**3 / ( EXP(hc nu/kT) - 1)
c                        (where h=Plancks constant, c=speed of
c                         light, nu=wavenumber, T=temperature,
c                         and k = Boltzmann constant)
c
c  Reference : Specifications of the Physical World: New Value
c                 of the Fundamental Constants, Dimensions/N.B.S.,
c                 Jan. 1974
c
c  Method :  For WNUMLO close to WNUMHI, a Simpson-rule quadrature
c            is done to avoid ill-conditioning; otherwise
c
c            (1)  For WNUMLO or WNUMHI small,
c                 integral(0 to WNUMLO/HI) is calculated by expanding
c                 the integrand in a power series and integrating
c                 term by term;
c
c            (2)  Otherwise, integral(WNUMLO/HI to INFINITY) is
c                 calculated by expanding the denominator of the
c                 integrand in powers of the exponential and
c                 integrating term by term.
c
c  Accuracy :  At least 6 significant digits, assuming the
c              physical constants are infinitely accurate
c
c  ERRORS WHICH ARE NOT TRAPPED:
c
c      * power or exponential series may underflow, giving no
c        significant digits.  This may or may not be of concern,
c        depending on the application.
c
c      * Simpson-rule special case is skipped when denominator of
c        integrand will cause overflow.  In that case the normal
c        procedure is used, which may be inaccurate if the
c        wavenumber limits (WNUMLO, WNUMHI) are close together.
c
c  LOCAL VARIABLES
c
c        A1,2,... :  Power series coefficients
c        C2       :  h * c / k, in units cm*K (h = Plancks constant,
c                      c = speed of light, k = Boltzmann constant)
c        D(I)     :  Exponential series expansion of integral of
c                       Planck function from WNUMLO (i=1) or WNUMHI
c                       (i=2) to infinity
c        EPSIL    :  Smallest number such that 1+EPSIL .GT. 1 on
c                       computer
c        EX       :  EXP( - V(I) )
c        EXM      :  EX**M
c        MMAX     :  No. of terms to take in exponential series
c        MV       :  Multiples of V(I)
c        P(I)     :  Power series expansion of integral of
c                       Planck function from zero to WNUMLO (I=1) or
c                       WNUMHI (I=2)
c        PI       :  3.14159...
c        SIGMA    :  Stefan-Boltzmann constant (W/m**2/K**4)
c        SIGDPI   :  SIGMA / PI
c        SMALLV   :  Number of times the power series is used (0,1,2)
c        V(I)     :  C2 * (WNUMLO(I=1) or WNUMHI(I=2)) / temperature
c        VCUT     :  Power-series cutoff point
c        VCP      :  Exponential series cutoff points
c        VMAX     :  Largest allowable argument of EXP function
c
c   Called by- DISORT
c   Calls- R1MACH, ERRMSG
c ----------------------------------------------------------------------

c     .. Parameters ..

      REAL      A1, A2, A3, A4, A5, A6
      PARAMETER ( A1 = 1. / 3., A2 = -1. / 8., A3 = 1. / 60.,
     &          A4 = -1. / 5040., A5 = 1. / 272160.,
     &          A6 = -1. / 13305600. )
c     ..
c     .. Scalar Arguments ..

      REAL      T, WNUMHI, WNUMLO
c     ..
c     .. Local Scalars ..

      INTEGER   I, K, M, MMAX, N, SMALLV
      REAL      C2, CONC, DEL, EPSIL, EX, EXM, HH, MV, OLDVAL, PI,
     &          SIGDPI, SIGMA, VAL, VAL0, VCUT, VMAX, VSQ, X
c     ..
c     .. Local Arrays ..

      REAL      D( 2 ), P( 2 ), V( 2 ), VCP( 7 )
c     ..
c     .. External Functions ..

      REAL      R1MACH
      EXTERNAL  R1MACH
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..

!      INTRINSIC ABS, ASIN, EXP, LOG, MOD
      INTRINSIC ABS, ASIN, LOG, MOD
c     ..
c     .. Statement Functions ..

      REAL      PLKF
c     ..
      SAVE      PI, CONC, VMAX, EPSIL, SIGDPI

      DATA      C2 / 1.438786 / , SIGMA / 5.67032E-8 / , VCUT / 1.5 / ,
     &          VCP / 10.25, 5.7, 3.9, 2.9, 2.3, 1.9, 0.0 /
      DATA      PI / 0.0 /

c     .. Statement Function definitions ..

      PLKF( X ) = X**3 / ( EXP( X ) - 1 )
c     ..

      P( 1 ) = 0.0
      P( 2 ) = 0.0
      D( 1 ) = 0.0
      D( 2 ) = 0.0
      
      IF( PI .EQ. 0.0 ) THEN

         PI     = 2.*ASIN( 1.0 )
         VMAX   = LOG( R1MACH( 2 ) )
         EPSIL  = R1MACH( 4 )
         SIGDPI = SIGMA / PI
         CONC   = 15. / PI**4

      END IF


      IF( T.LT.0.0 .OR. WNUMHI.LE.WNUMLO .OR. WNUMLO.LT.0. )
     &    CALL ERRMSG('PLKAVG--temperature or wavenums. wrong',.TRUE.)


      IF( T .LT. 1.E-4 ) THEN

         PLKAVG = 0.0
         RETURN

      END IF


      V( 1 ) = C2*WNUMLO / T
      V( 2 ) = C2*WNUMHI / T
!      PRINT*, V 
      IF( V( 1 ).GT.EPSIL .AND. V( 2 ).LT.VMAX .AND.
     &    ( WNUMHI - WNUMLO ) / WNUMHI .LT. 1.E-2 ) THEN

c                          ** Wavenumbers are very close.  Get integral
c                          ** by iterating Simpson rule to convergence.

         HH     = V( 2 ) - V( 1 )
         OLDVAL = 0.0
         VAL0   = PLKF( V( 1 ) ) + PLKF( V( 2 ) )

         DO 20 N = 1, 10

            DEL  = HH / ( 2*N )
            VAL  = VAL0

            DO 10 K = 1, 2*N - 1
               VAL  = VAL + 2*( 1 + MOD( K,2 ) )*
     &                      PLKF( V( 1 ) + K*DEL )
   10       CONTINUE

            VAL  = DEL / 3.*VAL
            IF( ABS( ( VAL - OLDVAL ) / VAL ).LE.1.E-6 ) GO TO  30
            OLDVAL = VAL

   20    CONTINUE

         CALL ERRMSG( 'PLKAVG--Simpson rule didnt converge',.FALSE.)

   30    CONTINUE

         PLKAVG = SIGDPI * T**4 * CONC * VAL

         RETURN

      END IF

c                          *** General case ***
      SMALLV = 0

      DO 60 I = 1, 2

         IF( V( I ).LT.VCUT ) THEN
c                                   ** Use power series
            SMALLV = SMALLV + 1
            VSQ    = V( I )**2
            P( I ) = CONC*VSQ*V( I )*( A1 +
     &               V( I )*( A2 + V( I )*( A3 + VSQ*( A4 + VSQ*( A5 +
     &               VSQ*A6 ) ) ) ) )

         ELSE
c                      ** Use exponential series
            MMAX  = 0
c                                ** Find upper limit of series
   40       CONTINUE
            MMAX  = MMAX + 1

            IF( V(I) .LT. VCP( MMAX ) ) GO TO  40
!            print*, I, V(I), EX
!            PRINT*, LOG(R1MACH(4))
!            print*, log(r1mach(1))
            IF( V(I) .LT. -LOG(R1MACH(1)) ) THEN 
                EX     = EXP( - V(I) )
            ELSE
!            EX     = EXP( LOG(R1MACH(4)) )
              EX = 0.0;
            ENDIF
            EXM    = 1.0
            D( I ) = 0.0

            DO 50 M = 1, MMAX
               MV     = M*V( I )
               EXM    = EX*EXM
               D( I ) = D( I ) + EXM*( 6.+ MV*( 6.+ MV*( 3.+ MV ) ) )
     &                  / M**4
   50       CONTINUE

            D( I ) = CONC*D( I )

         END IF

   60 CONTINUE

c                              ** Handle ill-conditioning
      IF( SMALLV.EQ.2 ) THEN
c                                    ** WNUMLO and WNUMHI both small
         PLKAVG = P( 2 ) - P( 1 )

      ELSE IF( SMALLV.EQ.1 ) THEN
c                                    ** WNUMLO small, WNUMHI large
         PLKAVG = 1.- P( 1 ) - D( 2 )

      ELSE
c                                    ** WNUMLO and WNUMHI both large
         PLKAVG = D( 1 ) - D( 2 )

      END IF

      PLKAVG = SIGDPI * T**4 * PLKAVG

      IF( PLKAVG.EQ.0.0 )
     &    CALL ERRMSG('PLKAVG--returns zero; possible underflow',
     &    .FALSE.)


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE PRAVIN( UMU, NUMU, MXUMU, UTAU, NTAU, U0U )

c        Print azimuthally averaged intensities at user angles
c
c   Called by- DISORT
c
c     LENFMT   Max number of polar angle cosines UMU that can be
c              printed on one line, as set in FORMAT statement
c --------------------------------------------------------------------

c     .. Scalar Arguments ..

      INTEGER   MXUMU, NTAU, NUMU
c     ..
c     .. Array Arguments ..

      REAL      U0U( MXUMU, * ), UMU( NUMU ), UTAU( NTAU )
c     ..
c     .. Local Scalars ..

      INTEGER   IU, IUMAX, IUMIN, LENFMT, LU, NP, NPASS
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC MIN
c     ..


      IF( NUMU.LT.1 )  RETURN

      WRITE( *, '(//,A)' )
     &   ' *******  AZIMUTHALLY AVERAGED INTENSITIES ' //
     &   '(at user polar angles)  ********'

      LENFMT = 8
      NPASS  = 1 + (NUMU-1) / LENFMT

      WRITE( *,'(/,A,/,A)') '   Optical   Polar Angle Cosines',
     &                      '     Depth'

      DO 20 NP = 1, NPASS

         IUMIN  = 1 + LENFMT * ( NP - 1 )
         IUMAX  = MIN( LENFMT*NP, NUMU )
         WRITE( *,'(/,10X,8F14.5)') ( UMU(IU), IU = IUMIN, IUMAX )

         DO 10 LU = 1, NTAU
            WRITE( *, '(0P,F10.4,1P,8E14.4)' ) UTAU( LU ),
     &           ( U0U( IU,LU ), IU = IUMIN, IUMAX )
   10    CONTINUE

   20 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE PRTINP( NLYR, DTAUC, DTAUCP, SSALB, NMOM, PMOM, TEMPER,
     &                   WVNMLO, WVNMHI, NTAU, UTAU, NSTR, NUMU, UMU,
     &                   NPHI, PHI, IBCND, FBEAM, UMU0, PHI0, FISOT,
     &                   LAMBER, ALBEDO, BTEMP, TTEMP, TEMIS, DELTAM,
     &                   PLANK, ONLYFL, CORINT, ACCUR, FLYR, LYRCUT,
     &                   OPRIM, TAUC, TAUCPR, MAXMOM, PRTMOM,
     &                   DO_PSEUDO_SPHERE, H_LYR, DELTAMPLUS )

c        Print values of input variables
c
c   Called by- DISORT
c --------------------------------------------------------------------

c     .. Scalar Arguments ..

      LOGICAL   CORINT, DELTAM, LAMBER, LYRCUT, ONLYFL, PLANK, PRTMOM
      LOGICAL   DO_PSEUDO_SPHERE, DELTAMPLUS
      INTEGER   IBCND, MAXMOM, NLYR, NMOM, NPHI, NSTR, NTAU, NUMU
      REAL      ACCUR, ALBEDO, BTEMP, FBEAM, FISOT, PHI0, TEMIS, TTEMP,
     &          UMU0, WVNMHI, WVNMLO
c     ..
c     .. Array Arguments ..

      REAL      DTAUC( * ), DTAUCP( * ), FLYR( * ), OPRIM( * ),
     &          PHI( * ), PMOM( 0:MAXMOM, * ), SSALB( * ), TAUC( 0:* ),
     &          TAUCPR( 0:* ), TEMPER( 0:* ), UMU( * ), UTAU( * ),
     &          H_LYR( * ) 
c     ..
c     .. Local Scalars ..

      INTEGER   IU, J, K, LC, LU
      REAL      YESSCT
c     ..


      WRITE( *, '(/,A,I4,A,I4)' ) ' No. streams =', NSTR,
     &       '     No. computational layers =', NLYR

      IF( IBCND.NE.1 ) WRITE( *, '(I4,A,10F10.4,/,(26X,10F10.4))' )
     &    NTAU, ' User optical depths :', ( UTAU(LU), LU = 1, NTAU )

      IF( .NOT.ONLYFL ) WRITE( *, '(I4,A,10F9.5,/,(31X,10F9.5))' )
     &    NUMU, ' User polar angle cosines :', ( UMU(IU), IU = 1, NUMU )

      IF( .NOT.ONLYFL .AND. IBCND.NE.1 )
     &    WRITE( *, '(I4,A,10F9.2,/,(28X,10F9.2))' )
     &           NPHI,' User azimuthal angles :',( PHI(J), J = 1, NPHI )

      IF( .NOT.PLANK .OR. IBCND.EQ.1 )
     &    WRITE( *, '(A)' ) ' No thermal emission'


      WRITE( *, '(A,I2)' ) ' Boundary condition flag: IBCND =', IBCND

      IF( IBCND.EQ.0 ) THEN

         WRITE( *, '(A,1P,E11.3,A,0P,F8.5,A,F7.2,/,A,1P,E11.3)' )
     &          '    Incident beam with intensity =', FBEAM,
     &          ' and polar angle cosine = ', UMU0,
     &          '  and azimuth angle =', PHI0,
     &          '    plus isotropic incident intensity =', FISOT

         IF( LAMBER ) WRITE( *, '(A,0P,F8.4)' )
     &                '    Bottom albedo (Lambertian) =', ALBEDO

         IF( .NOT.LAMBER ) WRITE( *, '(A)' )
     &       '    Bidirectional reflectivity at bottom'

         IF( PLANK ) WRITE( *, '(A,2F14.4,/,A,F10.2,A,F10.2,A,F8.4)' )
     &       '    Thermal emission in wavenumber interval :', WVNMLO,
     &       WVNMHI,
     &       '    Bottom temperature =', BTEMP,
     &       '    Top temperature =', TTEMP,
     &       '    Top emissivity =', TEMIS

      ELSE IF( IBCND.EQ.1 ) THEN

         WRITE( *, '(A)' )
     &          '    Isotropic illumination from top and bottom'
         WRITE( *, '(A,0P,F8.4)' )
     &          '    Bottom albedo (Lambertian) =', ALBEDO

      END IF


      IF( DELTAM ) WRITE( *, '(A)' ) ' Uses delta-M method'
      IF( DELTAMPLUS ) WRITE( *, '(A)' ) ' Uses New-Delta-M+ method'
      IF( .NOT.DELTAM .AND. .NOT. DELTAMPLUS ) 
     & WRITE( *, '(A)' ) ' Does not use delta-M / delta-M Plus method'

      IF( CORINT ) WRITE( *, '(A)' ) ' Uses TMS/IMS method'
      IF( .NOT.CORINT ) WRITE( *,'(A)' ) ' Does not use TMS/IMS method'

      IF(DO_PSEUDO_SPHERE) THEN
        WRITE(*,'(A)') ' Uses pseudo spherical method'
        WRITE(*,'(A15, 13(F5.2,2x))')' Layer height: ',
     &                               (H_LYR(LC),LC=1,NLYR)
      ENDIF
      IF(.NOT.DO_PSEUDO_SPHERE) WRITE(*, '(A)' ) ' Uses plane'//
     & ' parallel method'



      IF( IBCND.EQ.1 ) THEN

         WRITE( *, '(A)' ) ' Calculate albedo and transmissivity of'//
     &                     ' medium vs. incident beam angle'

      ELSE IF( ONLYFL ) THEN

         WRITE( *, '(A)' )
     &          ' Calculate fluxes only'

      ELSE

         WRITE( *, '(A)' ) ' Calculate fluxes and intensities'

      END IF

      WRITE( *, '(A,1P,E11.2)' )
     &       ' Relative convergence criterion for azimuth series =',
     &       ACCUR

      IF( LYRCUT ) WRITE( *, '(A)' )
     &    ' Sets radiation = 0 below absorption optical depth 10'


c                                    ** Print layer variables
c                                    ** (to read, skip every other line)

      IF( PLANK ) WRITE( *, '(/,37X,A,3(/,2A))' )
     &  '<------------- Delta-M --------------->',
     &  '                   Total    Single                           ',
     &  'Total    Single',
     &  '       Optical   Optical   Scatter   Separated   ',
     &  'Optical   Optical   Scatter    Asymm',
     &  '         Depth     Depth    Albedo    Fraction     ',
     &  'Depth     Depth    Albedo   Factor   Temperature'

      IF( .NOT.PLANK ) WRITE( *, '(/,37X,A,3(/,2A))' )
     &  '<------------- Delta-M --------------->',
     &  '                   Total    Single                           ',
     &  'Total    Single',
     &  '       Optical   Optical   Scatter   Separated   ',
     &  'Optical   Optical   Scatter    Asymm',
     &  '         Depth     Depth    Albedo    Fraction     ',
     &  'Depth     Depth    Albedo   Factor'


      YESSCT = 0.0

      DO 10 LC = 1, NLYR

         YESSCT = YESSCT + SSALB( LC )
c                                       ** f90 nonadvancing I/O would
c                                       ** simplify this a lot (also the
c                                       ** two WRITEs above)
         IF( PLANK )
     &       WRITE( *,'(I4,2F10.4,F10.5,F12.5,2F10.4,F10.5,F9.4,F14.3)')
     &             LC, DTAUC( LC ), TAUC( LC ), SSALB( LC ), FLYR( LC ),
     &             DTAUCP( LC ), TAUCPR( LC ), OPRIM( LC ),
     &             PMOM( 1,LC ), TEMPER( LC-1 )

         IF( .NOT.PLANK )
     &       WRITE( *,'(I4,2F10.4,F10.5,F12.5,2F10.4,F10.5,F9.4)' )
     &             LC, DTAUC( LC ), TAUC( LC ), SSALB( LC ), FLYR( LC ),
     &             DTAUCP( LC ), TAUCPR( LC ), OPRIM( LC ), PMOM( 1,LC )
   10 CONTINUE

      IF( PLANK ) WRITE( *, '(85X,F14.3)' ) TEMPER( NLYR )


      IF( PRTMOM .AND. YESSCT.GT.0.0 ) THEN

         WRITE( *, '(/,A,I5)' ) ' Number of Phase Function Moments = ',
     &        NMOM + 1
         WRITE( *, '(A)' ) ' Layer   Phase Function Moments'

         DO 20 LC = 1, NLYR

            IF( SSALB( LC ).GT.0.0 )
     &          WRITE( *, '(I6,10F11.6,/,(6X,10F11.6))' )
     &                 LC, ( PMOM( K, LC ), K = 0, NMOM )
   20    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE PRTINT( UU, UTAU, NTAU, UMU, NUMU, PHI, NPHI, MAXULV,
     &                   MAXUMU )

c     Prints the intensity at user polar and azimuthal angles
c
c     All arguments are DISORT input or output variables
c
c     Called by- DISORT
c
c     LENFMT   Max number of azimuth angles PHI that can be printed
c                on one line, as set in FORMAT statement
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAXULV, MAXUMU, NPHI, NTAU, NUMU
c     ..
c     .. Array Arguments ..

      REAL      PHI( * ), UMU( * ), UTAU( * ), UU( MAXUMU, MAXULV, * )
c     ..
c     .. Local Scalars ..

      INTEGER   IU, J, JMAX, JMIN, LENFMT, LU, NP, NPASS
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC MIN
c     ..


      IF( NPHI.LT.1 )  RETURN

      WRITE( *, '(//,A)' )
     &   ' *********  I N T E N S I T I E S  *********'

      LENFMT = 10
      NPASS  = 1 + (NPHI-1) / LENFMT

      WRITE( *, '(/,A,/,A,/,A)' )
     &   '             Polar   Azimuth angles (degrees)',
     &   '   Optical   Angle',
     &   '    Depth   Cosine'

      DO 30 LU = 1, NTAU

         DO 20 NP = 1, NPASS

            JMIN   = 1 + LENFMT * ( NP - 1 )
            JMAX   = MIN( LENFMT*NP, NPHI )

            WRITE( *, '(/,18X,10F11.2)' ) ( PHI(J), J = JMIN, JMAX )

            IF( NP.EQ.1 ) WRITE( *, '(F10.4,F8.4,1P,10E11.3)' )
     &             UTAU(LU), UMU(1), (UU(1, LU, J), J = JMIN, JMAX)
            IF( NP.GT.1 ) WRITE( *, '(10X,F8.4,1P,10E11.3)' )
     &                       UMU(1), (UU(1, LU, J), J = JMIN, JMAX)

            DO 10 IU = 2, NUMU
               WRITE( *, '(10X,F8.4,1P,10E11.3)' )
     &                 UMU( IU ), ( UU( IU, LU, J ), J = JMIN, JMAX )
   10       CONTINUE

   20    CONTINUE

   30 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE QGAUSN( M, GMU, GWT )

c       Compute weights and abscissae for ordinary Gaussian quadrature
c       on the interval (0,1); that is, such that
c
c       sum(i=1 to M) ( GWT(i) f(GMU(i)) )
c
c       is a good approximation to
c
c       integral(0 to 1) ( f(x) dx )
c
c   INPUT :    M       order of quadrature rule
c
c   OUTPUT :  GMU(I)   array of abscissae (I = 1 TO M)
c             GWT(I)   array of weights (I = 1 TO M)
c
c   REFERENCE:  Davis, P.J. and P. Rabinowitz, Methods of Numerical
c               Integration, Academic Press, New York, pp. 87, 1975
c
c   METHOD:  Compute the abscissae as roots of the Legendre
c            polynomial P-sub-M using a cubically convergent
c            refinement of Newton's method.  Compute the
c            weights from EQ. 2.7.3.8 of Davis/Rabinowitz.  Note
c            that Newton's method can very easily diverge; only a
c            very good initial guess can guarantee convergence.
c            The initial guess used here has never led to divergence
c            even for M up to 1000.
c
c   ACCURACY:  relative error no better than TOL or computer
c              precision (machine epsilon), whichever is larger
c
c   INTERNAL VARIABLES:
c
c    ITER      : number of Newton Method iterations
c    MAXIT     : maximum allowed iterations of Newton Method
c    PM2,PM1,P : 3 successive Legendre polynomials
c    PPR       : derivative of Legendre polynomial
c    P2PRI     : 2nd derivative of Legendre polynomial
c    TOL       : convergence criterion for Legendre poly root iteration
c    X,XI      : successive iterates in cubically-convergent version
c                of Newtons Method (seeking roots of Legendre poly.)
c
c   Called by- DREF, SETDIS, SURFAC
c   Calls- D1MACH, ERRMSG
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   M
c     ..
c     .. Array Arguments ..

      REAL      GMU( M ), GWT( M )
c     ..
c     .. Local Scalars ..

      INTEGER   ITER, K, LIM, MAXIT, NN, NP1
      DOUBLE PRECISION CONA, PI, T
      DOUBLE PRECISION EN, NNP1, ONE, P, P2PRI, PM1, PM2, PPR, PROD,
     &                 TMP, TOL, TWO, X, XI
c     ..
c     .. External Functions ..

      DOUBLE PRECISION D1MACH
      EXTERNAL  D1MACH
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, ASIN, COS, FLOAT, MOD, TAN
c     ..
      SAVE      PI, TOL

      DATA      PI / 0.D0 / , MAXIT / 1000 / , ONE / 1.D0 / ,
     &          TWO / 2.D0 /

      IF( PI.EQ.0.0 ) THEN
         PI   = 2.D0*DASIN( 1.D0 )
         TOL  = 10.*D1MACH( 4 )
      END IF

      IF( M.LT.1 ) CALL ERRMSG( 'QGAUSN--Bad value of M',.True.)

      IF( M.EQ.1 ) THEN
         GMU( 1 ) = 0.5
         GWT( 1 ) = 1.0
         RETURN
      END IF

      EN   = DBLE(M)
      NP1  = M + 1
      NNP1 = DBLE(M*NP1)
      CONA = DBLE( M - 1 ) / ( 8*M**3 )

      LIM  = M / 2

      DO 30 K = 1, LIM
c                                        ** Initial guess for k-th root
c                                        ** of Legendre polynomial, from
c                                        ** Davis/Rabinowitz (2.7.3.3a)
         T  = ( 4*K - 1 )*PI / ( 4*M + 2 )
         X  = DCOS( T + CONA / DTAN( T ) )
         ITER = 0
c                                        ** Upward recurrence for
c                                        ** Legendre polynomials
   10    CONTINUE
         ITER   = ITER + 1
         PM2    = ONE
         PM1    = X

         P = 0.D0
         DO 20 NN = 2, M
            P    = ( ( 2*NN - 1 )*X*PM1 - ( NN - 1 )*PM2 ) / NN
            PM2  = PM1
            PM1  = P
   20    CONTINUE
c                                              ** Newton Method
         TMP    = ONE / ( ONE - X**2 )
         PPR    = EN*( PM2 - X*P )*TMP
         P2PRI  = ( TWO*X*PPR - NNP1*P )*TMP
         XI     = X - ( P / PPR )*( ONE +
     &            ( P / PPR )*P2PRI / ( TWO*PPR ) )

c                                              ** Check for convergence
         IF( DABS( XI - X ).GT.TOL ) THEN

            IF( ITER.GT.MAXIT )
     &          CALL ERRMSG( 'QGAUSN--max iteration count',.True.)

            X  = XI
            GO TO  10

         END IF
c                             ** Iteration finished--calculate weights,
c                             ** abscissae for (-1,1)
         GMU( K ) = - REAL( X )
         GWT( K ) = REAL( TWO / ( TMP*( EN*PM2 )**2 ) )
         GMU( NP1 - K ) = -GMU( K )
         GWT( NP1 - K ) = GWT( K )
   30 CONTINUE
c                                    ** Set middle abscissa and weight
c                                    ** for rules of odd order
      IF( MOD( M,2 ).NE.0 ) THEN

         GMU( LIM + 1 ) = 0.0
         PROD   = ONE

         DO 40 K = 3, M, 2
            PROD   = PROD * K / ( K - 1 )
   40    CONTINUE

         GWT( LIM + 1 ) = REAL( TWO / PROD**2 )
      END IF

c                                        ** Convert from (-1,1) to (0,1)
      DO 50 K = 1, M
         GMU( K ) = 0.5*GMU( K ) + 0.5
         GWT( K ) = 0.5*GWT( K )
   50 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      REAL FUNCTION RATIO( A, B )

c        Calculate ratio  A/B  with over- and under-flow protection
c        (thanks to Prof. Jeff Dozier for some suggestions here).
c        Since this routine takes two logs, it is no speed demon,
c        but it is invaluable for comparing results from two runs
c        of a program under development.
c
c        NOTE:  In Fortran90, built-in functions TINY and HUGE
c               can replace the R1MACH calls.
c
c   Called by- DISORT
c   Calls- R1MACH
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      REAL      A, B
c     ..
c     .. Local Scalars ..

      LOGICAL   PASS1
      REAL      ABSA, ABSB, HUGE, POWA, POWB, POWMAX, POWMIN, TINY
c     ..
c     .. External Functions ..

      REAL      R1MACH
      EXTERNAL  R1MACH
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, LOG10, SIGN
c     ..
c     .. Save statement ..

      SAVE      PASS1, TINY, HUGE, POWMAX, POWMIN
c     ..
c     .. Data statements ..

      DATA      PASS1 / .TRUE. /
c     ..


      IF( PASS1 ) THEN

         TINY   = R1MACH( 1 )
         HUGE   = R1MACH( 2 )
         POWMAX = LOG10( HUGE )
         POWMIN = LOG10( TINY )
         PASS1  = .FALSE.

      END IF


      IF( A.EQ.0.0 ) THEN

         IF( B.EQ.0.0 ) THEN

            RATIO  = 1.0

         ELSE

            RATIO  = 0.0

         END IF


      ELSE IF( B.EQ.0.0 ) THEN

         RATIO  = SIGN( HUGE, A )

      ELSE

         ABSA   = ABS( A )
         ABSB   = ABS( B )
         POWA   = LOG10( ABSA )
         POWB   = LOG10( ABSB )

         IF( ABSA.LT.TINY .AND. ABSB.LT.TINY ) THEN

            RATIO  = 1.0

         ELSE IF( POWA - POWB.GE.POWMAX ) THEN

            RATIO  = HUGE

         ELSE IF( POWA - POWB.LE.POWMIN ) THEN

            RATIO  = TINY

         ELSE

            RATIO  = ABSA / ABSB

         END IF
c                      ** DONT use old trick of determining sign
c                      ** from A*B because A*B may (over/under)flow

         IF( ( A.GT.0.0 .AND. B.LT.0.0 ) .OR.
     &       ( A.LT.0.0 .AND. B.GT.0.0 ) ) RATIO = -RATIO

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SLFTST( CORINT, ACCUR, ALBEDO, BTEMP, DELTAM, DTAUC,
     &                   FBEAM, FISOT, IBCND, LAMBER, NLYR, PLANK, NPHI,
     &                   NUMU, NSTR, NTAU, ONLYFL, PHI, PHI0, NMOM,
     &                   PMOM, PRNT, PRNTU0, SSALB, TEMIS, TEMPER,
     &                   TTEMP, UMU, USRANG, USRTAU, UTAU, UMU0, WVNMHI,
     &                   WVNMLO, COMPAR, FLUP, RFLDIR, RFLDN, UU,
     &                   DO_PSEUDO_SPHERE, DELTAMPLUS )

c       If  COMPAR = FALSE, save user input values that would otherwise
c       be destroyed and replace them with input values for self-test.
c       If  COMPAR = TRUE, compare self-test case results with correct
c       answers and restore user input values if test is passed.
c
c       (See file 'DISORT.doc' for variable definitions.)
c
c
c     I N T E R N A L    V A R I A B L E S:
c
c         ACC     Relative accuracy required for passing self-test
c
c         ERRORn  Relative errors in DISORT output variables
c
c         OK      Logical variable for determining failure of self-test
c
c         All variables ending in 'S' are temporary 'S'torage for input
c
c   Called by- DISORT
c   Calls- TSTBAD, ERRMSG
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      LOGICAL   COMPAR, CORINT, DELTAM, LAMBER, ONLYFL, PLANK, USRANG,
     &          USRTAU, DO_PSEUDO_SPHERE, DELTAMPLUS  
      INTEGER   IBCND, NLYR, NMOM, NPHI, NSTR, NTAU, NUMU
      REAL      ACCUR, ALBEDO, BTEMP, DTAUC, FBEAM, FISOT, FLUP, PHI,
     &          PHI0, RFLDIR, RFLDN, SSALB, TEMIS, TTEMP, UMU, UMU0,
     &          UTAU, UU, WVNMHI, WVNMLO
c     ..
c     .. Array Arguments ..

      LOGICAL   PRNT( * ), PRNTU0( * )
      REAL      PMOM( 0:* ), TEMPER( 0:* )
c     ..
c     .. Local Scalars ..

      LOGICAL   CORINS, DELTAS, deltasp, LAMBES, OK, ONLYFS, PLANKS, 
     &          USRANS, USRTAS, DO_PSEUDO_SPHERES
      INTEGER   I, IBCNDS, N, NLYRS, NMOMS, NPHIS, NSTRS, NTAUS, NUMUS
      REAL      ACC, ACCURS, ALBEDS, BTEMPS, DTAUCS, ERROR1, ERROR2,
     &          ERROR3, ERROR4, FBEAMS, FISOTS, PHI0S, PHIS, SSALBS,
     &          TEMISS, TTEMPS, UMU0S, UMUS, UTAUS, WVNMHS, WVNMLS
c     ..
c     .. Local Arrays ..

      LOGICAL   PRNTS( 5 ), PRNU0S( 2 )
      REAL      PMOMS( 0:5 ), TEMPES( 0:1 )
c     ..
c     .. External Functions ..

      LOGICAL   TSTBAD
      EXTERNAL  TSTBAD
c     ..
c     .. External Subroutines ..

      EXTERNAL  ERRMSG
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS
c     ..
      SAVE
      DATA      ACC / 1.E-4 /


      IF( .NOT.COMPAR ) THEN
c                                     ** Save user input values
         NLYRS  = NLYR
         DTAUCS = DTAUC
         SSALBS = SSALB

         DO 10 N = 0, 5
            PMOMS( N ) = PMOM( N )
   10    CONTINUE

         NSTRS  = NSTR
         NMOMS  = NMOM
         USRANS = USRANG
         NUMUS  = NUMU
         UMUS   = UMU
         USRTAS = USRTAU
         NTAUS  = NTAU
         UTAUS  = UTAU
         NPHIS  = NPHI
         PHIS   = PHI
         IBCNDS = IBCND
         FBEAMS = FBEAM
         UMU0S  = UMU0
         PHI0S  = PHI0
         FISOTS = FISOT
         LAMBES = LAMBER
         ALBEDS = ALBEDO
         DELTAS = DELTAM
         DELTASP = DELTAMPLUS 
         ONLYFS = ONLYFL
         CORINS = CORINT
         ACCURS = ACCUR
         PLANKS = PLANK
         WVNMLS = WVNMLO
         WVNMHS = WVNMHI
         BTEMPS = BTEMP
         TTEMPS = TTEMP
         TEMISS = TEMIS
         TEMPES( 0 ) = TEMPER( 0 )
         TEMPES( 1 ) = TEMPER( 1 )
         DO_PSEUDO_SPHERES = DO_PSEUDO_SPHERE 

         DO 20 I = 1, 5
            PRNTS( I ) = PRNT( I )
   20    CONTINUE

         DO 30 I = 1, 2
            PRNU0S( I ) = PRNTU0( I )
   30    CONTINUE

c                                     ** Set input values for self-test
         NSTR   = 4
         NLYR   = 1
         DTAUC  = 1.0
         SSALB  = 0.9
         NMOM   = 4
c                          ** Haze L moments
         PMOM( 0 ) = 1.0
         PMOM( 1 ) = 0.8042
         PMOM( 2 ) = 0.646094
         PMOM( 3 ) = 0.481851
         PMOM( 4 ) = 0.359056
!         PMOM( 5 ) = 0.0
         USRANG = .TRUE.
         NUMU   = 1
         UMU    = 0.5
         USRTAU = .TRUE.
         NTAU   = 1
         UTAU   = 0.5
         NPHI   = 1
         PHI    = 90.0
         IBCND  = 0
         FBEAM  = 3.14159265
         UMU0   = 0.866
         PHI0   = 0.0
         FISOT  = 1.0
         LAMBER = .TRUE.
         ALBEDO = 0.7
         DELTAM = .TRUE.
         DELTAMPLUS = .FALSE.
         ONLYFL = .FALSE.
         CORINT = .TRUE.
         ACCUR  = 1.E-4
         PLANK  = .TRUE.
         WVNMLO = 0.0
         WVNMHI = 50000.
         BTEMP  = 300.0
         TTEMP  = 100.0
         TEMIS  = 0.8
         TEMPER( 0 ) = 210.0
         TEMPER( 1 ) = 200.0
         DO_PSEUDO_SPHERE = .FALSE.

         DO 40 I = 1, 5
            PRNT( I ) = .FALSE.
   40    CONTINUE

         DO 50 I = 1, 2
            PRNTU0( I ) = .FALSE.
   50    CONTINUE


      ELSE
c                                    ** Compare test case results with
c                                    ** correct answers and abort if bad
         OK     = .TRUE.


         ERROR1 = ( UU - 47.865571 ) / 47.865571
         ERROR2 = ( RFLDIR - 1.527286 ) / 1.527286
         ERROR3 = ( RFLDN - 28.372225 ) / 28.372225
         ERROR4 = ( FLUP - 152.585284 ) / 152.585284

         IF( ABS( ERROR1 ).GT.ACC ) OK = TSTBAD( 'UU', ERROR1 )

         IF( ABS( ERROR2 ).GT.ACC ) OK = TSTBAD( 'RFLDIR', ERROR2 )

         IF( ABS( ERROR3 ).GT.ACC ) OK = TSTBAD( 'RFLDN', ERROR3 )

         IF( ABS( ERROR4 ).GT.ACC ) OK = TSTBAD( 'FLUP', ERROR4 )

         IF( .NOT.OK ) CALL ERRMSG( 'DISORT--self-test failed', .True. )

c                                      ** Restore user input values
         NLYR   = NLYRS
         DTAUC  = DTAUCS
         SSALB  = SSALBS

         DO 60 N = 0, 5
            PMOM( N ) = PMOMS( N )
   60    CONTINUE

         NSTR   = NSTRS
         NMOM   = NMOMS
         USRANG = USRANS
         NUMU   = NUMUS
         UMU    = UMUS
         USRTAU = USRTAS
         NTAU   = NTAUS
         UTAU   = UTAUS
         NPHI   = NPHIS
         PHI    = PHIS
         IBCND  = IBCNDS
         FBEAM  = FBEAMS
         UMU0   = UMU0S
         PHI0   = PHI0S
         FISOT  = FISOTS
         LAMBER = LAMBES
         ALBEDO = ALBEDS
         DELTAM = DELTAS
         DELTAMPLUS = DELTASP 
         ONLYFL = ONLYFS
         CORINT = CORINS
         ACCUR  = ACCURS
         PLANK  = PLANKS
         WVNMLO = WVNMLS
         WVNMHI = WVNMHS
         BTEMP  = BTEMPS
         TTEMP  = TTEMPS
         TEMIS  = TEMISS
         TEMPER( 0 ) = TEMPES( 0 )
         TEMPER( 1 ) = TEMPES( 1 )
         DO_PSEUDO_SPHERE = DO_PSEUDO_SPHERES 

         DO 70 I = 1, 5
            PRNT( I ) = PRNTS( I )
   70    CONTINUE

         DO 80 I = 1, 2
            PRNTU0( I ) = PRNU0S( I )
   80    CONTINUE

      END IF


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE ZEROAL( ND1, EXPBEA, FLYR, OPRIM, PHASA, PHAST, PHASM,
     &                        TAUCPR, XR0, XR1,
     &                   ND2, CMU, CWT, PSI0, PSI1, WK, Z0, Z1, ZJ,
     &                   ND3, YLM0,
     &                   ND4, ARRAY, CC, EVECC,
     &                   ND5, GL,
     &                   ND6, YLMC,
     &                   ND7, YLMU,
     &                   ND8, KK, LL, ZZ, ZPLK0, ZPLK1,
     &                   ND9, GC,
     &                   ND10, LAYRU, UTAUPR,
     &                   ND11, GU,
     &                   ND12, Z0U, Z1U, ZBEAM,
     &                   ND13, EVAL,
     &                   ND14, AMB, APB,
     &                   ND15, IPVT, Z,
     &                   ND16, RFLDIR, RFLDN, FLUP, UAVG, DFDT,
     &                   ND17, ALBMED, TRNMED,
     &                   ND18, U0U,
     &                   ND19, UU )

c         ZERO ARRAYS; NDn is dimension of all arrays following
c         it in the argument list
c
c   Called by- DISORT
c --------------------------------------------------------------------

c     .. Scalar Arguments ..

      INTEGER   ND1, ND10, ND11, ND12, ND13, ND14, ND15, ND16, ND17,
     &          ND18, ND19, ND2, ND3, ND4, ND5, ND6, ND7, ND8, ND9
c     ..
c     .. Array Arguments ..

      INTEGER   IPVT( * ), LAYRU( * )
      REAL      ALBMED( * ), AMB( * ), APB( * ), ARRAY( * ), CC( * ),
     &          CMU( * ), CWT( * ), DFDT( * ), EVAL( * ), EVECC( * ),
     &          EXPBEA( * ), FLUP( * ), FLYR( * ), GC( * ), GL( * ),
     &          GU( * ), KK( * ), LL( * ), OPRIM( * ), PHASA( * ),
     &          PHAST( * ), PHASM( * ), PSI0( * ), PSI1( * ),
     &          RFLDIR( * ), RFLDN( * ), TAUCPR( * ), TRNMED( * ),
     &          U0U( * ), UAVG( * ), UTAUPR( * ), UU( * ), WK( * ),
     &          XR0( * ), XR1( * ), YLM0( * ), YLMC( * ), Z( * ),
     &          Z0( * ), Z0U( * ), Z1( * ), Z1U( * ), YLMU( * ),
     &          ZBEAM( * ), ZJ( * ), ZPLK0( * ), ZPLK1( * ), ZZ( * )
c     ..
c     .. Local Scalars ..

      INTEGER   N
c     ..


      DO 10 N = 1, ND1
         EXPBEA( N ) = 0.0
         FLYR( N )   = 0.0
         OPRIM( N )  = 0.0
         PHASA( N )  = 0.0
         PHAST( N )  = 0.0
         PHASM( N )  = 0.0
         TAUCPR( N ) = 0.0
         XR0( N )    = 0.0
         XR1( N )    = 0.0
   10 CONTINUE

      DO 20 N = 1, ND2
         CMU( N )  = 0.0
         CWT( N )  = 0.0
         PSI0( N ) = 0.0
         PSI1( N ) = 0.0
         WK( N )   = 0.0
         Z0( N )   = 0.0
         Z1( N )   = 0.0
         ZJ( N )   = 0.0
   20 CONTINUE

      DO 30 N = 1, ND3
         YLM0( N ) = 0.0
   30 CONTINUE

      DO 40 N = 1, ND4
         ARRAY( N ) = 0.0
         CC( N )    = 0.0
         EVECC( N ) = 0.0
   40 CONTINUE

      DO 50 N = 1, ND5
         GL( N ) = 0.0
   50 CONTINUE

      DO 60 N = 1, ND6
         YLMC( N ) = 0.0
   60 CONTINUE

      DO 70 N = 1, ND7
         YLMU( N ) = 0.0
   70 CONTINUE

      DO 80 N = 1, ND8
         KK( N )    = 0.0
         LL( N )    = 0.0
         ZZ( N )    = 0.0
         ZPLK0( N ) = 0.0
         ZPLK1( N ) = 0.0
   80 CONTINUE

      DO 90 N = 1, ND9
         GC( N ) = 0.0
   90 CONTINUE

      DO 100 N = 1, ND10
         LAYRU( N )  = 0
         UTAUPR( N ) = 0.0
  100 CONTINUE

      DO 110 N = 1, ND11
         GU( N ) = 0.0
  110 CONTINUE

      DO 120 N = 1, ND12
         Z0U( N )   = 0.0
         Z1U( N )   = 0.0
         ZBEAM( N ) = 0.0
  120 CONTINUE

      DO 130 N = 1, ND13
         EVAL( N ) = 0.0
  130 CONTINUE

      DO 140 N = 1, ND14
         AMB( N ) = 0.0
         APB( N ) = 0.0
  140 CONTINUE

      DO 150 N = 1, ND15
         IPVT( N ) = 0
         Z( N )    = 0.0
  150 CONTINUE

      DO 160 N = 1, ND16
         RFLDIR( N ) = 0.
         RFLDN( N )  = 0.
         FLUP( N )   = 0.
         UAVG( N )   = 0.
         DFDT( N )   = 0.
  160 CONTINUE

      DO 170 N = 1, ND17
         ALBMED( N ) = 0.
         TRNMED( N ) = 0.
  170 CONTINUE

      DO 180 N = 1, ND18
         U0U( N ) = 0.
  180 CONTINUE

      DO 190 N = 1, ND19
         UU( N ) = 0.
  190 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


c ---------------------------------------------------------------------
      SUBROUTINE ZEROIT( A, LENGTH )

c         Zeros a real array A having LENGTH elements
c
c   Called by- DISORT, ALBTRN, SOLVE1, SURFAC, SETMTX, SOLVE0, FLUXES
c --------------------------------------------------------------------

c     .. Scalar Arguments ..

      INTEGER   LENGTH
c     ..
c     .. Array Arguments ..

      REAL      A( LENGTH )
c     ..
c     .. Local Scalars ..

      INTEGER   L
c     ..


      DO 10 L = 1, LENGTH
         A( L ) = 0.0
   10 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      subroutine chapman(nlyr, umu0, r0, h_lyr, umu0p)
      INTEGER    nlyr
      REAL       r0, h_lyr(0:nlyr), h(0:nlyr), umu0
      REAL       umu0p(nlyr, nlyr)

c     .. local variable .. 
      INTEGER    lc, n
      REAL       cos_theta0, sin_theta0, cos_theta, sin_theta
      REAL       s1, s2


      cos_theta0 = umu0
      sin_theta0 = sqrt(1. - umu0*umu0)
      h = h_lyr + r0

      do 30 lc = 1, nlyr
        sin_theta = sin_theta0 * h(lc) / h(0)
        cos_theta = sqrt( 1. - sin_theta * sin_theta )
        s1        = 0.
        do 40 n = 1, lc
          s2 = h(0)*cos_theta 
     &       - sqrt( h(n)*h(n) - h(0)*h(0)*sin_theta*sin_theta )
          !consider special case for delta h = 0
          if (s1 .ne. s2 .and. h(n-1) .ne. h(n) ) then
            umu0p(lc,n) = ( h(n-1) - h(n) ) / ( s2 - s1 )
          else
            if ( n .gt. 1 ) then
              umu0p(lc,n) = umu0p(lc,n-1)
            else 
              umu0p(lc,n) = umu0
            end if
          endif
!          print*, acos(umu0)*90.0/asin(1.), lc,n,
!     &            acos(umu0p(lc,n))*90./asin(1.), s2-s1
          s1 = s2
   40   continue


        
   30 continue


      end









c ******************************************************************
c ********** end of DISORT service routines ************************
c ******************************************************************

c ******************************************************************
c ********** IBCND=1 special case routines *************************
c ******************************************************************

c ---------------------------------------------------------------------
      SUBROUTINE ALBTRN( ALBEDO, AMB, APB, ARRAY, B, BDR, CBAND, CC,
     &                   CMU, CWT, DTAUCP, EVAL, EVECC, GL, GC, GU,
     &                   IPVT, KK, LL, NLYR, NN, NSTR, NUMU, PRNT,
     &                   TAUCPR, UMU, U0U, WK, YLMC, YLMU, Z, AAD,
     &                   EVALD, EVECCD, WKD, MAXUMU,
     &                   MXCMU, MXUMU, SQT, ALBMED, TRNMED )

c    DISORT special case to get only albedo and transmissivity
c    of entire medium as a function of incident beam angle
c    (many simplifications because boundary condition is just
c    isotropic illumination, there are no thermal sources, and
c    particular solutions do not need to be computed).  See
c    Ref. S2 and references therein for details.
c
c    The basic idea is as follows.  The reciprocity principle leads to
c    the following relationships for a plane-parallel, vertically
c    inhomogeneous medium lacking thermal (or other internal) sources:
c
c       albedo(theta) = u_0(theta) for unit-intensity isotropic
c                       illumination at *top* boundary
c
c       trans(theta) =  u_0(theta) for unit-intensity isotropic
c                       illumination at *bottom* boundary
c
c    where
c
c       albedo(theta) = albedo for beam incidence at angle theta
c       trans(theta) = transmissivity for beam incidence at angle theta
c       u_0(theta) = upward azim-avg intensity at top boundary
c                    at angle theta
c
c
c    O U T P U T   V A R I A B L E S:
c
c       ALBMED(IU)   Albedo of the medium as a function of incident
c                    beam angle cosine UMU(IU)
c
c       TRNMED(IU)   Transmissivity of the medium as a function of
c                    incident beam angle cosine UMU(IU)
c
c
c    I N T E R N A L   V A R I A B L E S:
c
c       NCD         number of diagonals below/above main diagonal
c
c       RCOND       estimate of the reciprocal condition of matrix
c                   CBAND; for system  CBAND*X = B, relative
c                   perturbations in CBAND and B of size epsilon may
c                   cause relative perturbations in X of size
c                   epsilon/RCOND.  If RCOND is so small that
c                          1.0 + RCOND .EQ. 1.0
c                   is true, then CBAND may be singular to working
c                   precision.
c
c       CBAND       Left-hand side matrix of linear system Eq. SC(5),
c                   scaled by Eq. SC(12); in banded form required
c                   by LINPACK solution routines
c
c       NCOL        number of columns in CBAND matrix
c
c       IPVT        INTEGER vector of pivot indices
c
c       (most others documented in DISORT)
c
c   Called by- DISORT
c   Calls- LEPOLY, ZEROIT, SGBCO, SOLEIG, TERPEV, SETMTX, SOLVE1,
c          ALTRIN, SPALTR, PRALTR
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAXUMU, MXCMU, MXUMU, NLYR, NN, 
     &          NSTR, NUMU
      REAL      ALBEDO
c     ..
c     .. Array Arguments ..

      LOGICAL   PRNT( * )
      INTEGER   IPVT( * )
      REAL      ALBMED( MAXUMU ), AMB( NN, NN ), APB( NN, NN ),
     &          ARRAY( NSTR, NSTR ), B( NSTR*NLYR ), BDR( NN, 0:NN ),
     &          CBAND(9*NN-2,NLYR*NSTR ), CC(NSTR, NSTR ),
     &          CMU( MXCMU ), CWT( MXCMU ), DTAUCP( * ), EVAL( NN ),
     &          EVECC(NSTR,NSTR ), GC( MXCMU, MXCMU, * ),
     &          GL( 0:NSTR, * ), GU( MXUMU, MXCMU, * ), KK( MXCMU, * ),
     &          LL( MXCMU, * ), SQT( * ), TAUCPR( 0:* ),
     &          TRNMED( MAXUMU ), U0U( MXUMU, * ), UMU( MAXUMU ),
     &          WK( MXCMU ), YLMC( 0:MXCMU, MXCMU ), YLMU( 0:MXCMU, * ),
     &          Z(NSTR*NLYR )

      DOUBLE PRECISION AAD( NN, NN ), EVALD( NN ), EVECCD( NN, NN ),
     &                 WKD( MXCMU )
c     ..
c     .. Local Scalars ..

      LOGICAL   LAMBER, LYRCUT
      INTEGER   IQ, IU, L, LC, MAZIM, NCD, NCOL, NCUT
      REAL      DELM0, FISOT, RCOND, SGN, SPHALB, SPHTRN
c     ..
c     .. External Subroutines ..

      EXTERNAL  ALTRIN, ERRMSG, LEPOLY, PRALTR, SETMTX, SGBCO, SOLEIG,
     &          SOLVE1, SPALTR, TERPEV, ZEROIT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..

      MAZIM  = 0
      DELM0  = 1.0
c                    ** Set DISORT variables that are ignored in this
c                    ** special case but are needed below in argument
c                    ** lists of subroutines shared with general case
      NCUT   = NLYR
      LYRCUT = .FALSE.
      FISOT  = 1.0
      LAMBER = .TRUE.
c                          ** Get Legendre polynomials for computational
c                          ** and user polar angle cosines

      CALL LEPOLY( NUMU, MAZIM, MXCMU, NSTR-1, UMU, SQT, YLMU )

      CALL LEPOLY( NN, MAZIM, MXCMU, NSTR-1, CMU, SQT, YLMC )

c                       ** Evaluate Legendre polynomials with negative
c                       ** arguments from those with positive arguments;
c                       ** Dave/Armstrong Eq. (15), STWL(59)
      SGN  = -1.0

      DO 20 L = MAZIM, NSTR - 1

         SGN  = -SGN

         DO 10 IQ = NN + 1, NSTR
            YLMC( L, IQ ) = SGN*YLMC( L, IQ - NN )
   10    CONTINUE

   20 CONTINUE
c                                  ** Zero out bottom reflectivity
c                                  ** (ALBEDO is used only in analytic
c                                  ** formulae involving ALBEDO = 0
c                                  ** solutions; Eqs 16-17 of Ref S2)

      CALL ZEROIT( BDR, NN*( NN+1 ) )

c ===================  BEGIN LOOP ON COMPUTATIONAL LAYERS  =============

      DO 30 LC = 1, NLYR

c                                       ** Solve eigenfunction problem
c                                       ** in Eq. STWJ(8b), STWL(23f)

         CALL SOLEIG( AMB, APB, ARRAY, CMU, CWT, GL( 0,LC ), MAZIM,
     &                MXCMU, NN, NSTR, YLMC, CC, EVECC, EVAL,
     &                KK( 1,LC ), GC( 1,1,LC ), AAD, EVECCD, EVALD,
     &                WKD )

c                          ** Interpolate eigenvectors to user angles

         CALL TERPEV( CWT, EVECC, GL( 0,LC ), GU( 1,1,LC ), MAZIM,
     &                MXCMU, MXUMU, NN, NSTR, NUMU, WK, YLMC, YLMU )

   30 CONTINUE

c ===================  END LOOP ON COMPUTATIONAL LAYERS  ===============


c                      ** Set coefficient matrix (CBAND) of equations
c                      ** combining boundary and layer interface
c                      ** conditions (in band-storage mode required by
c                      ** LINPACK routines)

      CALL SETMTX( BDR, CBAND, CMU, CWT, DELM0, DTAUCP, GC, KK,
     &             LAMBER, LYRCUT, MXCMU, NCOL, NCUT,
     &             NLYR, NN, NSTR, TAUCPR, WK )

c                      ** LU-decompose the coeff. matrix (LINPACK)

      NCD  = 3*NN - 1
      CALL SGBCO( CBAND, 9*NN-2, NCOL, NCD, NCD, IPVT, RCOND, Z )
      IF( 1.0+RCOND .EQ. 1.0 )
     &    CALL ERRMSG('ALBTRN--SGBCO says matrix near singular',.FALSE.)

c                             ** First, illuminate from top; if only
c                             ** one layer, this will give us everything

c                             ** Solve for constants of integration in
c                             ** homogeneous solution

      CALL SOLVE1( B, CBAND, FISOT, 1, IPVT, LL, MXCMU,
     &             NCOL, NLYR, NN, NLYR, NSTR )

c                             ** Compute azimuthally-averaged intensity
c                             ** at user angles; gives albedo if multi-
c                             ** layer (Eq. 9 of Ref S2); gives both
c                             ** albedo and transmissivity if single
c                             ** layer (Eqs. 3-4 of Ref S2)

      CALL ALTRIN( GU, KK, LL, MXCMU, MXUMU, MAXUMU, NLYR, NN, NSTR,
     &             NUMU, TAUCPR, UMU, U0U, WK )

c                               ** Get beam-incidence albedos from
c                               ** reciprocity principle
      DO 40 IU = 1, NUMU / 2
         ALBMED( IU ) = U0U( IU + NUMU/2, 1 )
   40 CONTINUE


      IF( NLYR.EQ.1 ) THEN

         DO 50 IU = 1, NUMU / 2
c                               ** Get beam-incidence transmissivities
c                               ** from reciprocity principle (1 layer);
c                               ** flip them end over end to correspond
c                               ** to positive UMU instead of negative

            TRNMED( IU ) = U0U( NUMU/2 + 1 - IU, 2 )
     &                     + EXP( -TAUCPR( NLYR ) / UMU( IU + NUMU/2 ) )

   50    CONTINUE

      ELSE
c                             ** Second, illuminate from bottom
c                             ** (if multiple layers)

         CALL SOLVE1( B, CBAND, FISOT, 2, IPVT, LL, MXCMU,
     &                NCOL, NLYR, NN, NLYR, NSTR )

         CALL ALTRIN( GU, KK, LL, MXCMU, MXUMU, MAXUMU, NLYR, NN, NSTR,
     &                NUMU, TAUCPR, UMU, U0U, WK )

c                               ** Get beam-incidence transmissivities
c                               ** from reciprocity principle
         DO 60 IU = 1, NUMU / 2
            TRNMED( IU ) = U0U( IU + NUMU/2, 1 )
     &                     + EXP( -TAUCPR( NLYR ) / UMU( IU + NUMU/2 ) )
   60    CONTINUE

      END IF


      IF( ALBEDO.GT.0.0 ) THEN

c                             ** Get spherical albedo and transmissivity
         IF( NLYR.EQ.1 ) THEN

            CALL SPALTR( CMU, CWT, GC, KK, LL, MXCMU, NLYR,
     &                    NN, NSTR, TAUCPR, SPHALB, SPHTRN )
         ELSE

            CALL SPALTR( CMU, CWT, GC, KK, LL, MXCMU, NLYR,
     &                    NN, NSTR, TAUCPR, SPHTRN, SPHALB )
         END IF

c                                ** Ref. S2, Eqs. 16-17 (these eqs. have
c                                ** a simple physical interpretation
c                                ** like that of adding-doubling eqs.)
         DO 70 IU = 1, NUMU

            ALBMED(IU) = ALBMED(IU) + ( ALBEDO / (1.-ALBEDO*SPHALB) )
     &                                * SPHTRN * TRNMED(IU)

            TRNMED(IU) = TRNMED(IU) + ( ALBEDO / (1.-ALBEDO*SPHALB) )
     &                                * SPHALB * TRNMED(IU)
   70    CONTINUE

      END IF
c                          ** Return UMU to all positive values, to
c                          ** agree with ordering in ALBMED, TRNMED
      NUMU  = NUMU / 2
      DO 80 IU = 1, NUMU
         UMU( IU ) = UMU( IU + NUMU )
   80 CONTINUE

      IF( PRNT(4) ) CALL PRALTR( UMU, NUMU, ALBMED, TRNMED )


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE ALTRIN( GU, KK, LL, MXCMU, MXUMU, MAXUMU, NLYR, NN,
     &                   NSTR, NUMU, TAUCPR, UMU, U0U, WK )

c       Computes azimuthally-averaged intensity at top and bottom
c       of medium (related to albedo and transmission of medium by
c       reciprocity principles; see Ref S2).  User polar angles are
c       used as incident beam angles. (This is a very specialized
c       version of USRINT)
c
c       ** NOTE **  User input values of UMU (assumed positive) are
c                   temporarily in upper locations of  UMU  and
c                   corresponding negatives are in lower locations
c                   (this makes GU come out right).  I.e. the contents
c                   of the temporary UMU array are:
c
c                     -UMU(NUMU),..., -UMU(1), UMU(1),..., UMU(NUMU)
c
c
c   I N P U T    V A R I A B L E S:
c
c       GU     :  Eigenvectors interpolated to user polar angles
c                   (i.e., g in Eq. SC(1), STWL(31ab))
c
c       KK     :  Eigenvalues of coeff. matrix in Eq. SS(7), STWL(23b)
c
c       LL     :  Constants of integration in Eq. SC(1), obtained
c                   by solving scaled version of Eq. SC(5);
c                   exponential term of Eq. SC(12) not included
c
c       NN     :  Order of double-Gauss quadrature (NSTR/2)
c
c       TAUCPR :  Cumulative optical depth (delta-M-scaled)
c
c       (remainder are DISORT input variables)
c
c
c   O U T P U T    V A R I A B L E:
c
c       U0U  :    Diffuse azimuthally-averaged intensity at top and
c                 bottom of medium (directly transmitted component,
c                 corresponding to BNDINT in USRINT, is omitted).
c
c
c   I N T E R N A L    V A R I A B L E S:
c
c       DTAU   :  Optical depth of a computational layer
c       PALINT :  Non-boundary-forced intensity component
c       UTAUPR :  Optical depths of user output levels (delta-M scaled)
c       WK     :  Scratch vector for saving 'EXP' evaluations
c       All the exponential factors (i.e., EXP1, EXPN,... etc.)
c       come from the substitution of constants of integration in
c       Eq. SC(12) into Eqs. S1(8-9).  All have negative arguments.
c
c   Called by- ALBTRN
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   MAXUMU, MXCMU, MXUMU, NLYR, NN, NSTR, NUMU
c     ..
c     .. Array Arguments ..

      REAL      GU( MXUMU, MXCMU, * ), KK( MXCMU, * ), LL( MXCMU, * ),
     &          TAUCPR( 0:* ), U0U( MXUMU, * ), UMU( MAXUMU ),
     &          WK( MXCMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, IU, IUMAX, IUMIN, LC, LU
      REAL      DENOM, DTAU, EXP1, EXP2, EXPN, MU, PALINT, SGN
c     ..
c     .. Local Arrays ..

      REAL      UTAUPR( 2 )
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ABS, EXP
c     ..


      UTAUPR( 1 ) = 0.0
      UTAUPR( 2 ) = TAUCPR( NLYR )

      DO 50 LU = 1, 2

         IF( LU.EQ.1 ) THEN

            IUMIN  = NUMU / 2 + 1
            IUMAX  = NUMU
            SGN    = 1.0

         ELSE

            IUMIN  = 1
            IUMAX  = NUMU / 2
            SGN    = - 1.0

         END IF
c                                   ** Loop over polar angles at which
c                                   ** albedos/transmissivities desired
c                                   ** ( upward angles at top boundary,
c                                   ** downward angles at bottom )
         DO 40 IU = IUMIN, IUMAX

            MU   = UMU( IU )
c                                     ** Integrate from top to bottom
c                                     ** computational layer
            PALINT = 0.0

            DO 30 LC = 1, NLYR

               DTAU   = TAUCPR( LC ) - TAUCPR( LC - 1 )
               EXP1   = EXP( ( UTAUPR( LU ) - TAUCPR( LC - 1 ) ) / MU )
               EXP2   = EXP( ( UTAUPR( LU ) - TAUCPR( LC ) ) / MU )

c                                      ** KK is negative
               DO 10 IQ = 1, NN

                  WK( IQ ) = EXP( KK( IQ,LC )*DTAU )
                  DENOM  = 1.0 + MU*KK( IQ, LC )

                  IF( ABS( DENOM ).LT.0.0001 ) THEN
c                                                   ** L'Hospital limit
                     EXPN   = DTAU / MU*EXP2

                  ELSE

                     EXPN   = ( EXP1*WK( IQ ) - EXP2 )*SGN / DENOM

                  END IF

                  PALINT = PALINT + GU( IU, IQ, LC )*LL( IQ, LC )*EXPN

   10          CONTINUE

c                                        ** KK is positive
               DO 20 IQ = NN + 1, NSTR

                  DENOM  = 1.0 + MU*KK( IQ, LC )

                  IF( ABS( DENOM ).LT.0.0001 ) THEN

                     EXPN   = - DTAU / MU * EXP1

                  ELSE

                     EXPN = ( EXP1 - EXP2 * WK(NSTR+1-IQ) ) *SGN / DENOM

                  END IF

                  PALINT = PALINT + GU( IU, IQ, LC )*LL( IQ, LC )*EXPN

   20          CONTINUE

   30       CONTINUE

            U0U( IU, LU ) = PALINT

   40    CONTINUE

   50 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE PRALTR( UMU, NUMU, ALBMED, TRNMED )

c        Print planar albedo and transmissivity of medium
c        as a function of incident beam angle

c   Called by- ALBTRN
c --------------------------------------------------------------------

c     .. Parameters ..

      REAL      DPR
      PARAMETER ( DPR = 180.0 / 3.14159265 )
c     ..
c     .. Scalar Arguments ..

      INTEGER   NUMU
c     ..
c     .. Array Arguments ..

      REAL      ALBMED( NUMU ), TRNMED( NUMU ), UMU( NUMU )
c     ..
c     .. Local Scalars ..

      INTEGER   IU
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC ACOS
c     ..


      WRITE( *, '(///,A,//,A)' )
     &   ' *******  Flux Albedo and/or Transmissivity of ' //
     &   'entire medium  ********',
     &  ' Beam Zen Ang   cos(Beam Zen Ang)      Albedo   Transmissivity'

      DO 10 IU = 1, NUMU
         WRITE( *, '(0P,F13.4,F20.6,F12.5,1P,E17.4)' )
     &      DPR*ACOS( UMU( IU ) ), UMU( IU ), ALBMED( IU ), TRNMED( IU )
   10 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SOLVE1( B, CBAND, FISOT, IHOM, IPVT, LL, MXCMU,
     &                   NCOL, NCUT, NN, NLYR, NSTR )

c        Construct right-hand side vector B for isotropic incidence
c        (only) on either top or bottom boundary and solve system
c        of equations obtained from the boundary conditions and the
c        continuity-of-intensity-at-layer-interface equations
c
c
c     I N P U T      V A R I A B L E S:
c
c       CBAND    :  Left-hand side matrix of banded linear system
c                   Eq. SC(5), scaled by Eq. SC(12); assumed already
c                   in LU-decomposed form, ready for LINPACK solver
c
c       IHOM     :  Direction of illumination flag (1, top; 2, bottom)
c
c       NCOL     :  Number of columns in CBAND
c
c       NN       :  Order of double-Gauss quadrature (NSTR/2)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T     V A R I A B L E S:
c
c       B        :  Right-hand side vector of Eq. SC(5) going into
c                   SGBSL; returns as solution vector of Eq.
c                   SC(12), constants of integration without
c                   exponential term
c
c       LL      :   permanent storage for B, but re-ordered
c
c
c    I N T E R N A L    V A R I A B L E S:
c
c       IPVT     :  INTEGER vector of pivot indices
c       NCD      :  Number of diagonals below or above main diagonal
c
c   Called by- ALBTRN
c   Calls- ZEROIT, SGBSL
c +-------------------------------------------------------------------+

c     .. Scalar Arguments ..

      INTEGER   IHOM, MXCMU, NCOL, NCUT, NN, NLYR, NSTR
      REAL      FISOT
c     ..
c     .. Array Arguments ..

      INTEGER   IPVT(NSTR*NLYR )
      REAL      B(NSTR*NLYR),CBAND(9*NN-2, NLYR*NSTR), LL( MXCMU, * )
c     ..
c     .. Local Scalars ..

      INTEGER   I, IPNT, IQ, LC, NCD
c     ..
c     .. External Subroutines ..

      EXTERNAL  SGBSL, ZEROIT
c     ..


      CALL ZEROIT( B, NSTR*NLYR  )

      IF( IHOM.EQ.1 ) THEN
c                             ** Because there are no beam or emission
c                             ** sources, remainder of B array is zero
         DO 10 I = 1, NN
            B( I )             = FISOT
            B( NCOL - NN + I ) = 0.0
   10    CONTINUE

      ELSE IF( IHOM.EQ.2 ) THEN

         DO 20 I = 1, NN
            B( I )             = 0.0
            B( NCOL - NN + I ) = FISOT
   20    CONTINUE

      END IF


      NCD  = 3*NN - 1
      CALL SGBSL( CBAND, 9*NN-2, NCOL, NCD, NCD, IPVT, B, 0 )

      DO 40 LC = 1, NCUT

         IPNT  = LC*NSTR - NN

         DO 30 IQ = 1, NN
            LL( NN + 1 - IQ, LC ) = B( IPNT + 1 - IQ )
            LL( IQ + NN,     LC ) = B( IQ + IPNT )
   30    CONTINUE

   40 CONTINUE


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ---------------------------------------------------------------------
      SUBROUTINE SPALTR( CMU, CWT, GC, KK, LL, MXCMU, NLYR, NN, NSTR,
     &                   TAUCPR, SFLUP, SFLDN )

c       Calculates spherical albedo and transmissivity for the entire
c       medium from the m=0 intensity components
c       (this is a very specialized version of FLUXES)
c
c
c    I N P U T    V A R I A B L E S:
c
c       CMU,CWT    Abscissae, weights for Gauss quadrature
c                  over angle cosine
c
c       KK      :  Eigenvalues of coeff. matrix in eq. SS(7)
c
c       GC      :  Eigenvectors at polar quadrature angles, SC(1)
c
c       LL      :  Constants of integration in eq. SC(1), obtained
c                  by solving scaled version of Eq. SC(5);
c                  exponential term of Eq. SC(12) not included
c
c       NN      :  Order of double-Gauss quadrature (NSTR/2)
c
c       (remainder are DISORT input variables)
c
c
c    O U T P U T   V A R I A B L E S:
c
c       SFLUP   :  Up-flux at top (equivalent to spherical albedo due to
c                  reciprocity).  For illumination from below it gives
c                  spherical transmissivity
c
c       SFLDN   :  Down-flux at bottom (for single layer, equivalent to
c                  spherical transmissivity due to reciprocity)
c
c
c    I N T E R N A L   V A R I A B L E S:
c
c       ZINT    :  Intensity of m=0 case, in Eq. SC(1)
c
c   Called by- ALBTRN
c +--------------------------------------------------------------------

c     .. Scalar Arguments ..

      INTEGER   MXCMU, NLYR, NN, NSTR
      REAL      SFLDN, SFLUP
c     ..
c     .. Array Arguments ..

      REAL      CMU( MXCMU ), CWT( MXCMU ), GC( MXCMU, MXCMU, * ),
     &          KK( MXCMU, * ), LL( MXCMU, * ), TAUCPR( 0:* )
c     ..
c     .. Local Scalars ..

      INTEGER   IQ, JQ
      REAL      ZINT
c     ..
c     .. Intrinsic Functions ..

      INTRINSIC EXP
c     ..


      SFLUP  = 0.0

      DO 30 IQ = NN + 1, NSTR

         ZINT   = 0.0
         DO 10 JQ = 1, NN
            ZINT  = ZINT + GC( IQ, JQ, 1 )*LL( JQ, 1 )*
     &                     EXP( KK( JQ,1 )*TAUCPR( 1 ) )
   10    CONTINUE

         DO 20 JQ = NN + 1, NSTR
            ZINT  = ZINT + GC( IQ, JQ, 1 )*LL( JQ, 1 )
   20    CONTINUE

         SFLUP  = SFLUP + CWT( IQ - NN )*CMU( IQ - NN )*ZINT

   30 CONTINUE


      SFLDN  = 0.0

      DO 60 IQ = 1, NN

         ZINT   = 0.0
         DO 40 JQ = 1, NN
            ZINT  = ZINT + GC( IQ, JQ, NLYR )*LL( JQ, NLYR )
   40    CONTINUE

         DO 50 JQ = NN + 1, NSTR
            ZINT  = ZINT + GC( IQ, JQ, NLYR )*LL( JQ, NLYR )*
     &                     EXP( - KK( JQ,NLYR ) *
     &                     ( TAUCPR( NLYR ) - TAUCPR( NLYR-1 ) ) )
   50    CONTINUE

         SFLDN  = SFLDN + CWT( NN + 1 - IQ )*CMU( NN + 1 - IQ )*ZINT

   60 CONTINUE

      SFLUP  = 2.0*SFLUP
      SFLDN  = 2.0*SFLDN


      RETURN
      END
c - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

c ******************************************************************
c ********** End of IBCND=1 special case routines ******************
c ******************************************************************
