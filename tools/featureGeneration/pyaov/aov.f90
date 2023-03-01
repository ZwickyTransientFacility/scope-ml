Module aov
!
! Fortran 95 routines for period anlysis to be called from python
! (C) Alex Schwarzenberg-Czerny 1998-2011, alex@camk.edu.pl

! for debugging one could compile/run them with:
! f2py -m aov -h aov.pyf aovconst.f90 aovsub.f90 aov.f90
! f2py -m aov -c aovconst.f90 aovsub.f90 aov.f90

  Use aovconst
  Use aovsub
  Implicit None

  Private
  Public test,aovmhw,powspw,aovtrw,aovw,fouw,fgrid,totals,covar,fitcor

Contains

Subroutine covar(t1,v1,e1,t2,v2,e2,nlag,eps,iscale,ifunct,n1,n2,&
    lav,lmi,lmx,cc,cmi,cmx,ilag)
! TO BE DONE:
! -more flexibility in choosing the lag grid
! Purpose:
!   Calculates cross-correlation function(CCF) of two time series sampled
!   unevenly in time. Note that for good result each hump in the time series
!   should be covered by several observations (over-sampling).
!
  Integer,Intent(IN)   :: n1,n2,nlag
  Integer,Intent(IN)   :: iscale,ifunct
  Real(SP),Intent(IN)  :: eps,v1(n1),e1(n1),v2(n2),e2(n2)
  Real(TIME),Intent(IN):: t1(n1),t2(n2)
  Real(SP),Intent(OUT) :: cc(nlag),cmi(nlag),cmx(nlag), &
                          lav(nlag),lmi(nlag),lmx(nlag)
  Integer,Intent(OUT)  :: ilag
! f2py Integer,Intent(IN),optional   :: iscale=0,ifunct=0
! f2py Real(SP),Intent(IN),optional  :: eps=0.

!   Input:
!   n1      - number of observations in 1st set
!   n2      - number of observations in 2nd set
!   t1,v1,e1(n1)  - 1st set of observations: time, value, error
!   t2,v2,e2(n2)  - 2nd set of observations: time, value, error
!   iscale  - (ignored)output scale: iscale/=1 -linear; iscale=1 -logarythmic
!   ifunct  - (ignored)output function: ifunct/=1 -correlation; ifunct=1 -structure
!   nlag    - size of lag arrays

!   Output:
!   lav,lmi,lmx(nlag)- average & range of lags and ...
!   cc,cmi,cmx(nlag)-       ... average & range of correlations
!   ilag    - actual number of computed lags

! Method/Reference:
! Alexander, T., 1997, Astronomical Time Series,
! Eds. D. Maoz, A. Sternberg, and E.M. Leibowitz,
! (Dordrecht: Kluwer),218, 163

!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 2011,  alex@camk.edu.pl

! n1*n2 could be large dimension
  Integer  :: ind(n1*n2),mdt,mct,idt,ict,i,i1,i2,nct, &
        ib1(2*n1*n2/nlag),ib2(2*n1*n2/nlag)
  Real(SP) :: dt(n1*n2),e,z,s,r,r2,ad1,ad2,sd,sw
  Real(SP),Allocatable :: db1(:),db2(:),wb1(:),wb2(:),wb(:),cb(:),lags(:)

  mdt=iscale+ifunct ! use dummy variables to prevent warnings
  mdt=n1*n2
  nct=mdt/nlag
  mct=2*nct
  lav=0._SP;lmi=0._SP;lmx=0._SP;cc=0._SP;cmi=0._SP;cmx=0._SP

  Forall(i1=1:n1,i2=1:n2) dt((i1-1)*n2+i2)=t2(i2)-t1(i1)
  Call sortx(dt,ind)

  idt=1; ilag=1
  Do ! increment bins
    ict=0
    If(idt>mdt.or.ilag>nlag) Exit

    Do ! increment pairs
      If(idt>mdt) Exit
! check If lag bin finished
      If(ict>nct .And. dt(ind(idt))-dt(ind(max(1,idt-1)))>eps &
          .And.mdt-idt>nct/2) Exit
      i1=(ind(idt)-1)/n2+1
      i2=ind(idt)-(i1-1)*n2
      e=e1(i1)*e2(i2)
      If (e>0._SP) then
        Do i=1,ict          ! check dependence
          if(ib1(i)==i1 .Or. ib2(i)==i2) Exit
        End Do
        If (i<=ict) Then    ! dependent
          If(e<e1(ib1(i))*e2(ib2(i))) Then
            ib1(i)=i1
            ib2(i)=i2
          End If
        Else                ! independent
          If(ict>=mct) Exit ! this should never happen
          ict=ict+1
          ib1(ict)=i1
          ib2(ict)=i2
        End If
      End If
      idt=idt+1
    End Do

! evaluate correlation within a bin
    If (ict<2) Cycle
    Allocate(db1(ict),db2(ict),wb1(ict),wb2(ict),wb(ict),cb(ict),lags(ict))
    db1=v1(ib1(:ict));wb1=0._SP
    where (e1(ib1(:ict))>0._SP) wb1=1._SP/e1(ib1(:ict))**2
    db2=v2(ib2(:ict));wb2=0._SP
    where (e2(ib2(:ict))>0._SP) wb2=1._SP/e2(ib2(:ict))**2
    wb=sqrt(wb1*wb2)
    sw=sum(wb)

    ad1=sum(db1*wb1)/sum(wb1);db1=db1-ad1 ! data variance and averages
    ad2=sum(db2*wb2)/sum(wb2);db2=db2-ad2
    sd=sum(db1**2*wb1)*sum(db2**2*wb2)
    r=sum(db1*db2*wb)

    lags=t2(ib2(:ict))-t1(ib1(:ict))   ! average & extreme intervals
    lav(ilag)=sum(lags*wb)/sw
    lmx(ilag)=maxval(lags)
    lmi(ilag)=minval(lags)
    Deallocate(db1,db2,wb1,wb2,wb,cb,lags)

! correlation and its variance in z-variable
    If (sd<=0._SP) Cycle
    r=r/sqrt(sd)
    cc(ilag)=r
    r2=r*r
    z=(((((r2*3._SP+2._SP)*r2+11._SP)*0.5_SP/(ict-1)+ &
         r2+5._SP)*0.25/(ict-1)+1._SP) *r/(ict-1)+ &
         log(max(1._SP-r,tiny(z))/(1._SP+r)))*0.5_SP
    s=sqrt(((((-r2*3._SP-6._SP)*r2+22._SP)/3._SP/(ict-1)+ &
        4._SP-r2)*0.5/(ict-1)+1._SP)/(ict-1))
    cmi(ilag)=r-abs(tanh(z-s)-r)
    cmx(ilag)=r+abs(tanh(z+s)+r)

    ilag=ilag+1
  End Do
  ilag=ilag-1
End Subroutine covar


Subroutine fitcor(cn,co,npar1,no,cp,chi2)

  Integer,Intent(IN)     :: no,npar1
  Real(SP),Intent(IN)    :: cn(npar1,no)
  Real(SP),Intent(IN)    :: co(no+1,no+1)
  Real(SP),Intent(OUT)   :: cp(npar1,npar1)
  Real(SP),Intent(OUT)   :: chi2

!  Solves linear least sqares problem with weighting by a given full
!  covariance matrix of obserations. This routine applies to observations
!  mutually correlated, with covariance matrix known.

!  Input:
!  co(no,no)         - covariance matrix of observations (to be destroyed)
!  cn(npar1,no)      - condition equations
!  no                - number of observations
!  npar1             - number of parameters+1 (1 for calculation of chi2 only)

!  Output:
!  chi2              - chi2
!  cp(npar1,npar1)   - fitted parameters and their covariance matrix
!                        (See CRACOW for details)

!  Method:
!  First finds Cholesky triangle root Q of covariance matrix C by a call of CRACOW.
!  Next multiply Q by condition equations. Square the result to get the normal
!  equations. Solve them using  CRACOW and return values of the parameters
!  and their covariance matrix.

!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1992-2011,  alex@camk.edu.pl
  Integer  :: i,j
  Real(SP) :: cn1(npar1,no)
  Real(SP) :: co1(no+1,no+1)

  chi2=0._SP
!  Find inverse triangle root Q of the observations covariance matrix
  co1(:no,:no)=co(:no,:no)
  If (cracow(co1,0)==0._SP) Then
    Print *,' Covariance matrix of observations is singular (empty?)'
    Return
  End If
! Multiply condition equations by Q
  Do i=no,1,-1
    Do j=1,npar1
      cn1(j,i)=dot_product(cn(j,1:i),co1(1:i,i))
    End Do
  End Do
! Square the result to get normal equations
  Do i=1,npar1
    Do j=i,npar1
      cp(j,i)=dot_product(cn1(i,1:no),cn1(j,1:no))
    End Do
  End Do
!  Fit parameters by LSQ, return them and their covariance matrix and chi2
  If (cracow(cp,no)==0._SP) Then
    Print *,' Normal equations are singular, modify trend functions'
    Return
  End If
  chi2=cp(npar1,npar1)*(no-npar1+1)
End Subroutine fitcor

Subroutine totals(x,n)
! evaluate general properties of x
! note: quantiles computed in a crude way
  Integer,Intent(IN)  :: n
  Real(SP),Intent(IN) :: x(n)
!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1998-2011,  alex@camk.edu.pl

  Integer, Parameter :: nb = 8
  Real (SP) :: fmax, fmin, fmean, fdev, fq25, fq75, fmed
  Integer   :: ind(n), signs
  Character (Len=10) :: tstr

  If (n > 0) Then
    signs = 1
    fmean = sum (x) / n
    fdev = 0._SP
    If (n > 1) Then
      signs = count (x(:n-1)*x(2:) <= 0) + 1
      fdev = Max (sum((x-fmean)**2)/(n-1), 0._SP)
    End If
    If (fdev > 0._SP) fdev = Sqrt (fdev)

! calculate approx. quantiles by interpolation of
! a roughly binned histogram
    Call sortx (x, ind)
    fmax = x (ind(n))
    fmin = x (ind(1))
    fmed =(x(ind((n+1)/2))+x(ind((n+2)/2))) * 0.5_SP
    fq25 = x (ind((n+3)/4))
    fq75 = x (ind(3*n/4+1))
!
    Print '(4(a,i8))', ' length=', n,' signs=', signs
    Print '(3(a,1pg16.6))', ' min=', fmin, ' max=', fmax,' mean=', fmean
    Print '(3(a,1pg16.6))', ' q25=', fq25, ' q75=', fq75,' median=', fmed
    Call date_and_time (time=tstr)! standard system routine
    Print '(1(a,1pg16.6),a)', ' std.dev=', fdev, ' time= '// &
          tstr (1:2) // ':' // tstr (3:4) // ':' // tstr (5:6)
  End If

End Subroutine totals

Subroutine fgrid(t,nobs,fstop,fstep,fr0)
  Integer,Intent(IN)    :: nobs
  Real(TIME),Intent(IN) :: t(nobs)
  Real(TIME),Intent(OUT):: fr0,fstep,fstop
!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1998-2011,  alex@camk.edu.pl

  Integer      :: ind(nobs),nobs1,iobs,nfr
  Real(SP)     :: del(nobs-1)
  Real(TIME)   :: steps

  Integer,parameter :: MAXNFR=huge(nfr)

  nobs1=nobs-1
  If (nobs1 < 5) Then
    Print *,'FGRID: Too few observations'
    Return
  Endif

  Call sortx(Real(t,Kind=SP),ind)
  fstep=t(ind(nobs))-t(ind(1))
  If (fstep <= 0._SP) Then
     Print *,'FGRID: Input TIME values are wrong'
     Return
  Endif
  fstep=0.3_SP/fstep


  del=t(ind(2:))-t(ind(:nobs1))
  Call sortx(del,ind(:nobs1))

  fstop=Log(Real(nobs,Kind=SP))
  iobs=Nint(nobs1*(1._SP/(6._SP+0.3_SP*fstop)+0.05_SP))+1
  fstop=del(ind(iobs))
  If (fstop <= 0._SP) Then
     Print *, 'FGRID: Too finely spaced observations: bin them coarsly'
     Return
  Endif

  fstop=0.5_SP/fstop*(del(ind(nobs1/2))/fstop)**(0.6_SP)
  steps=fstop/fstep
  If (steps >= MAXNFR) Then
    steps=MAXNFR
    Print *,'FGRID: *** DANGER *** Data span too long interval'// &
          ' for good sampling of periodogrammes.'
    Print *,'Splitting data into shorter intervals and taking'// &
          ' average of periodograms could help by reducing resolution.'
  Endif

  nfr=Nint(steps)
  fstep=fstop/steps
  fr0=0._SP
  Print *,' RESULTS OF FREQUENCY BAND EVALUATION:'
  Print '(3(A,1PE10.1))', 'Max. Freq.: ',fstop,'  Resolution: ',fstep, &
      '  Min. Freq.: ',fr0
  Print '(A,i10)', 'No. of points:  ',nfr
End Subroutine fgrid

Subroutine test(no,t,f,er)
!   Simulated light curve
  Integer,Intent(IN)     :: no
  Real(TIME),Intent(OUT) :: t(no)
  Real(SP),Intent(OUT)   :: f(no),er(no)

  Integer k,k1
  Real(SP) s2n,span,omt
  Real(TIME) fr,dph

!   simulate data
  s2n=100._SP
  span=40._SP
  fr=1._TIME/0.3617869_TIME
  Write(*,*) 'Simulated data, FR0=',fr
  Do k=1,no
    k1=k-1
    er(k)=0.5_SP
    t(k)=span*sin(k1*1._TIME)**2.
    t(1)=0._TIME
    t(no)=span
    dph=fr*t(k)
    omt=pi2*(dph-floor(dph))
    f(k)=11.90_SP+0.434_SP*cos(omt+4.72_SP)+ &
        0.237_SP*cos(2._SP*omt+0.741_SP)+sin(k1*1949._SP)/s2n
  End Do
End Subroutine test

Subroutine fouw(t,valin,er,frin,nh2,frout,valout,cof,dcof,nobs)
!
! FOUW - Fit Fourier series (cof) and Return vector of residuals
!          (fout), adjusting frequency (fr) by NLSQ
!
!  Purpose:
!   Twofold: (i) Find Fourier coefficients and improve frequency
!   (ii) Pre-whiten data with a given frequency

!  Method:
!   Fits Fourier series of nh2/2 harmonics with frequency adjustment
!   NLSQ by Newton-Raphson iterations:
!   fin(ph)=cof(1)+sum_{n=1}^{nh2/2}(cof(2n-1)Cos(ph*n)+cof(2n)*Sin(ph*n))
!   where ph=2pi*frout*(t-t0) and t0=(max(t)+min(t))/2
!  Input:
!   t(:),fin(:),er(:) -times, values and weights of observations;
!   nh2- number of model parameters: nh2/2-no. of harmonics
!   frin- initial value of frequency is abs(fr)
!       for frin<0 on Return frout=frin
!       for frin>0 value of frout is adjusted by NLSQ iterations
!  Output:
!   frout-final value of frequency (see frin)
!   cof(:),dcof(:)-Fourier coefficients and errors
!      cof(nh2+2)-approximate t0 (for full accuracy use t0=(max(t)+min(t))/2)
!      dcof(nh2+2)-frequency error (0 for fr<0)
!   fout(:)- residuals from fit/prewhitened data
!
!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1998-2011,  alex@camk.edu.pl
!  Please quote A.Schwarzenberg-Czerny, 1995, Astr. & Astroph. Suppl, 110,405 ;

  Integer,parameter     :: ITMAX=100
  Integer,Intent(IN)    :: nobs,nh2
  Real(SP),Intent(IN)   :: valin(nobs),er(nobs)
  Real(TIME),Intent(IN) :: t(nobs),frin
  Real(TIME),Intent(OUT):: frout
  Real(SP),Intent(OUT)  :: valout(nobs),cof(nh2/2*2+2),dcof(nh2/2*2+2)

  logical    :: finish
  Integer    :: l,n,nn2,nh,nx,nx1,it
  Real(SP),allocatable :: e(:),a(:,:)
  Real(SP)   :: cond,detnth,rhs
  Real(TIME) :: ph,t0,dfr

  nh=max(1,nh2/2)
  nn2=nh+nh


  If (nobs<nh+nh+2 .Or. Size(valin)/=nobs .Or. &
         Size(er)/=nobs .Or. Size(valout)/=nobs) Then
    Write(*,*) 'FOUW:error: wrong Size of arrays'
    Return
  End If

  t0=(maxval(t)+minval(t))*0.5_TIME
  dfr=1./(maxval(t)-minval(t))

  cof=0._SP
  dcof=0._SP
  frout=abs(frin)
  Print *,'  it    sigma   step    cond'
  Do it=1,ITMAX
    nx=nn2+2
    nx1=nx            ! adjust only coefficients
    if(it>1) nx1=nx+1 ! adjust also frequency
    allocate(e(nx1),a(nx1,nx1))

    Do l=1,nobs
      ph = frout*(t(l)-t0)
      ph = PI2 * (ph-floor(ph))*0.5_SP
      rhs=valin(l)-cof(1)
      e(1)=1._SP
      e(nx)=0._SP

      Do n = 2, nn2, 2
        e(n)=  cos(ph*n)
        e(n+1)=sin(ph*n)
        e(nx)=e(nx)+(-e(n+1)*cof(n)+e(n)*cof(n+1))*n
        rhs=rhs-e(n)*cof(n)-e(n+1)*cof(n+1)
      End Do

      e(nx)=e(nx)*PI2*0.5_SP*(t(l)-t0)
      e(nx1)=rhs
      valout(l)=rhs
      if(er(l)>0._SP) Then
        e=e/er(l)
        Call givensacc(e,a)
      End If
    End Do
    cond=givens(a,nobs-nx1+1,detnth)
    cof(nx)=0._SP
    cof(:nx1-1)=cof(:nx1-1)+a(nx1,:nx1-1)
    frout=frout+cof(nx)
    Do l=1,nx1-1
       dcof(l)=sqrt(max(a(l,l),tiny(a)))
    End Do
    finish=frin<0. .Or. abs(cof(nx))>dfr .Or. abs(cof(nx))<dcof(nx)*0.01

    If (nint(exp(nint(2.5*log(Real(it)))/2.5))==it .Or.finish) &
       Print '(i5,f10.5,2(1pe8.1))',it,sqrt(a(nx1,nx1)), &
       sqrt(sum(a(nx1,:nx1-1)**2)/(nx1-1)),cond
    if(finish) Exit
    deallocate(e,a)
  End Do
  cof(nx)=t0

  Print *,'frout=',frout
  Do l=1,nx1-1
    Print '(2(f10.5,a))',cof(l),' +/-',dcof(l)
  End Do
End Subroutine fouw

Subroutine aovmhw(t,f,er,frs,th,nfr,nh2,fr0,nobs,ncov,frmax)
!
!  Purpose:
!   Twofold: (i) Period search in an unevenly sampled time series.
!   Use coarse frequncy grid for broad band.
!   (ii) Finding exact value of period by fit of Fourier series.
!   Use fine frequency grid over narrow band.

!  Input:
!   t(:),f(:),er(:) -times, values and weights of observations;
!   nh2       - 2*number of Fourier harmonics
!   ncov      -ignored, for conformity only
!   (frs*(l-1)+fr0,l=1,nfr) - frequency grid of the periodogram

!  Note: take care in setting nfr and frs:      rough guess:
!   frs=1./(5.*DT),   nfr=(fru-frl)/frs+1
!   for observations spanning interval DT. Occasionaly nfr
!   becomes very large and code may compute very slowly,
!   split data and use coarse grids to acceterate
!
!  Output:
!   (th(l),l=1,nfr) - the AOV periodogram
!   The AOV periodogram has F(2*nh+1,nobs-2*nh-2) distribution
!
!  Advantages:
!   Particularly sensitive for detection of sharp signals
!   (eclipses, narrow pulses). Fast algorithm based on recurrence
!   formulae for orthogonal polynomials. Enables use of long Fourier
!   series with reasonable computing overhead (scales linearly with nh2).
!   For nh2=2 reduces to AOV (i.e. SignalPower/Residual Power)
!   transformation of the Lomb-Scargle Periodogram. In statistical sense
!   (sensitivity) it never performs worse than either Lomb-Scargle or Power Spectrum.
!   For non-sinusoidal signal and suitable nh>1 performes much better than the above
!   mentioned methods.

!  Method:
!   Calculates AOV periodogram by fitting data with orthogonal trigonometric
!   polynomials of nh2/2 harmonics, by fast recurrence projection procedure,
!   due to Schwarzenberg-Czerny, 1996. For nh2=2 or 3 it reduces to AOV version
!   of Ferraz-Mello (1991) variant of Lomb-Scargle periodogram, improved by
!   constant shift of data values. Advantage of the shift is vividly illustrated by
!   Foster (1995).

!   Please quote:
!        A.Schwarzenberg-Czerny, 1996, Astrophys. J.,460, L107.
!    Other references:
!	Foster, G., 1995, AJ v.109, p.1889 (his Fig.1).
!       Ferraz-Mello, S., 1981, AJ v.86, p.619.
!	Lomb, N. R., 1976, Ap&SS v.39, p.447.
!       Scargle, J. D., 1982, ApJ v.263, p.835.

!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1998-2005,  alex@camk.edu.pl
!  Its distribution is free, except that distribution of modifications is prohibited.


  Integer,Intent(IN)    :: nobs,nfr,ncov ! ncov-unused
  Real(SP),Intent(IN)   :: f(nobs),er(nobs)
  Real(TIME),Intent(IN) :: t(nobs),frs
  Real(TIME),Intent(IN) :: fr0
  Integer,Intent(IN)    :: nh2
  Real(SP),Intent(OUT)  :: th(nfr)
  Real(TIME),Intent(OUT):: frmax
!f2py  Real(TIME),Intent(IN),optional :: fr0=0.
!f2py  Integer,Intent(IN),optional    :: nh2=3
!f2py  Integer,Intent(IN),optional    :: ncov=2

  Integer    :: l,n,nn2,nh
  Real(SP)   :: rw(nobs),sn,avf,vrf,d1,d2,fm,dx
  Real(TIME) :: ph(nobs)

  complex(CP),dimension(nobs) :: cf,p,z,zn
  complex(CP)  :: al,sc

  nh=max(1,nh2/2)
  nn2=nh+nh

  If (nobs<nh+nh+2.Or.nfr<1.Or.Size(f)/=nobs.Or. &
         Size(er)/=nobs) Then
    Write(*,*) 'AOVMHW:error: wrong size of arrays'
    Return
  End If
  If ((maxval(t)-minval(t))*frs>0.3) Then
    Write(*,*) 'AOVMHW:warning: undersampling in frequency'
  End If
  l=ncov ! use spurious dummy argument

  d1=nn2
  d2=nobs-nn2-1

  Where(er>0._SP)
    rw=1._SP/er
  Elsewhere
    rw=0._SP
  End Where
  avf=sum(f*rw*rw)/sum(rw*rw)
  vrf=sum(((f-avf)*rw)**2)

  th=0.
  Do l=1,nfr
    ph = (frs * (l-1) + fr0)*t

!   (f,g)=sum(w*conjg(f)*g) -definition
    ph = PI2 * (ph-floor(ph))
    z = cmplx (Cos(ph), Sin(ph), Kind=CP)
    ph = ph * nh
    cf = Real((f-avf)* rw) * cmplx (Cos(ph), Sin(ph), Kind=CP)
    zn = 1._SP
    p = rw

    Do n = 0, nn2
      sn = Real(sum (abs(p)**2),Kind=SP)
      al = sum (rw*z*p)
! NOTE: dot_product(a,b)=sum(conjg(a)*b)
      sc = dot_product (p, cf)

!      sn = Max (sn, 10._SP*tiny(sn))
      sn = Max (sn, epsilon(sn))
      al = al / sn
      th(l) = th(l) + Abs (sc) ** 2/sn
      p = p * z - al * zn * conjg(p)
      zn = zn * z
    End Do

    th(l)=d2*th(l)/(d1*max(vrf-th(l),MINVAR))
  End Do
  Call peak(nfr,th,frmax,fm,dx)
  frmax=fr0+(frmax-1._TIME)*frs
  !Print '(a,1pe12.3,a,g23.12)','Peak of ',fm,' at frequency ',frmax
End Subroutine aovmhw

Subroutine powspw(t,f,er,frs,th,nfr, nh2,fr0,nobs,ncov,frmax)
!
!   AOVMHW - AOV Multiharmonic Periodogram for Uneven Sampling & Weights
!
!  Copyrights and Distribution:
!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1998-2005,  alex@camk.edu.pl
!  Its distribution is free, except that distribution of modifications is prohibited.

  Integer,Intent(IN)    :: nobs,nfr,nh2,ncov ! nh2,ncov-unused
  Real(SP),Intent(IN)   :: f(nobs),er(nobs)
  Real(TIME),Intent(IN) :: t(nobs),frs
  Real(TIME),Intent(IN) :: fr0
  Real(SP),Intent(OUT)  :: th(nfr)
  Real(TIME),Intent(OUT):: frmax
!f2py  Real(TIME),Intent(IN),optional :: fr0=0.
!f2py  Integer,Intent(IN),optional    :: nh2=3
!f2py  Integer,Intent(IN),optional    :: ncov=2

  Integer    :: l
  Real(SP)   :: w(nobs),avf,vrf,fm,dx,sw
  Real(TIME) :: ph(nobs)


  If (nobs<4.Or.nfr<1 .Or. Size(f)/=nobs .Or. &
         Size(er)/=nobs) Then
    Write(*,*) 'POWSP:error: wrong size of arrays'
    Return
  End If
  If ((maxval(t)-minval(t))*frs>0.3) Then
    Write(*,*) 'POWSP:warning: undersampling in frequency'
  End If
  l=ncov+nh2 ! use spurious dummy arguments

  Where(er>0._SP)
    w=1._SP/(er*er)
  Elsewhere
    w=0._SP
  End Where
  sw=sum(w)
  avf=sum(f*w)/sw
  vrf=sum((f-avf)**2*w)
  if(abs(vrf)<MINVAR) Then ! calculate window function
    avf=0.
    vrf=sw
  End If

  Do l=1,nfr
    ph = (frs * (l-1) + fr0)*t
    ph = PI2 * (ph-floor(ph))
    th(l) = abs(sum(Real((f-avf)*w)* &
              cmplx (Cos(ph), Sin(ph), Kind=CP)))**2
  End Do
  th=th/vrf/sw

  Call peak(nfr,th,frmax,fm,dx)
  frmax=fr0+(frmax-1._TIME)*frs
  Print '(a,1pe12.3,a,g23.12)','Peak of ',fm,' at frequency ',frmax
End Subroutine powspw

Subroutine  aovtrw(tin,fin,er,frs,th,nfr, nh2,fr0,nobs,ncov,frmax)

!AOVTR - AOV Multiharmonic Periodogram for planetary transits & Weights

!Purpose:
!AoV search for planetary transits, pulses and eclipses and other
!short duty cycle periodic phenomena, with constat signal elsewhere.

!Method:
!Implements Transit Analysis of Variance Periodogramme. It acts by
!fitting data with a top hat function deduced from phase folding and
!binning of data. Note: Transients are defined as MAXIMA (i.e. in
!magnitude units). For flux units (minima) reverse sign of fin.
!Employes phase folding and binning; usually bins are of fixed width
!in phase. For poor phase coverage, yielding less then CTMIN
!points in some bins, the routine switches to using flexible bin sizes;

!Advantage:
!Statistically near-optimal yet fast and well behaving for both
!small and large data; Robust against poor phase coverage;

!Input:
  Real(TIME),Intent(IN) :: tin(nobs)! times of observations
  Real(SP),Intent(IN)   :: fin(nobs)! values of observations
  Real(SP),Intent(IN)   :: er(nobs)! weights of observations
  Integer,Intent(IN)    :: nh2  ! number of phase bins
  Integer,Intent(IN)    :: ncov! number of phase coverages
  Real(TIME),Intent(IN) :: fr0 ! initial frequency
  Real(TIME),Intent(IN) :: frs ! frequency increment
! Output
  Real(SP),Intent(OUT)  :: th(nfr) ! periodogram
  Real(TIME),Intent(OUT):: frmax
!f2py  Real(TIME),Intent(IN),optional :: fr0=0.
!f2py  Integer,Intent(IN),optional    :: nh2=30
!f2py  Integer,Intent(IN),optional    :: ncov=2

!Copyrights and Distribution:
!This package is subject to copyrights by its author,
!(C) A. Schwarzenberg-Czerny 2003-2005,  alex@camk.edu.pl
!Its distribution is free, except that distribution of modifications is prohibited.
!Please quote A. Schwarzenberg-Czerny & J.-Ph. Beaulieu, 2006, MNRAS 365, 165

  Integer    :: ind(Size(tin)),i,ibin,ip,nbc,ifr,iflex,nobs,nobsw,nfr
  Real(SP)   :: f(Size(tin)),w(Size(tin)),ph(Size(tin)), &
     nct((nh2+1)*ncov),ave((nh2+1)*ncov),af,vf,sav,sw,ctw,fm,dx
  Real(TIME) :: t(Size(tin)), fr, at, dbc, dph

  nobsw=count(er>0.)

!  aovtrw=-1
  If (nobsw<=nh2+nh2 .Or. nobsw<CTMIN*nh2 .Or. Size(fin)/=nobs .Or. &
          Size(er)/=nobs) Then
    Write(*,*) 'AOVtrw: error: wrong Size of arrays'
    Return
  End If

  nbc = nh2 * ncov
  dbc = Real(nbc,Kind=TIME)
!  calculate totals and normalize variables
  iflex = 0
  w(1:nobsw)=1._SP/pack(er,er>0.)**2; sw=sum(w(1:nobsw))
  t(1:nobsw)=pack(tin,er>0.); f(1:nobsw)=pack(fin,er>0.)
  at=sum(w(1:nobsw)*t(1:nobsw))/sw; af=sum(w(1:nobsw)*f(1:nobsw))/sw
  t(1:nobsw)=t(1:nobsw)-at; f(1:nobsw)=f(1:nobsw)-af
  vf=sum(w(1:nobsw)*f(1:nobsw)*f(1:nobsw))
  f(1:nobsw)=w(1:nobsw)*f(1:nobsw)
  ctw=CTMIN*sw/nobsw
! assumed: sum(f)=0, sum(f*f)=vf and sum(t) is small
  Do ifr=1,nfr ! Loop over frequencies
    fr = Real(ifr-1,Kind=TIME) * frs + fr0
    Do ip = 0,1
      ave = 0.; nct = 0.
      If ( ip == 0) Then ! Try default fixed bins ...
        Do i = 1,nobsw ! MOST LABOR HERE
          dph=t(i)*fr ! Real(TIME) :: dph, t, fr
          sav=dph-floor(dph);ph(i)=sav
! note: mod accounts for rounding up due to double-->real conversion
          ibin=mod(int(sav*dbc),nbc)+1
          ave(ibin)=ave(ibin)+f(i)
          nct(ibin)=nct(ibin)+w(i)
                End Do
      Else !... elastic bins, if necesseary

        iflex=iflex+1     ! sort index ind using key ph
        Call sortx(ph(1:nobsw),ind(1:nobsw)); ! NRf77 indexx would do
        Do i=1,nobsw
          ibin=i*nbc/nobsw+1
          ave(ibin)=ave(ibin)+f(ind(i))
          nct(ibin)=nct(ibin)+w(ind(i))
        End Do
      End If
! counts: sub-bins=>bins
      nct(1+nbc:ncov+nbc)=nct(1:ncov)
      sav=0.
          Do i=ncov+nbc,1,-1
            sav=sav+nct(i)
                nct(i)=sav
          End Do
          Do i=1,nbc
            nct(i)=nct(i)-nct(i+ncov)
          End Do
          If (ip>0.Or.minval(nct(1:nbc))>=ctw) Exit
    End Do
! data: sub-bins=>bins */
    ave(1+nbc:ncov+nbc)=ave(1:ncov)
    sav=0.
        Do i=ncov+nbc,1,-1
          sav=sav+ave(i)
          ave(i)=sav
        End Do
        Do i=1,nbc
          ave(i)=ave(i)-ave(i+ncov)
        End Do

! AoV statistics for transits ...
        ave(1:nbc)=ave(1:nbc)/nct(1:nbc)
        ibin=sum(maxloc(ave(1:nbc)))
        sav=ave(ibin)
        sav=sav*sav*nct(ibin)*sw/(sw-nct(ibin))
    th(ifr) = sav/MAX(vf-sav,MINVAR)*(nobsw-2)
! ...where 'vf' keeps sum(f**2)
! the same for the classical AoV statistics:
!    sav=sum(ave(1:nbc)**2/nct(1:nbc))/ncov
!    th(ifr) = sav/(nh2-1)/MAX(vf-sav,MINVAR)*(nobsw-nh2)
  End Do

! if (iflex > 0) Write(*,*) 'AOVtrw:warning: poor phase coverage at ', &
!    iflex,' frequencies'
!  aovtrw=0
  Call peak(nfr,th,frmax,fm,dx)
  frmax=fr0+(frmax-1._TIME)*frs
  Print '(a,1pe12.3,a,g23.12)','Peak of ',fm,' at frequency ',frmax
End Subroutine aovtrw

Subroutine aovw(tin,fin,er,frs,th,nfr, nh2,fr0,nobs,ncov,frmax)

!AOV - Classical Analysis of Variance Periodogramme & Weights

!Purpose:
!AoV search for planetary transits, pulses and eclipses and other
!short duty cycle periodic phenomena, with constat signal elsewhere.

!Method:
!Employs phase folding and binning to calculate the classical
!Analysis of Variance (one way) periodogram. Re-implementation after TATRY.
!Usually bins are of fixed width in phase. For poor phase coverage,
!yielding less then CTMIN points in some bins, the routine switches
!to using flexible bin sizes;

!Advantage:
!Fastest general purpose periodogramme for uneven sampling, well behaving
!for both small and large data, robust against poor phase coverage;

!Input:
  Real(TIME),Intent(IN) :: tin(nobs)! times of observations
  Real(SP),Intent(IN)   :: fin(nobs)! values of observations
  Real(SP),Intent(IN)   :: er(nobs)! weights of observations
  Integer,Intent(IN)    :: nh2  ! number of phase bins
  Integer,Intent(IN)    :: ncov! number of phase coverages
  Real(TIME),Intent(IN) :: fr0 ! initial frequency
  Real(TIME),Intent(IN) :: frs ! frequency increment
! Output
  Real(SP),Intent(OUT)  :: th(nfr) ! periodogram
  Real(TIME),Intent(OUT):: frmax
!f2py  Real(TIME),Intent(IN),optional :: fr0=0.
!f2py  Integer,Intent(IN),optional    :: nh2=3
!f2py  Integer,Intent(IN),optional    :: ncov=2

!Copyrights and Distribution:
!This package is subject to copyrights by its author,
!(C) A. Schwarzenberg-Czerny 2003-2005,  alex@camk.edu.pl
!Its distribution is free, except that distribution of modifications is prohibited.
!Please quote A. Schwarzenberg-Czerny, 1989, M.N.R.A.S. 241, 153

  Integer   :: ind(Size(tin)),i,ibin,ip,nbc,nobs,nobsw,nfr
  Integer   :: ifr, iflex
  Real(SP)  :: f(Size(tin)),w(Size(tin)),ph(Size(tin)), &
     ncnt((nh2+1)*ncov),ave((nh2+1)*ncov),af, vf, sav, sw,ctw,fm,dx
  Real(TIME):: t(Size(tin)), fr, at, dbc, dph

!   Set variables (incl. mean and variance)
  nobsw=count(er>0._SP)

!  aovw=-1
  If (nobsw<=nh2+nh2 .Or. nobsw<CTMIN*nh2 .Or. Size(fin)/=nobs .Or. &
               Size(er)/=nobs) Then
    Write(*,*) 'AOVw: error: wrong size of arrays'
    Return
  End If
  nbc = nh2 * ncov
  dbc = Real(nbc,Kind=TIME)

!  calculate totals and normalize variables
  iflex = 0
  w(1:nobsw)=1._SP/pack(er,er>0.)**2;sw=sum(w(1:nobsw))
  t(1:nobsw)=pack(tin,er>0.);f(1:nobsw)=pack(fin,er>0.)
  at=sum(w(1:nobsw)*t(1:nobsw))/sw; af=sum(w(1:nobsw)*f(1:nobsw))/sw
  t(1:nobsw)=t(1:nobsw)-at; f(1:nobsw)=f(1:nobsw)-af
  vf=sum(w(1:nobsw)*f(1:nobsw)*f(1:nobsw))
  f(1:nobsw)=w(1:nobsw)*f(1:nobsw)
  ctw=CTMIN*sw/nobsw
! assumed: sum(w*f)=0, sum(w*f*f)=vf and sum(w*t) is small

!  Loop over all frequencies
  Do ifr = 1,nfr
        fr = Real(ifr-1,Kind=TIME) * frs + fr0
!  Up to two passes over all frequencies, depending on phase coverage
        Do ip = 0,1
      ave(1:nbc)=0.; ncnt(1:nbc) = 0.

          If ( ip == 0) Then !  Default fixed size phase bins ...
                Do i = 1,nobsw
                  dph=t(i)*fr; ! only dph, t0, fr, must keep TIME precision
                  sav=dph-floor(dph);ph(i)=sav
! note: mod accounts for rounding up due to double-->real conversion
                  ibin=mod(int(sav*dbc),nbc)+1
                  ave(ibin)=ave(ibin)+f(i)
                  ncnt(ibin)=ncnt(ibin)+w(i)
            End Do
          Else ! ...for poor phase coverage optional flexible bins
         ! sort index ind using key ph0, any index sort routine would Do
                iflex=iflex+1
                Call sortx(ph(1:nobsw),ind(1:nobsw))
                Do i = 1,nobsw
                  ibin=i*nbc/nobsw+1
                  ave(ibin)=ave(ibin)+f(ind(i))
                  ncnt(ibin)=ncnt(ibin)+w(ind(i))
            End Do
          End If

          ncnt(1+nbc:ncov+nbc)=ncnt(1:ncov)
          sav=0.
          Do i=ncov+nbc,1,-1
            sav=sav+ncnt(i)
                ncnt(i)=sav
          End Do
          Do i=1,nbc
            ncnt(i)=ncnt(i)-ncnt(i+ncov)
          End Do
          If (ip>0 .Or. minval(ncnt(1:nbc))>=ctw) Exit
    End Do

!    Calculate A.O.V. statistics for a given frequency
        ave(1+nbc:ncov+nbc)=ave(1:ncov)
    sav=0.
        Do i=ncov+nbc,1,-1
      sav=sav+ave(i)
      ave(i)=sav
        End Do
        Do i=1,nbc
          ave(i)=(ave(i)-ave(i+ncov))
        End Do

        sav=sum(ave(1:nbc)**2/ncnt(1:nbc))/ncov
        th(ifr) = sav/(nh2-1)/MAX(vf-sav,MINVAR)*(nobsw-nh2)
  End Do

! If (iflex > 0) Write(*,*) 'AOVw:warning: poor phase coverage at ',&
! iflex,' frequencies'
!  aovw=0
  Call peak(nfr,th,frmax,fm,dx)
  frmax=fr0+(frmax-1._TIME)*frs
  Print '(a,1pe12.3,a,g23.12)','Peak of ',fm,' at frequency ',frmax
End Subroutine aovw

End Module aov
