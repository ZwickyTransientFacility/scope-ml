Module aovsub
  use aovconst
  Implicit None

Interface sortx
  Module Procedure sortx_m
End Interface sortx

!Private
!Public sortx,sortx_m,sortx_h,givensacc,givens,peak

Contains

!  C R A C O W

!  Purpose:
!  To find least squares solution of normal equations.

!  Input:
!  A(n1,n1) - a symmetric matrix. If matrix A represents normal
!    equations, put the sum of squares of RHS into element A(n1,n1)
!    for calculation of the chi2 and covariance matrix. Only A(i,j)
!    for i>=j are requires for input.
!  N1       - size of matrix A (number of unknowns) +1
!  M        - M>=N1 - number of condition equations
!           - M< N1 - only triangle root and its triangle inverse are
!             returned

!  Output:
!  M>=N1:
!    A(i,n1) - unknowns (i=1,...,n)
!    A(i,i)  - their variances
!    A(n1,n1)- variance of an observation of unit weight (Chi2/D.F.)
!    A(i,j)  - for j>=i half of the covariance matrix of unknowns

!  M<N1:
!    A(i,j)  - for j>=i inverse triangle root matrix (its square is A**(-1))
!    A(i,j)  - for j<i contains square root matrix except of diagonal.
!              The diagonal contains arithmetic inverses of its elements.

!  Method:
!  Banachiewicz-Cholesky decomposition of the normal equations.
!  This subroutine is particularly suitable for well posed problems
!  with near-orthogonal base functions.
!    This method should not be used for near singular matrices,
!  since it sufferes from large numerical errors then. Use the singular
!  value decomposition method in such cases.

!  This package is subject to copyrights by its author,
!  (C) A. Schwarzenberg-Czerny 1987-2011,  alex@camk.edu.pl

Function cracow(a,m)
  Real(SP) :: cracow
  Integer,Intent(IN)     :: m
  Real(SP),Intent(INOUT) :: a(:,:)

  Integer  :: n1,i,j
  Real(SP) :: aii

!  Find Banachiewicz-Cholesky triangle root and its inverse Q, so that
!  Q*Q=A**(-1). The unknowns -a(i,n1) are a byproduct. So...
  n1=size(a,dim=1)
  cracow=0._SP
  Do i=1,n1-1
    If (a(i,i)<=0._SP) goto 99
    aii=sqrt(a(i,i))
!  ... complete previous i step, ...
    a(1:n1,i)=a(1:n1,i)/aii
!  ... complete next row of the triangle matrix, ...
    Do j=i+1,n1
      a(i,i+1)= a(i,i+1)-dot_product(a(j,1:i),a(i+1,1:i))
    End Do
!  ... compute next column of the inverse matrix.
    a(i,i)=1._SP/aii
    Do j=1,i
      a(j,i+1)=-dot_product(a(j,j:i),a(i+1,j:i))
    End Do
  End Do

!  Compute covariance matrix and reverse sign of unknowns
  If (m>=n1) Then
    a(n1,n1)=a(n1,n1)/(m-n1+1)
    Do i=1,n1-1
      a(i,n1)=-a(i,n1)
      Do j=1,i
        a(j,i)=dot_product(a(i,i:n1-1),a(j,i:n1-1))*a(n1,n1)
      End Do
    End Do
  End If
  cracow=1._SP
 99 return
End Function cracow


subroutine peak(n,fx,xm,fm,dx)
! Scan periodogram for a peak
! INPUT:
! n- periodogram length
! fx-periodogram values
! OUTPUT:
! xm-peak location
! fm-peak value
! dx-peak halfwidth
! SPECIAL CONDITIONS:
! fm<0 no peak in valid region
! dx<0 no valid (parabolic) peak
! METHOD:
! For f=log(fx) fits a parabola 0.5(d2*x**2+d1*x)+f(2)
! where d2=f(1)+f(3)-2f(2), d1=f(3)-f(1)
! finds dxl & dxp such, that the linear and quadratic terms drop
! by 0.7 in log, i.e. by a factor of 2 (dx=HWHI)
!(C) Alex Schwarzenberg-Czerny, 1999-2005 alex@camk.edu.pl

  integer,intent(in)     :: n
  real(SP),intent(in)    :: fx(n)
  real(TIME),intent(out) :: xm
  real(SP),intent(out)   :: fm
  real(SP),intent(out)   :: dx

  integer                :: nm,n0
  real(SP)               :: f(3),d2,d1

! valid maximum?
  nm=sum(maxloc(fx))
  xm=nm
  dx=-1.
  fm=fx(nm)
  if(fm<0.) return
! linear case (peak at edge)
  n0=max(2,min(n-1,nm))
  f=log(max(abs(fx(n0-1:n0+1))/fm,epsilon(1.)))
  d1=f(3)-f(1)
  dx=-abs(d1)/1.4

! parabolic case (not on edge)
  d2=-(f(1)-f(2)-f(2)+f(3))
  if (d2>0.7*dx*dx) then ! original 1.4
    fm=exp(0.125*d1*d1/d2+f(2))*fm
    xm=0.5*d1/d2+n0
    dx=sqrt(0.7*d2)
  endif
  dx=1./dx
end subroutine peak

subroutine givensacc(a,r)
! (C) Alex Schwarzenberg-Czerny,2000      alex@camk.edu.pl
  implicit none
  real(SP),intent(IN)    :: a(:)
  real(SP),intent(INOUT) :: r(:,:)
! Purpose: GIVENSACC prepares the least squares solution (LSQ) of an
! overdetermined linear system by Givens rotations, a state-of-art
! algorithm. Routine takes current observation equation from a and
! accumulates it into r:
! Input:
! a(n1)    - an equation with its RHS as a(n1) element,
!            multiplied by sqrt of weight.
!            Note: sum of all weights must equall no.observations,
!            lest GIVENS returns wrong covariance estimates.
! r(n1,n1) - triangle system R and its R.H.S
!            set r = 0 before te first call of GIVENSACC.
! Output:
! r(n1,n1) - updated system accounting of a

! tested OK
  integer       :: n1,n,i,j
  real(SP)      :: e(size(a)),p,q,s,rii,ei
  n1=size(a)
  n=n1-1
  e=a
  r(n1,n1)=r(n1,n1)+e(n1)*e(n1)
  do i=1,n
    rii=r(i,i)
    ei=e(i)
    s=rii*rii+ei*ei
    if (s<=0._SP ) then
      p=1._SP
      q=0._SP
    else
      s=sign(sqrt(s),rii)
      p=rii/s
      q=ei/s
    endif
    r(i,i)=s
    e(i)=0._SP
    do j=i+1,n1
      s=   q*e(j)+p*r(i,j)
      e(j)=p*e(j)-q*r(i,j)
      r(i,j)=s
    enddo
  enddo

end subroutine givensacc

real(SP) function givens(r,idf,detnth)
! (C) Alex Schwarzenberg-Czerny,2000      alex@camk.edu.pl
  implicit none
  integer,intent(IN)     :: idf
  real(SP),intent(INOUT) :: r(:,:)
  real(SP),intent(OUT)   :: detnth

! Purpose: GIVENS calculates the least squares solution of an
! overdetermined linear system by accurate Givens rotations.
! Prior to GIVENS call m-times GIVENSA in order to build the
! triangle system R
! Input:
! r(n1,n1) - triangle system R and its R.H.S
! idf - >0 solve normal equations for idf degrees of freedom
!       <0 and r(n1,n1)=1: inverse triangle matrix R
! Output:
! givens - condition ratio (min eigenv/max eigenv)
! detnth=Det(en)^(1/n) - n-th root of determinant
! For idf>0:
! r(n1,:)       - unknowns
! (r(k,k),k=1,n) variances
! (r(:k,k),k=1,n) half of covariance matrix
! (r(k+1:,k),k=2,n) half of inverse triangle root Q
! r(n1,n1) - chi^2/idf (=RHS variance=std.dev^2)
! For idf<0:
! ((r(j,k),j=1,k),k=1,n) inverse triangle root  Q
! (C) Alex Schwarzenberg-Czerny, Dec.2000,     alex@camk.edu.pl

! tested OK

  integer      ::n1,n,i,j
  real(SP),parameter :: tol=30._SP*epsilon(1._SP)
  real(SP)     ::rii,riimx,riimn

  n1=size(r,dim=1)
  n=n1-1
  r(n1,n1)=(r(n1,n1)-dot_product(r(1:n,n1),r(1:n,n1)))/max(1,idf)
  do i=1,n
    rii=0._SP
    if (abs(r(i,i))>tol) rii=1._SP/r(i,i)
    r(i,i)=-1._SP
    do j=1,i
      r(i,j)=r(i,j)*rii
      r(i+1,j)=-dot_product(r(j:i,j),r(j:i,i+1))
! en(j,i+1)=-dot_product(en(j,k:i),en(i+1,k:i))
    enddo
  enddo
! Already solved, square Q to get  inverse/covariance
  detnth=0.
  riimx=abs(r(1,1))
  riimn=riimx
  do i=1,n
    rii=abs(r(i,i))
    if (idf.gt.0) then
      do j=1,i
        r(j,i)=dot_product(r(i:n,i),r(i:n,j))*r(n1,n1)
      enddo
    endif
    if (rii>0._SP) detnth=detnth-log(rii)
    riimx=max(rii,riimx)
    riimn=min(rii,riimn)
  enddo
  givens=riimn/riimx
  detnth=exp(2._SP*detnth/n)

end function givens

subroutine sortx_h(dat,ind)
! (C) Alex Schwarzenberg-Czerny, 2001,2005 alex@camk.edu.pl
! Unstable index sorting. Method based on heap sort routine in
! Numerical Recipes, 1992, Press et al., Cambridge U.P.
  implicit none
  real(SP),intent(IN)    :: dat(:)
  integer,intent(OUT)    :: ind(:)

  integer                :: i,j,n
  n=size(dat)
  if (size(ind)<n) then
    write(*,*) 'sortx_h: wrong dimension'
    stop
  endif

  do i=1,n
    ind(i)=i
  enddo

  do i=n/2,1,-1
    call put(i,n)
  end do
  do i=n,2,-1
    j=ind(1)
    ind(1)=ind(i)
    ind(i)=j
    call put(1,i-1)
  end do

  contains

  subroutine put(low,up)
    integer,intent(IN) :: low,up

    integer            :: i,j,jsav
    real(SP)           :: sav
    jsav=ind(low)
    sav=dat(jsav)
    j=low
    i=low+low
    do
        if (i > up) exit
        if (i < up) then
          if (dat(ind(i)) < dat(ind(i+1))) i=i+1
        endif
        if (sav >= dat(ind(i))) exit
        ind(j)=ind(i)
        j=i
        i=i+i
    end do
    ind(j)=jsav

  end subroutine put

end subroutine sortx_h

subroutine sortx_m(dat,ina)
! (C) Alex Schwarzenberg-Czerny, 2005      alex@camk.edu.pl
! Stable index sorting by merge sort. Requires extra memory (inb)
  implicit none
  real(SP),intent(IN)    :: dat(:)
  integer,intent(OUT)    :: ina(:)

  integer                :: inb((size(dat)+1)/2),i,n

  n=size(dat)
  if (size(ina)<n) then
    write(*,*) 'sortx_m: wrong dimension'
    stop
  endif

  forall(i=1:n) ina(i)=i

  call sort_part(1, n);

  contains

  recursive subroutine sort_part(low,up)
    integer,intent(IN) :: low,up

    integer            :: med

    if (low<up) then
      med=(low+up)/2;
      call sort_part(low, med);
      call sort_part(med+1, up);
      call merge_part(low, med, up);
    endif
  end subroutine sort_part

  subroutine merge_part(low,med,up)
    integer,intent(IN) :: low,med,up

    integer            :: i,j,r

    inb(1:med-low+1)=ina(low:med)

    i=1; r=low; j=med+1
    do
      if (r>=j .or. j>up) exit
      if (dat(inb(i))<=dat(ina(j))) then
        ina(r)=inb(i)
        i=i+1
      else
        ina(r)=ina(j)
        j=j+1
      endif
      r=r+1
    enddo

    ina(r:j-1)=inb(i:i+j-1-r)
  end subroutine merge_part

end subroutine sortx_m

end MODULE aovsub
