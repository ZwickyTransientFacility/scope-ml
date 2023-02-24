module aovconst

! AovConst.f90 : global definitions for Aov*.f90
  implicit none

  integer,parameter :: SP=kind(1d0)  ! type of measurements
  integer,parameter :: CP=kind((1.0_SP,1.0_SP))  ! type of measurements
  integer,parameter :: TIME=kind(1d0)! type of time & frequency variables
  Real (SP), Parameter :: PI2 = &
      6.283185307179586476925286766559005768394_SP
  integer,parameter :: CTMIN=5                   !minimum bin occupancy
  real(SP),parameter:: MINVAR=4.*tiny(1.0_SP)    !minimum variance difference

end module aovconst
