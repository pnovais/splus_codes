!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! This subroutine is the Friends-of-Friends adapted to be
! converted in Python code (package f2py). 
! 
! Commands: f2py -c fof2py.f -m fof_fortran
! Load the package: import fof_fortran
! idx_group = fof_fortran.fof(x_DF[idx],y_DF[idx],z_DF[idx],l_linking,n_DF)
! OUTPUT: idx_group array has the indexes of groups/structures of each galaxy. 
!
!                        Author: Costa-Duarte, M.V. - 19/02/2015
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
subroutine fof(x,y,z,l_linking,idx_group_out,n_DF)
Implicit None
integer, intent(in)   :: n_DF,l_linking
real, intent(in)   :: x(n_DF),y(n_DF),z(n_DF)
real, intent(out)  :: idx_group_out(n_DF)
integer i,j,k,n_group,neff
real ii,idx_i,idx_j,idx_new,idx_group(n_DF),idx0(n_DF)
real dist 
!
ii = 0
n_group = 0 ! Total number of groups
idx_group = 0.  ! Indexes of groups
do i=1,n_DF-1,1
 if (mod(i,1000).eq.0)print*,i,n_DF-1
 do j=i+1,n_DF,1
  if (dist(x(i),y(i),z(i),x(j),y(j),z(j)).le.l_linking) then ! distance is closer than l_linking?
   if ((idx_group(i).eq.0).and.(idx_group(j).eq.0)) then   ! both do not have group yet
    n_group = n_group + 1
    idx_group(i) = float(n_group)
    idx_group(j) = float(n_group)
   else
    if ((idx_group(i).eq.0).and.(idx_group(j).gt.0)) then 
     ii = idx_group(j) ! j-th has group then i is part of the group
    endif
    if ((idx_group(i).gt.0).and.(idx_group(j).eq.0)) then 
     ii = idx_group(i) ! i-th has group then j is part of the group
    endif
    if ((idx_group(i).gt.0).and.(idx_group(j).gt.0)) then 
     ii = min(idx_group(i),idx_group(j)) ! both have groups, choose lower index
    endif
    !
    idx_i = idx_group(i)
    idx_j = idx_group(j)
    idx_group(i) = ii ! i-th and j-th belong to the same group now.
    idx_group(j) = ii
    ! changing group indexes after merging i-th and j-th group
    do k=1,n_DF,1
     if ((idx_i.gt.0).and.(idx_group(k).eq.idx_i)) then 
      idx_group(k) = ii
     endif
     if ((idx_j.gt.0).and.(idx_group(k).eq.idx_j)) then 
      idx_group(k) = ii
     endif
    enddo
    !
   endif
  endif
 enddo
enddo
!
! Reorganizing index array and calculating effective number of groups
!
neff = 0
do i=1,n_DF-1,1
 if (i.eq.1) then ! 1st group
  neff = neff + 1 
  idx0(neff)=idx_group(i)
  do j=1,n_DF,1 ! change all of them with neff
   if (idx_group(j).eq.idx_group(i)) then
    idx_group_out(j)=idx0(neff) 
   endif
  enddo
  if (idx_group(i).eq.0.)stop ! Problem!
 endif
 !
 if (idx_group(i).ne.idx_group(i+1)) then ! next element is different from current one
  idx_new = 0.
  do j=1,neff,1 ! check if it's already in new vector
   if (idx0(j).eq.idx_group(i+1)) then
    idx_new = 1.
   endif
  enddo
  if (idx_new.eq.0) then 
   neff = neff + 1
   idx0(neff) = idx_group(i+1)
   !print*,'new',i,neff,idx0(neff)
   do j=1,n_DF,1
    if (idx_group(j).eq.idx_group(i+1)) then 
     idx_group_out(j)=float(neff)
    endif
   enddo
  endif
 endif
enddo
!
return
end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function dist(x1,y1,z1,x2,y2,z2)
Implicit None
real x1,y1,z1,x2,y2,z2,dist
dist = sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
return
end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!subroutine completeness_purity(idx_CD11_DF,ra_CD11_DF,dec_CD11_DF,redshift_CD11_DF,&
!x_CD11_DF,y_CD11_DF,z_CD11_DF,da_kpc_CD11_DF,& ! CD11,spec
!idx_DF,ra_DF,dec_DF,redshift_DF,x_DF,y_DF,z_DF,da_kpc_DF,& ! NEW,phot
!id_gal,ra_gal,dec_gal,redshift_gal,x_gal,y_gal,z_gal,& ! galaxies
!redshift_min,redshift_max,n_redshift,n_CD11_DF,n_DF,ngal,&  ! etc
!completeness,purity,redshift_bin,photoz_err,sigma_smoothing,n_sigma)
!
!    completeness,purity,redshift_bins = pack_fortran.completeness_purity(\
!    idx_CD11_spec_DF,ra_CD11_spec_DF,dec_CD11_spec_DF,redshift_CD11_spec_DF,\
!    x_CD11_spec_DF,y_CD11_spec_DF,z_CD11_spec_DF,\
!    idx_DF,ra_DF,dec_DF,redshift_DF,x_DF,y_DF,z_DF,\
!    id_gal,ra_gal,dec_gal,redshift_gal,x_gal,y_gal,z_gal,\
!    redshift_min,redshift_max,n_redshift_bins,N_CD11_DF,N_DF,photoz_err,sigma_smoothing)
!
!
!Implicit none
!integer i,j,k,npoint,npoint_bin,n_recovered_bin,n_tot_bin
!integer, intent(in)   :: n_DF,n_CD11_DF,ngal,n_redshift
! indexes of structures
!integer, intent(in)   :: idx_CD11_DF(n_CD11_DF),idx_DF(n_DF),id_gal(ngal)
! ra,dec, redshift and cartesian coordinates - CD11,spec
!real, intent(in)      :: ra_CD11_DF(n_CD11_DF),dec_CD11_DF(n_CD11_DF),redshift_CD11_DF(n_CD11_DF)
!real, intent(in)      :: x_CD11_DF(n_CD11_DF),y_CD11_DF(n_CD11_DF),z_CD11_DF(n_CD11_DF),da_kpc_CD11_DF(n_CD11_DF)
! ra,dec, redshift and cartesian coordinates - NEW,phot
!real, intent(in)      :: ra_DF(n_DF),dec_DF(n_DF),redshift_DF(n_DF)
!real, intent(in)      :: x_DF(n_DF),y_DF(n_DF),z_DF(n_DF),da_kpc_DF(n_DF)
! ra,dec, redshift and cartesian coordinates - GALAXIES
!real, intent(in)      :: ra_gal(ngal),dec_gal(ngal),redshift_gal(ngal)
!real, intent(in)      :: x_gal(ngal),y_gal(ngal),z_gal(ngal)
! redshift range
!real, intent(in)      :: redshift_min,redshift_max,photoz_err,sigma_smoothing,n_sigma
!real, intent(out)     :: completeness(n_redshift),purity(n_redshift),redshift_bin(n_redshift)
!
!integer n_SC_recovered,flag_gal_CD11(n_CD11_DF),flag_gal(n_DF),flag_SC(n_DF),n_SC_CD11,n_SC
!real redshift_i(n_redshift),redshift_f(n_redshift)
!real delta_redshift,distance_angular
!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
! Define redshift ranges with same number of DF grid points
!print*,'Number of DF points in CD11,spec',n_CD11_DF
!print*,'Number of DF points in NEW,phot',n_DF
!print*,'Number of redshift bins',n_redshift
!print*,
! delta_redshift for iteraction
!delta_redshift = 1e-4
! Number of DF grid points per redshift bin
!npoint_bin = aint(float(n_CD11_DF)/float(n_redshift))
! Define initial redshift bin
!redshift_i(1)=redshift_min
!redshift_f(1)=redshift_i(1) + delta_redshift
!redshift_bin = 0.
!
!do i=1,n_redshift,1 ! each redshift bin
! do while (npoint.lt.npoint_bin) ! while npoint < npoint per bin
!  npoint=0
!  redshift_f(i) = redshift_f(i) + delta_redshift ! increase redshift range of the bin
!  do j=1,n_CD11_DF,1
!   if ((redshift_CD11_DF(j).gt.redshift_i(i)).and.&
!   (redshift_CD11_DF(j).lt.redshift_f(i))) npoint=npoint+1 ! counting...
!  enddo
! enddo
! redshift_bin(i) = 0.5*(redshift_i(i)+redshift_f(i))
! print*,'redshift range',i,redshift_i(i),redshift_f(i),redshift_bin(i),npoint
! redshift_i(i+1) = redshift_f(i)
! redshift_f(i+1) = redshift_i(i+1) + delta_redshift
!enddo
!
! Number of structures in CD11,spec space
!n_SC_CD11 = 0
!do i=1,n_CD11_DF,1
!  n_SC_CD11 = max(n_SC_CD11,idx_CD11_DF(i))
!enddo
!print*,'n_SC_CD11=',n_SC_CD11
!
! Number of structures in NEW,phot space
!n_SC = 0
!do i=1,n_DF,1
!  n_SC = max(n_SC,idx_DF(i))
!enddo
!print*,'n_SC=',n_SC
!
!n_SC_recovered = 0
!flag_gal_CD11 = 0
!flag_gal = 0
!flag_SC = 0
!do i=1,n_SC_CD11,1 ! each SC in CD11,spec space
 !
 ! Which galaxies are in this structure?
! flag_gal_CD11 = 0
! do j=1,n_CD11_DF,1
!  !
!  if (idx_CD11_DF(j).eq.i) then ! Same index = same structure 
!  !
!   do k=1,ngal,1 ! look for galaxies which belong to this DF grid point (comparison in kpc)
!    if (((distance_angular(ra_CD11_DF(j),ra_gal(k),dec_CD11_DF(j),dec_gal(k)) * da_kpc_CD11_DF(j)).le.sigma_smoothing * 1000.)&
!    .and.(redshift_gal(k).lt.redshift_CD11_DF(j) + n_sigma * photoz_err).and.&
!    (redshift_gal(k).gt.redshift_CD11_DF(j) - n_sigma * photoz_err)) then 
!      flag_gal_CD11(k) = 1
!    endif
!   enddo
!  !
!  endif ! Same index = same structure 
!  !
! enddo
! ! With these CD11 structure and galaxies, look for the structure which has the
! ! most number of galaxies in common in NEW,phot space
! do j=1,n_SC,1
!  flag_gal = 0
!  do k=1,n_DF,1
!   !
!   if (idx_DF(j).eq.i) then ! Same index = same structure 
!   !
!    do k=1,ngal,1 ! look for galaxies which belong to this DF grid point (comparison in kpc)
!     if (((distance_angular(ra_CD11_DF(j),ra_gal(k),dec_CD11_DF(j),dec_gal(k)) * da_kpc_CD11_DF(j)).le.sigma_smoothing * 1000.)&
!     .and.(redshift_gal(k).lt.redshift_CD11_DF(j) + n_sigma * photoz_err).and.&
!     (redshift_gal(k).gt.redshift_CD11_DF(j) - n_sigma * photoz_err)) then 
!       flag_gal_CD11(k) = 1
!     endif
!    enddo
   !
!   endif ! Same index = same structure 
   !
!  enddo
  !
! enddo
!enddo
!do i=1,n_redshift,1 ! each redshift bin
!enddo
!completeness = 0.
!purity = 0.
!redshift_bin = 0.
!return
!end
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function distance_angular(ra1,ra2,dec1,dec2)
Implicit none
real ra1,ra2,dec1,dec2,distance_angular,d2r,arcs2r
!
d2r=acos(-1.)/180. ! degree2radian
arcs2r = acos(-1.)/(180. * 3600.) !arcsec2radian
!
distance_angular = acos(sin(dec1 * d2r) * sin(dec2 * d2r) + &
cos(dec1 * d2r) * cos(dec2 * d2r) * cos(ra1 * d2r - ra2 * d2r)) / arcs2r
return
end
