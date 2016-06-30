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
subroutine fof(x,y,l_linking,idx_group_out,n_DF)
Implicit None
integer, intent(in)   :: n_DF,l_linking
real, intent(in)   :: x(n_DF),y(n_DF)
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
  if (dist(x(i),y(i),x(j),y(j)).le.l_linking) then ! distance is closer than l_linking?
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
function dist(x1,y1,x2,y2)
Implicit None
real x1,y1,x2,y2,dist
dist = sqrt((x1-x2)**2+(y1-y2)**2)
return
end

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
