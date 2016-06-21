!----------------------------------------------------------------------
!programa para analisar a distribuição espacial de populaçoes estelares
!dentro das galaxias observadas pelo J-PAS/J-PLUS/S-PLUS
!
! 12 de maio de 2016
!
!----------------------------Patricia Novais---------------------------

program Jplus
implicit none
integer, parameter:: nmax=31000000, npmax=2048,nmmax=251001
character(LEN=35)::b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,output,results
character(LEN=11)::nome
real,dimension(3,nmmax)::Jg,Jr,Ju,Ji,Jz,J1,J2,J3,J4,J5,J6,J7
real,dimension(14,nmmax)::F
real::fband(nmmax,12),countp(12,npmax,npmax),xaux(nmmax),yaux(nmmax),xcor(nmax)
real::ceu(12),ceum(12),sigceu(12),count1(npmax,npmax),countb(12,npmax,npmax)
real::corb(npmax,npmax),countf(npmax,npmax)
integer::count2(npmax,npmax),itipo(nmax),countc(4,npmax,npmax),count3(npmax,npmax)
integer::countf1(npmax,npmax),countf2(npmax,npmax),countf3(npmax,npmax),countf4(npmax,npmax)
real::xx,yy,fmin,fmax,alpha,alimiar,xxmin,xxinf,xxmax,xxmed,xxsup,percent,cc
integer::i,j,k,l,nc1,nc2,ii,jj,kb,nn,np,mm,bin,ixco,jyco
integer::x1,x2,y1,y2,npix,banda,xc1,xc2,yc1,yc2,npbx,npby,xr1,xr2,yr1,yr2


read(*,*) b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,output,results,nome,&
          & xc1,xc2,yc1,yc2,bin,xr1,xr2,yr1,yr2
!write(*,*)b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12

open(1,file=b1)
open(2,file=b2)
open(3,file=b3)
open(4,file=b4)
open(5,file=b5)
open(6,file=b6)
open(7,file=b7)
open(8,file=b8)
open(9,file=b9)
open(10,file=b10)
open(11,file=b11)
open(12,file=b12)
open(13,file=output)
open(14,file=results)


 read(1,*) Ju
 read(2,*) Jg
 read(3,*) Jr
 read(4,*) Ji
 read(5,*) Jz
 read(6,*) J1
 read(7,*) J2
 read(8,*) J3
 read(9,*) J4
 read(10,*) J5
 read(11,*) J6
 read(12,*) J7


do i=1,nmmax
	F(1,i)=Ju(1,i)
	F(2,i)=Ju(2,i)
	F(3,i)=Ju(3,i)
	F(4,i)=Jg(3,i)
	F(5,i)=Jr(3,i)
	F(6,i)=Ji(3,i)
	F(7,i)=Jz(3,i)
	F(8,i)=J1(3,i)
	F(9,i)=J2(3,i)
	F(10,i)=J3(3,i)
	F(11,i)=J4(3,i)
	F(12,i)=J5(3,i)
	F(13,i)=J6(3,i)
	F(14,i)=J7(3,i)
	write(13,*)F(1,i),F(2,i),F(3,i),F(4,i),F(5,i),F(6,i),F(7,i),F(8,i),F(9,i),&
                   & F(10,i),F(11,i),F(12,i),F(13,i),F(14,i)

enddo

 close(1)
 close(2)
 close(3)
 close(4)
 close(5)
 close(6)
 close(7)
 close(8)
 close(9)
 close(10)
 close(11)
 close(12)
 close(13)


! inicializacao de variaveis para estabelecer o tamanho da imagem, 
! numero de pixels e fluxos maximo e minimo na banda r
npix=0
x1=10000
x2=-10000
y1=10000
y2=-10000
fmin=1e30
fmax=-1e30




do i=1,nmmax
	npix=npix+1
	xx=F(1,i)
	yy=F(2,i)
	fband(i,1)=F(3,i)
	fband(i,2)=F(4,i)
	fband(i,3)=F(5,i)
	fband(i,4)=F(6,i)
	fband(i,5)=F(7,i)
	fband(i,6)=F(8,i)
	fband(i,7)=F(9,i)
	fband(i,8)=F(10,i)
	fband(i,9)=F(11,i)
	fband(i,10)=F(12,i)
	fband(i,11)=F(13,i)
	fband(i,12)=F(14,i)

	ii=xx
	jj=yy

	x1=min(x1,ii)
	x2=max(x2,ii)
	y1=min(y1,jj)
	y2=max(y2,jj)

!leitura das contagens em cada pixel, em cada banda
!countp = guarda as contagens nos pixeis, nas coord. x e y, na banda k.
!na imagem ainda nao binada

	do k=1,12
		countp(k,ii,jj)=fband(i,k)
	enddo

!calculo dos fluxos na banda r
	fmin=min(fmin,countp(3,ii,jj))
	fmax=max(fmax,countp(3,ii,jj))

enddo

write(14,*)'=========================================================================================='
write(14,*)'======================================S3P PROGRAM========================================='
write(14,*)'============================Stellar Populations Pixel by Pixel============================'
write(14,*)'=========================================================================================='
write(14,*)""
write(14,*)"------------------------------------------------------------------------------------------"
write(14,*)""
write(14,*)'==========================================GALÁXIA========================================='
write(14,*)'========================================',nome,'======================================='
write(14,*)'=========================================================================================='
write(14,*)'     #pixeis        imin        imax          jmin      jmax    fmin(r)            fmax(r)'
write(14,*) npix,x1,x2,y1,y2,fmin,fmax
write(14,*)""



!-------------------------------------------------------------------------
!-------------------------ESTATISTICAS DO CEU-----------------------------

!Dada uma area do ceu, calcularemos a media, media e sigma do ceu

!xc1,xc2,yc1,yc2 são as coordenadas iniciais e finais da regiao do ceu

!k contara o numero de pixeis da area do ceu


!banda = contador que identifica a banda que estamos usando
!ceu,sigceu e ceum sao a media, mediana e sigma do ceu

!os limites do ceu (xc1,xc2,yc1,yc2) já foram lidos no inicio do programa

write(14,*)""
write(14,*) "ESTATÍSTICAS DO CÉU"
write(14,*) "--------------------------------------------"

do banda=1,12
k=0
   do i=xc1,xc2
	do j=yc1,yc2
	   k=k+1
	   xaux(k)=countp(banda,i,j)
	enddo
   enddo

call avevar(xaux,k,ceu(banda),sigceu(banda))
call MDIAN1(xaux,k,ceum(banda))

write(14,*)'Band: ',banda,'Mean:', ceu(banda),'Median:',ceum(banda),'Sigma:',sigceu(banda)

enddo

write(14,*)""


!-------------------------------------------------------------------------
!------------------------SEGMENTACAO NA BANDA r---------------------------

!A segmentacao eh um passo muito importante, porque através disso 
!determinando o que é objeto e o que é fundo


!alpha = valor utilizado para o corte do ceu
!alimiar = limiar usado pra subtracao do ceu


write(14,*)""
write(14,*) "SEGMENTAÇÃO NA BANDA r"
write(14,*) "--------------------------------------------"


!calculo dos novos valores de pixeis em x e y, apos recorte
!onde as coordenadas do recorte sao dados por (x1:x2, y1:y2)
npbx=(xr2-xr1+1)/bin
npby=(yr2-yr1+1)/bin

write(14,*) 'tamanho do bin:', bin
write(14,*) 'numero de pixeis por bin:', bin*bin
write(14,*) 'numero de pixeis binados, em x e y:', npbx,npby
write(14,*) ''

!Zerar os vetores para que seja feita uma somatoria
do i=1,npbx
   do j=1,npby
	count1(i,j)=0.
	corb(i,j)=0.
	do k=1,12
	    countb(k,i,j)=0
	enddo
   enddo
enddo



!Subtraçao do ceu
do i=xr1,xr2
   ii=(i-xr1)/bin+1
	do j=yr1,yr2
	   jj=(j-yr1)/bin+1
		do k=1,12
			countb(k,ii,jj)=countb(k,ii,jj)+countp(k,i,j)-ceu(k)
		enddo
	count1(ii,jj)=count1(ii,jj)+countp(3,i,j)
	corb(ii,jj)=countb(2,ii,jj)/countb(3,ii,jj)
	enddo
enddo



!agora vamos calcular os valores minimos e maximos da contagem e cor

fmin=1.e10
fmax=-1.e30
do i=1,x2
   do j=1,y2
      fmin=min(fmin,count1(i,j))
      fmax=max(fmax,count1(i,j))
   enddo
enddo
write(14,*)'contagens minima e maxima na banda r, com céu subtraído'
write(14,*)fmin,fmax
write(14,*)""
      


fmin=1.e30
fmax=-1.e30
do i=1,npbx
   do j=1,npby
      fmin=min(fmin,corb(i,j))
      fmax=max(fmax,corb(i,j))
   enddo
enddo

write(14,*)'cor* g-r: minima		maxima '
write(14,*)	fmin,fmax
write(14,*)""
write(14,*)'**********************************************'
write(14,*) "a cor foi calculada, de forma simplista e apenas para conferir os resultados,"
write(14,*) "como a razao entre as bandas 1 e 2 (flux_banda1/flux_banda2), quanto menor "
write(14,*) 'a razao, mais vermelho o pixel'
write(14,*) "*********************************************"
write(14,*) ""



alpha=1.5
kb=3

alimiar=alpha*sigceu(kb)

np=0
nn=0
do i=1,npbx
   do j=1,npby
      count2(i,j)=0
      nn=nn+1
      itipo(nn)=0
      if(countb(kb,i,j).gt.alimiar)then
         np=np+1
         count2(i,j)=1
         itipo(nn)=1
      endif
   enddo
enddo

write(14,*) ""
write(14,*) 'Alpha usado na segmentacao (banda r):',alpha
write(14,*) 'Pixeis acima do limiar:',np
write(14,*) 'o valor do limiar:',alimiar
write(14,*) 'fracao de pixeis acima do limiar:',float(np)/float(npbx*npby)
write(14,*) ""


!========================================================================
!DESCOMENTAR AS LINHAS ABAIXO, PARA GERAR OS MAPAS BINARIOS DA
!IMAGEM SEGMENTADA
!========================================================================
!write(14,*)"-Mapa mostrando os pixeis que estao acima de 1.5ceu"

!do j=1,npby
!   l=npby-j+1
!   write(14,'(310i1)') (count2(i,l),i=1,npbx)
!enddo


!foi utilizado o formato '(100i1)' para que a imagem impressa
!seja de mais facil visualizacao


!========================================================================
!Esse laço será importante para o cálculo dos funcionais de Minkowski
do i=1,npbx
   do j=1,npby
      countf(i,j)=count2(i,j)
!      write(9,*) i,j,count2(i,j),countf(i,j)
   enddo
enddo
!========================================================================



!-------------------------------------------------------------------------
!-------------------------FRIENDS OF FRIENDS------------------------------

write(14,*)""
write(14,*) "FRIENDS OF FRIENDS"
write(14,*) "--------------------------------------------"

!Identificacao de objetos na banda r, utilizando a subrotina
!fof (friends of friends) do prof. Laerte Sodré Jr.

call fof2(npbx,npby,count2,itipo)

!o proximo passo novamente imprimi a imagem da galaxia segmentada
!e ja com os objetos identificados com o algortimo fof


!========================================================================
!DESCOMENTAR AS LINHAS ABAIXO, PARA GERAR OS MAPAS BINARIOS DA
!IMAGEM SEGMENTADA
!========================================================================
!write(14,*)"-----------------"
!write(14,*) ""
!write(14,*) "-Mapa com as estruturas identificadas pelo algoritmo FoF"
!do j=1,npby
!   l=npby-j+1
!   write(14,'(300i1)') (count2(i,l),i=1,npbx)	
!enddo


write(14,*)""

!-------------------------------------------------------------------------
!---------------------------QUARTIS DE CORES------------------------------

!aqui o count2=2 para o objeto detectado pelo fof

!utilizamos o vetor 1D xcor para facilitar no calculo dos percentis,
!onde somente os pixeis dtectados como pertencente a galaxia
!serao utilizados

write(14,*)""
write(14,*) "QUARTIS DE CORES"
write(14,*) "--------------------------------------------"


open(8,file="cores.txt")
k=0
do j=1,npby
   l=npby-j+1
   do i=1,npbx
	if(count2(i,j).eq.2)then
               k=k+1
               xcor(k)=corb(i,j)
		write(8,*)i,j,corb(i,j),k,xcor(k)
!		write(13,*) i,l,countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l)
         endif   
   enddo
enddo

!a variavel percent indica o valor alpha dos quartis que sera usado
!na subrotina percentis

percent=25.0
call percentis(xcor,k,percent,xxmin,xxmax,xxinf,xxmed,xxsup)

!xxmin,xxinf,xxmed,xxsup,xxmax são os valores minimo, inferior,
!medio, superior e maximo das cores

write(14,*)"numero de pixeis da cor g-r:",k
write(14,*)'cor: min 	       25             med 	      75	      max: '
write(14,*) xxmin,xxinf,xxmed,xxsup,xxmax
write(14,*)""

!aqui, k indica o numero de pixeis que foram detectados como pertencendo
!a um mesmo objeto, identificado pelo fof

 close(8)


!-------------------------------------------------------------------------
!-------------------------MAPAS DE POPULACOES-----------------------------

!Dividindo as cores em 4 populações, utilizando os quartis, vamos criar
!mapas espaciais com as 4 'populacoes' de cores


!countc = separa e recebe os valores das populacoes, segundo
!se abaixo do inferior => 1
!se entre inferior e media => 2
!se entre media e superior =>3
!se acima do superior => 4

k=0
do j=1,npby
   do i=1,npbx
       do l=1,4
           countc(l,i,j)=0
       enddo
       if(count2(i,j).eq.2.and.corb(i,j).le.xxinf)countc(1,i,j)=1 ! mais vermelho
       if(count2(i,j).eq.2.and.corb(i,j).gt.xxinf.and.corb(i,j).le.xxmed)countc(2,i,j)=1
       if(count2(i,j).eq.2.and.corb(i,j).gt.xxmed.and.corb(i,j).le.xxsup)countc(3,i,j)=1
       if(count2(i,j).eq.2.and.corb(i,j).gt.xxsup)countc(4,i,j)=1 ! mais azul
   enddo
enddo

open(3,file="out1.txt")
open(4,file="out2.txt")
open(5,file="out3.txt")
open(6,file="out4.txt")
open(12,file="contagens.txt")

write(14,*) "-Mapa com a distribuicao das 4 'populações' de cor "

!write(12,*) '# x   y   u  g  r  i  z'

! zerar os vetores que serão usados nos funcionais de minkowski
do j=1,npby
   l=npby-j+1
   do i=1,npbx 
   	countf1(i,l)=0
   	countf2(i,l)=0
   	countf3(i,l)=0
   	countf4(i,l)=0
   enddo
enddo


do j=1,npby
   l=npby-j+1
   do i=1,npbx 
       count3(i,l)=count2(i,l)
      if(countc(1,i,l).eq.1) then
		count3(i,l)=1
		countf1(i,l)=1
		write(3,*)i,l,corb(i,l),countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l)
		write(12,*) i,l,countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l),countb(6,i,l),&
			    & countb(7,i,l),countb(8,i,l),countb(9,i,l),countb(10,i,l),countb(11,i,l),countb(12,i,l)
	endif
       if(countc(2,i,l).eq.1) then
		count3(i,l)=2
		countf2(i,l)=1
		write(4,*) i,l,corb(i,l),countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l)
 		write(12,*) i,l,countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l),countb(6,i,l),&
			    & countb(7,i,l),countb(8,i,l),countb(9,i,l),countb(10,i,l),countb(11,i,l),countb(12,i,l)
	endif
       if(countc(3,i,l).eq.1) then
		count3(i,l)=3
		countf3(i,l)=1
		write(5,*) i,l,corb(i,l),countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l)
 		write(12,*) i,l,countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l),countb(6,i,l),&
			    & countb(7,i,l),countb(8,i,l),countb(9,i,l),countb(10,i,l),countb(11,i,l),countb(12,i,l)
	endif
       if(countc(4,i,l).eq.1)then 
		count3(i,l)=4
		countf4(i,l)=1
		write(6,*) i,l,corb(i,l),countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l)
 		write(12,*) i,l,countb(1,i,l),countb(2,i,l),countb(3,i,l),countb(4,i,l),countb(5,i,l),countb(6,i,l),&
			    & countb(7,i,l),countb(8,i,l),countb(9,i,l),countb(10,i,l),countb(11,i,l),countb(12,i,l)
	endif
   enddo
   write(14,'(300i1)')(count3(i,l),i=1,npbx)
enddo

 write(14,*) ""


 close(3)
 close(4)
 close(5)
 close(6)
 close(12)


!-------------------------------------------------------------------------
!------------------------------Centro Optico------------------------------

write(14,*)""
write(14,*)""
write(14,*) "CENTRO ÓPTICO"
write(14,*) "--------------------------------------------"


!centro optico, determina o pixel mais luminoso em r
!ixco e jyco = sao as coord do centro optico

fmax=-1.e30
do i=1,npbx
    do j=1,npby
       if(count2(i,j).eq.2.and.count1(i,j).gt.fmax)then
          fmax=count1(i,j)
          ixco=i
          jyco=j
        endif   
     enddo
enddo

!cc= valor do pixel no centro optico
write(14,*)'  fluxo	   	   X	      Y '
write(14,*) fmax,ixco,jyco
write(14,*)""

 cc=count2(ixco,jyco)
count2(ixco,jyco)=3

do j=1,npby
    l=npby-j+1
enddo

count2(ixco,jyco)=cc





 close(14)


stop

end program



!========================================================================
!==============================SUBROTINAS================================
!========================================================================


       SUBROUTINE avevar(data,n,ave,var)
      INTEGER n
      REAL ave,var,data(n)
      INTEGER j
      REAL s,ep
      ave=0.0
      do 11 j=1,n
        ave=ave+data(j)
11    continue
      ave=ave/n
      var=0.0
      ep=0.0
      do 12 j=1,n
        s=data(j)-ave
        ep=ep+s
        var=var+s*s
12    continue
      var=(var-ep**2/n)/(n-1)
      var=sqrt(var)
      return
      END
!  (C) Copr. 1986-92 Numerical Recipes Software YLu.

!========================================================================
      SUBROUTINE MDIAN1(X,N,YMED)
      DIMENSION X(N)
      CALL SORT(N,X)
      N2=N/2
      IF(2*N2.EQ.N)THEN
        YMED=0.5*(X(N2)+X(N2+1))
      ELSE
        YMED=X(N2+1)
      ENDIF
      RETURN
      END

!========================================================================

      SUBROUTINE SORT(N,RA)
      DIMENSION RA(N)
      L=N/2+1
      IR=N
10    CONTINUE
        IF(L.GT.1)THEN
          L=L-1
          RRA=RA(L)
        ELSE
          RRA=RA(IR)
          RA(IR)=RA(1)
          IR=IR-1
          IF(IR.EQ.1)THEN
            RA(1)=RRA
            RETURN
          ENDIF
        ENDIF
        I=L
        J=L+L
20      IF(J.LE.IR)THEN
          IF(J.LT.IR)THEN
            IF(RA(J).LT.RA(J+1))J=J+1
          ENDIF
          IF(RRA.LT.RA(J))THEN
            RA(I)=RA(J)
            I=J
            J=J+J
          ELSE
            J=IR+1
          ENDIF
        GO TO 20
        ENDIF
        RA(I)=RRA
      GO TO 10
      END

!========================================================================

!========================================================================

      subroutine fof2(npbx,npby,image,itipo)
!     fof problem:
!     Let us define objects as as a set of conected pixels of at least NCMIN
!     members, where each member has at least one neighbor from
!     the same group (friend). Given a segmented image, finds the objects in
!     the image (clusters of pixels).
!     laerte 30/03/2008  following Szapudi
!     revisao 090313
!
!     posicoes das galaxias: x(n),y(n)
!     imagem segmentada: image(i,j) = z(n)  pixels com valores 0 ou 1
!     icl(n): icl=0 se o pixel e' isolado; icl= no. do objeto a que ele
!             pertence

      parameter(nmax=10000,nm=200000,npmax=2048)
      real x(nm),y(nm),z(nm)
      integer icl(nmax),itipo(nmax),hist(nmax),image(npmax,npmax)
      real xx(nm),yy(nm)


      write(14,*)'FOF/Laerte'

      nt=npbx*npby
      if(nt.gt.nm)then
         write(14,*)' FoF: NPIX > NMAX!',nt,nm
         stop
      endif  
!     cria vetores com coordenadas dos pixels acima do limiar
!     e atribui um objeto a cada um
!     icl(i) e' a identificacao do objeto ao qual pertence o pixel i
!     pixel abaixo do limiar: itipo=0 acima: itipo=1
!     n: no. de pixels acima do limiar
      np=0 
      n=0
      do i=1,npbx
        do j=1,npby
           np=np+1
           if(image(i,j).eq.1)then
           n=n+1 
           x(n)=i
           y(n)=j
           icl(n)=n
           endif
        enddo
      enddo 
      write(14,*)'no. total de pixels: ',np
      write(14,*) 'no. de pixels acima do limiar: ',n

!     analisa a vizinhanca de cada pixel e redefine seu objeto
!     relacao entre os pixels: analisa um par de pixels de cada vez
!      do i=1,n-1
!         do j=i+1,n
      do i=1,n
         do j=1,n
            call  distancia(x(i),y(i),x(j),y(j),d)
!     se d=1 os pixels i e j sao conexos
            if(d.eq.1.)then
!     se ambos os pixels ja pertencem a algum objeto, atribuem-se os
!     dois ao objeto de menor numero de identificacao: ii
                  ii=min(icl(i),icl(j))
                  iclio=icl(i)
                  icljo=icl(j)
                  icl(i)=ii
                  icl(j)=ii
!     aos demais pixels dos dois objetos se atribui a mesma identificacao ii
                  do k=1,n
                     if(icl(k).eq.iclio.or.icl(k).eq.icljo)icl(k)=ii
                  enddo
            endif
         enddo
      enddo  

!     determinacao do numero de objetos resultantes
      do k=1,n
         hist(k)=0
      enddo  
      do k=1,n
         l=icl(k)
         hist(l)=hist(l)+1
      enddo  
      nct=0
      kmax=0
      do k=1,n
         if(hist(k).gt.0)then
            nct=nct+1
!           write(14,*)nct,hist(k)
            kmax=max(kmax,hist(k))
            if(hist(k).eq.kmax)iclmax=k
         endif   
      enddo 
      write(14,*)'n_objetos = ',nct,'  npix do maior objeto: ',kmax
      write(14,*)   '  ident. do maior objeto: ',iclmax

!     cria a imagem segmentada do maior objeto
      k=0
      do i=1,n
         ii=x(i)
         jj=y(i)
         image(ii,jj)=0
         if(icl(i).eq.iclmax)then
            image(ii,jj)=2
            k=k+1
!          xx(k)=x(i)
!          yy(k)=y(i)
         endif   
      enddo
      write(14,*)k
!      call plota(k,xx,yy)
      end


!========================================================================

      subroutine distancia(x,y,xl,yl,d)
!     pixels contiguos com z=zl=1: os 8 com d<=sqrt(2) -> d=1
!     dm2=1.5**2
      parameter(dm2=2.25)
      d=0.
      dist2=(x-xl)**2+(y-yl)**2
      if(dist2.le.dm2)d=1.
      return
      end

!========================================================================

      subroutine percentis(x,n,alfa,xmin,xmax,xinf,xmed,xsup)
!     percentis alfa (0.-50.) de x(n)
!     alfa=25 -> quartis
!     saida: percentil inferior, mediana e percentil superior
      parameter(nmax=100000)
      real x(n),xx(nmax)
      integer indx(nmax)

      if(n.gt.nmax)then
         write(*,*)' n > nmax!!! aumente nmax! '
         stop
      endif   
      if(alfa.ge.50..or.alfa.le.0.)then
         write(*,*)'alfa=',alfa,'     alfa: 0<alfa<50!!! '
         stop
      endif   

      call indexx(n,x,indx)
      xmin=x(indx(1))
      xmax=x(indx(n))
      ip=0
      ip25=0
      ip75=0
      n2=n/2
      ninf=n*alfa/100.
      nsup=n*(100.-alfa)/100.
      if(2*n2.eq.n)ip=1
      if((100.*ninf/alfa).eq.n)ipinf=1
      if((100.*nsup/(100.-alfa)).eq.n)ipsup=1
      if(ip.eq.1)then
         xmed=0.5*(x(indx(n2))+x(indx(n2+1)))        
      else
         xmed=x(indx(n2+1))
      endif   
      if(ipinf.eq.1)then
         xinf=0.5*(x(indx(ninf))+x(indx(ninf+1)))        
      else
         xinf=x(indx(ninf+1))
      endif    
      if(ipsup.eq.1)then
         xsup=0.5*(x(indx(nsup))+x(indx(nsup+1)))        
      else
         xsup=x(indx(nsup+1))
      endif  
      return
      end

!======================================================================== 
      SUBROUTINE indexx(n,arr,indx)
      INTEGER n,indx(n),M,NSTACK
      REAL arr(n)
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,indxt,ir,itemp,j,jstack,k,l,istack(NSTACK)
      REAL a
      do 11 j=1,n
        indx(j)=j
11    continue
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 13 j=l+1,ir
          indxt=indx(j)
          a=arr(indxt)
          do 12 i=j-1,1,-1
            if(arr(indx(i)).le.a)goto 2
            indx(i+1)=indx(i)
12        continue
          i=0
2         indx(i+1)=indxt
13      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        itemp=indx(k)
        indx(k)=indx(l+1)
        indx(l+1)=itemp
        if(arr(indx(l+1)).gt.arr(indx(ir)))then
          itemp=indx(l+1)
          indx(l+1)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l)).gt.arr(indx(ir)))then
          itemp=indx(l)
          indx(l)=indx(ir)
          indx(ir)=itemp
        endif
        if(arr(indx(l+1)).gt.arr(indx(l)))then
          itemp=indx(l+1)
          indx(l+1)=indx(l)
          indx(l)=itemp
        endif
        i=l+1
        j=ir
        indxt=indx(l)
        a=arr(indxt)
3       continue
          i=i+1
        if(arr(indx(i)).lt.a)goto 3
4       continue
          j=j-1
        if(arr(indx(j)).gt.a)goto 4
        if(j.lt.i)goto 5
        itemp=indx(i)
        indx(i)=indx(j)
        indx(j)=itemp
        goto 3
5       indx(l)=indx(j)
        indx(j)=indxt
        jstack=jstack+2
        if(jstack.gt.NSTACK)write(*,*) 'NSTACK too small in indexx'
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
      END
!  (C) Copr. 1986-92 Numerical Recipes Software YLu.

!========================================================================

!Subrotina que cálcula o indice de gini segundo a definição dada por Gini(1912)
!G=(1/2*Xmedio*n*(n-1)) * Somatoria(somatoria (abs(Xi-Xj))

subroutine gini(gaux,n,G)
implicit none

integer::i,j,n
real:: gaux(n)
real::soma,soma2,G,ni,cte,med,sigma

soma2=0.

do i=1,n
	soma=0.
	do j=1,n
		ni=abs(gaux(i)-gaux(j))
		soma=soma+ni
	enddo
	soma2=soma2+soma
enddo

call avevar(gaux,n,med,sigma)

 cte=(2*med*n*(n-1))
 G=(1/cte)*soma2


return
stop
end subroutine

!========================================================================
!Subrotina que cálcula o indice de gini segundo a definição dada por Glasser(1962)
!G=(1/Xmedio*n*(n-1)) * Somatoria((2*i-n-1)*Xi)


subroutine gini2(gaux,n,G)
implicit none

integer::i,j,n
real:: gaux(n)
real::soma,G,ni,cte,med,sigma

soma=0.

call sort(n,gaux)

do i=1,n
	ni=(2*i-n-1)*(gaux(i))
	soma=soma+ni
enddo

call avevar(gaux,n,med,sigma)

 cte=(med*n*(n-1))
 G=(1/cte)*soma


return
stop
end subroutine

!========================================================================
! subrotina Funcionais de minkowski

     subroutine minkowski(nx,ny,image,nobj,npixobj) 

!     fof problem: 
!     Let us define objects as as a set of conected pixels of at least NCMIN 
!     members, where each member has at least one neighbor from 
!     the same group (friend). Given a segmented image, finds the objects in 
!     the image (clusters of pixels). 
!     laerte 30/03/2008  following Szapudi 
!     revisao 090313 
! 
!     posicoes das galaxias: x(n),y(n) 
!     imagem segmentada: image(i,j) = z(n)  pixels com valores 0 ou 1 
!     icl(n): icl=0 se o pixel e' isolado; icl= no. do objeto a que ele 
!             pertence 
! 
!     nobj: no. de objetos acima do limiar 
!     npixobj(i): no. de pixels em cada objeto 

     parameter(nmax=10000,nm=200000,npmax=2048) 
     real x(nm),y(nm),z(nm) 
     integer icl(nmax),hist(nmax),image(npmax,npmax),npixobj(nmax) 
     integer icl1(nmax),imp(npmax,npmax),nper(nmax),npixobj2(nmax) 
     integer nbur(nmax) 

     nt=nx*ny 
     if(nt.gt.nm)then 
        write(14,*)' segmenta: NPIX > NMAX!',nt,nm 
        stop 
     endif   
!     cria vetores com coordenadas dos pixels acima do limiar 
!     e atribui um objeto a cada um 
!     icl(i) e' a identificacao do objeto ao qual pertence o pixel i 
!     n: no. de pixels acima do limiar 
     np=0 
     n=0 
     do i=1,nx 
       do j=1,ny 
          if(image(i,j).eq.1)then 
          n=n+1 
          x(n)=i 
          y(n)=j 
          icl(n)=n 
          endif 
       enddo 
     enddo 

     write(14,*) 'pixeis acima do limiar', n


!     analisa a vizinhanca de cada pixel e redefine seu objeto 
!     relacao entre os pixels: analisa um par de pixels de cada vez 
     do i=1,n 
        do j=1,n 
           call  distancia(x(i),y(i),x(j),y(j),d) 
!     se d=1 os pixels i e j sao conexos 
           if(d.eq.1.)then 
!     se ambos os pixels ja pertencem a algum objeto, atribuem-se os 
!     dois ao objeto de menor numero de identificacao: ii 
                 ii=min(icl(i),icl(j)) 
                 iclio=icl(i) 
                 icljo=icl(j) 
                 icl(i)=ii 
                 icl(j)=ii 
!     aos demais pixels dos dois objetos se atribui a mesma identificacao ii 
                 do k=1,n 
                    if(icl(k).eq.iclio.or.icl(k).eq.icljo)icl(k)=ii 
                 enddo 
           endif 
        enddo 
     enddo   

!     determinacao do numero de objetos resultantes 
!     npixobj(j): area (numero de pixels) do objeto j 
     do k=1,n 
        hist(k)=0 
     enddo   
     do k=1,n 
        l=icl(k) 
        hist(l)=hist(l)+1 
     enddo   
     nobj=0 
     do k=1,n 
        if(hist(k).gt.0)then 
           nobj=nobj+1 
           npixobj(nobj)=hist(k) 
           do l=1,n 
              if(icl(l).eq.k)icl1(l)=nobj 
           enddo   
        endif   
     enddo 

!     imagem com os numeros dos objetos 
     do k=1,n 
        i=x(k) 
        j=y(k) 
        if(image(i,j).gt.0)image(i,j)=icl1(k) 
     enddo 

!     perimetro dos objetos 
!     perimetro: numero de pixels do objeto em contacto com pixels "0" 
     write(14,*)' perimetro:' 
     do l=1,nobj 
        do i=1,nx 
           do j=1,ny 
              imp(i,j)=0 
           enddo 
        enddo   
        do k=1,n 
           i=x(k) 
           j=y(k) 
           if(image(i,j).eq.l)then 
              if(image(i-1,j-1).eq.0)imp(i,j)=1 
              if(image(i-1,j).eq.0)imp(i,j)=1 
              if(image(i-1,j+1).eq.0)imp(i,j)=1 
              if(image(i,j-1).eq.0)imp(i,j)=1 
              if(image(i,j+1).eq.0)imp(i,j)=1 
              if(image(i+1,j-1).eq.0)imp(i,j)=1 
              if(image(i+1,j).eq.0)imp(i,j)=1 
              if(image(i+1,j+1).eq.0)imp(i,j)=1 
           endif   
        enddo 
        nper(l)=0 
        do i=1,nx 
           do j=1,ny 
              if(imp(i,j).eq.1)nper(l)=nper(l)+1 
           enddo 
        enddo   
     enddo 

!     buracos nos objetos 
     write(14,*)' buracos:' 
     do l=1,nobj 
        do i=1,nx 
           do j=1,ny 
              imp(i,j)=0 
           enddo 
        enddo   
        do k=1,n 
           i=x(k) 
           j=y(k) 
           if(image(i,j).eq.l)then 
              if(image(i-1,j-1).eq.0)imp(i-1,j-1)=1 
              if(image(i-1,j).eq.0)imp(i-1,j)=1 
              if(image(i-1,j+1).eq.0)imp(i-1,j+1)=1 
              if(image(i,j-1).eq.0)imp(i,j-1)=1 
              if(image(i,j+1).eq.0)imp(i,j+1)=1 
              if(image(i+1,j-1).eq.0)imp(i+1,j-1)=1 
              if(image(i+1,j).eq.0)imp(i+1,j)=1 
              if(image(i+1,j+1).eq.0)imp(i+1,j+1)=1 
           endif   
        enddo 
        call segmenta(nx,ny,imp,nb,npixobj2) 
        nbur(l)=nb-1 
     enddo 

     write(14,'(10x,a)')'obj, area, perimetro, buracos:' 
     nbt=0 
     do l=1,nobj 
        write(14,*)l,npixobj(l),nper(l),nbur(l) 
        nbt=nbt+nbur(l) 
     enddo 
!     caracteristica de Euler-Poincare 
     nep=nobj-nbt 
     write(14,*)'caracteristica de Euler-Poincare: ',nep 

     end 
!======================================================================== 
     subroutine segmenta(nx,ny,image,nobj,npixobj) 
!     fof problem: 
!     Let us define objects as as a set of conected pixels of at least NCMIN 
!     members, where each member has at least one neighbor from 
!     the same group (friend). Given a segmented image, finds the objects in 
!     the image (clusters of pixels). 
!     laerte 30/03/2008  following Szapudi 
!     revisao 090313 
! 
!     posicoes das galaxias: x(n),y(n) 
!     imagem segmentada: image(i,j) = z(n)  pixels com valores 0 ou 1 
!     icl(n): icl=0 se o pixel e' isolado; icl= no. do objeto a que ele 
!             pertence 
! 
!     nobj: no. de objetos acima do limiar 
!     npixobj(i): no. de pixels em cada objeto 

     parameter(nmax=10000,nm=200000,npmax=2048) 
     real x(nm),y(nm),z(nm) 
     integer icl(nmax),hist(nmax),image(npmax,npmax),npixobj(nmax) 

     nt=nx*ny 
     if(nt.gt.nm)then 
        write(*,*)' segmenta: NPIX > NMAX!',nt,nm 
        stop 
     endif   
!     cria vetores com coordenadas dos pixels acima do limiar 
!     e atribui um objeto a cada um 
!     icl(i) e' a identificacao do objeto ao qual pertence o pixel i 
!     n: no. de pixels acima do limiar 
     np=0 
     n=0 
     do i=1,nx 
       do j=1,ny 
          if(image(i,j).eq.1)then 
          n=n+1 
          x(n)=i 
          y(n)=j 
          icl(n)=n 
          endif 
       enddo 
     enddo 

!     analisa a vizinhanca de cada pixel e redefine seu objeto 
!     relacao entre os pixels: analisa um par de pixels de cada vez 
     do i=1,n 
        do j=1,n 
           call  distancia(x(i),y(i),x(j),y(j),d) 
!     se d=1 os pixels i e j sao conexos 
           if(d.eq.1.)then 
!     se ambos os pixels ja pertencem a algum objeto, atribuem-se os 
!     dois ao objeto de menor numero de identificacao: ii 
                 ii=min(icl(i),icl(j)) 
                 iclio=icl(i) 
                 icljo=icl(j) 
                 icl(i)=ii 
                 icl(j)=ii 
!     aos demais pixels dos dois objetos se atribui a mesma identificacao ii 
                 do k=1,n 
                    if(icl(k).eq.iclio.or.icl(k).eq.icljo)icl(k)=ii 
                 enddo 
           endif 
        enddo 
     enddo   

!     determinacao do numero de objetos resultantes 
     do k=1,n 
        hist(k)=0 
     enddo   
     do k=1,n 
        l=icl(k) 
        hist(l)=hist(l)+1 
     enddo   
     nobj=0 
     do k=1,n 
        if(hist(k).gt.0)then 
           nobj=nobj+1 
           npixobj(nobj)=hist(k) 
        endif   
     enddo 

     end 
!======================================================================== 


