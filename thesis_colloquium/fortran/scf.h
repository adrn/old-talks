C=======================================================================
C
C
C                        INCLUDE FILE scf.h
C
C
C=======================================================================
C
C
C     Parameter declarations, allocation of array storage, common
C     block definitions.
C
C
C=======================================================================

        INTEGER nbodsmax,nmax,lmax
 
        PARAMETER(nbodsmax=100001,nmax=6,lmax=4)

        CHARACTER*50 headline
        CHARACTER*4 outfile
        INTEGER nsteps,noutbod,nbodies,noutlog,ibound,ntide,nsort,
     &          nlumps,nlumpsin
        LOGICAL selfgrav,inptcoef,outpcoef,zeroodd,zeroeven,fixacc,
     &          david,disk,flat
        REAL*8 tnow,x,y,z,vx,vy,vz,mass,pot,dtime,G,ax,ay,az,one,pi,
     &         twoopi,onesixth,tpos,tvel,
     &         potext,two,zero,tiny,
     &         mrem,ekrem,eprem,xframe,yframe,zframe,vxframe,
     &         vyframe,vzframe,sinsum,cossum,ep,strength,ru,mu,dt0,
     &         tfac2,c,vh2,tend,
     &         ml,rl,Gmh,rh,vu,tub
	REAL cputime0,cputime1,cputime,tick
        COMMON/charcom/outfile
        COMMON/bodscom/x(nbodsmax),y(nbodsmax),z(nbodsmax),vx(nbodsmax),
     &                 vy(nbodsmax),vz(nbodsmax),mass(nbodsmax),
     &                 pot(nbodsmax),ax(nbodsmax),ay(nbodsmax),
     &                 az(nbodsmax),potext(nbodsmax),ep(nbodsmax),
     &                 tub(nbodsmax)
        COMMON/parcomi/nbodies,nsteps,noutbod,noutlog,ibound(nbodsmax),
     &                 ntide,nsort,nlumps,nlumpsin
        COMMON/parcomr/dtime,G,one,pi,twoopi,onesixth,two,tiny,zero
        COMMON/parcomc/headline
        COMMON/parcoml/selfgrav,inptcoef,outpcoef,zeroodd,zeroeven,
     &                 fixacc,david,disk,flat
        COMMON/timecom/tpos,tnow,tvel,dt0,tfac2,tend
        COMMON/cpucom/cputime0,cputime1,cputime,tick
        COMMON/coefcom/sinsum(0:nmax,0:lmax,0:lmax),
     &                 cossum(0:nmax,0:lmax,0:lmax)
        COMMON/remcom/mrem,ekrem,eprem
        COMMON/orbcom/xframe,yframe,zframe,vxframe,vyframe,vzframe,
     &                strength,ru,mu,c,vh2
        COMMON/lumpcom/ml,rl,GMh,rh,vu

C=======================================================================
C   Definitions specific to input/output.
C=======================================================================
        INTEGER uterm,upars,ulog,ubodsin,ubodsout,utermfil,uoutcoef,
     &          uincoef,ubodsel,uorb,ucen,ucpu
        CHARACTER*8 parsfile,logfile,termfile,
     &              outcfile,incfile,elfile,partfile,cenfile,cpufile
        CHARACTER*8 ibodfile,obodfile
        PARAMETER(uterm=6,upars=10,ulog=11,ubodsin=12,ubodsout=13,
     &            utermfil=15,uoutcoef=16,uincoef=17,ubodsel=18,
     &            uorb=19,ucpu=20,ucen=21)
        PARAMETER(parsfile='SCFPAR',logfile='SCFLOG',
     &            ibodfile='SCFBI',
     &            obodfile='SNAPxxxx',
     &            termfile='SCFOUT',outcfile='SCFCOEF',
     &            incfile='SCFICOEF',elfile='SCFELxxx',
     &            partfile='SCFORB',
     &            cpufile='SCFCPU',cenfile='SCFCEN')





