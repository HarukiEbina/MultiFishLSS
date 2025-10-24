from headers import *
from twoPoint import *
from definitions import MFISHLSS_BASE

'''
Values and defintions from Table 3 of Wilson and White 2019.
'''
zs = np.array([2.,3.,3.8,4.9,5.9])
Muvstar = np.array([-20.60,-20.86,-20.63,-20.96,-20.91])
Muvstar = interp1d(zs, Muvstar, kind='linear', bounds_error=False,fill_value='extrapolate')
muv = np.array([24.2,24.7,25.4,25.5,25.8])
muv = interp1d(zs, muv, kind='linear', bounds_error=False,fill_value='extrapolate')
phi = np.array([9.70,5.04,9.25,3.22,1.64])*0.001
phi = interp1d(zs, phi, kind='linear', bounds_error=False,fill_value='extrapolate')
alpha = np.array([-1.6,-1.78,-1.57,-1.60,-1.87])
alpha = interp1d(zs, alpha, kind='linear', bounds_error=False,fill_value='extrapolate')

def compute_covariance_matrix(fishcast, zbin_index, nratio=1,fskyratio = 1.):
   '''
   Returns an array of shape (npairs,npairs,Nk*Nmu). 
   '''
   nsample = len(fishcast.experiment.b)
   if nratio==1: nratio = np.ones(nsample)
   if not (isinstance(nratio,list) or isinstance(nratio,np.ndarray)): nratio=np.array([nratio])
   z = fishcast.experiment.zcenters[zbin_index]
   prefactor = (2.*np.pi**2.) / (fishcast.dk*fishcast.dmu*fishcast.Vsurvey[zbin_index]*fskyratio*fishcast.k**2.)
   Hz = fishcast.cosmo_fid.Hubble(z)*(299792.458)/fishcast.cosmo.h()
   sigma_parallel = (3.e5)*(1.+z)*fishcast.experiment.sigma_z/Hz
   P_fid = fishcast.P_fid[:,zbin_index,:]+0.
   if fishcast.recon: P_fid = fishcast.P_recon_fid[:,zbin_index]+0.
   for i in range(len(P_fid)):
        s1, s2 = fishcast.index2sample(i)
        if s1==s2: 
            number_density = compute_n(fishcast, z,s1)*np.maximum(np.exp(-fishcast.k**2. * fishcast.mu**2. * sigma_parallel**2.),1.e-20)
            P_fid[i]+= -1/compute_n(fishcast, z,s1)+1/nratio[s1]/number_density
   
   maxpair = int(nsample*(nsample-1)/2+nsample)
   C=np.zeros((maxpair,maxpair,fishcast.Nk*fishcast.Nmu))
   for i in range(maxpair):
       sample1,sample2 = fishcast.index2sample(i)
       for j in range(maxpair):
            sample3,sample4=fishcast.index2sample(j)
            tuples = np.array([[[sample1,sample3],[sample2,sample4]],[[sample1,sample4],[sample2,sample3]]])
            
            for k in range(2):
                C[i,j,:]+=P_fid[fishcast.sample2index(tuples[k,0,0],tuples[k,0,1]),:]* P_fid[fishcast.sample2index(tuples[k,1,0],tuples[k,1,1]),:]
   return np.maximum(np.einsum('ijk,k->ijk',C,prefactor), 1e-50*np.ones(C.shape))  # avoiding numerical nonsense with possible 0's               

def covariance_Cls(fishcast,kmax_knl=1.,CMB='SO',fsky_CMB=0.4,fsky_intersect=None,nratio=1,fskyratio=1.):
   '''
   Returns a covariance matrix Cov[X,Y] as a function of l. X (and Y) is in the basis
   
        X \in {k-k, k-g1, ..., k-gn, g1-g1, ..., gn-gn}   (the basis has dimension 2*n+1)
   
   where g1 is the galaxies in the first redshift bin, k-gi is the cross-correlation of
   the CMB kappa map and the galaxies in the i'th bin, and so on.
   
   This function returns a numpy.array with shape (2*n+1,2*n+1,len(l)), where l = fishcast.l
   '''

   n = fishcast.experiment.nbins
   zs = fishcast.experiment.zcenters
   zes = fishcast.experiment.zedges
   nsamples = len(fishcast.experiment.b)
   npairs = int(nsamples*(nsamples+1)/2)
   if nratio==1: nratio = np.ones(nsamples)
   if not (isinstance(nratio,list) or isinstance(nratio,np.ndarray)): nratio=np.array([nratio])

   # Lensing noise
   if CMB == 'SO': 
      data = np.genfromtxt(os.path.join(MFISHLSS_BASE, 'input/nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat'))
      l,N = data[:,0],data[:,7]
   elif CMB == 'Planck': 
      data = np.genfromtxt(os.path.join(MFISHLSS_BASE, 'input/nlkk_planck.dat'))
      l,N = data[:,0],data[:,1]
   elif CMB == 'Perfect':
      l,N = fishcast.ell, fishcast.ell*0
   elif CMB == 'Powerful': 
      data = np.genfromtxt(os.path.join(MFISHLSS_BASE, 'input/S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat'))
      l,N = data[:,0],data[:,7]/100
   else: 
      data = np.genfromtxt(os.path.join(MFISHLSS_BASE, 'input/S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat'))
      l,N = data[:,0],data[:,7]
   
   Nkk_interp = interp1d(l,N,kind='linear',bounds_error=False, fill_value=1)
   l = fishcast.ell  ; Nkk = Nkk_interp(l)
   # Cuttoff high ell by blowing up the covariance for ell > ellmax
   chi = lambda z: (1.+z)*fishcast.cosmo_fid.angular_distance(z)*fishcast.cosmo.h()
   ellmaxs =  np.array([kmax_knl*chi(z)/np.sqrt(fishcast.Sigma2(z)) for z in zs])
   constraint = np.ones((n,len(l)))
   idx = np.array([np.where(l)[0] >= ellmax for ellmax in ellmaxs])
   for i in range(n): constraint[i][idx[i]] *= 1e10
   # relevant fsky's
   fsky_LSS = fishcast.experiment.fsky*fskyratio
   if fsky_intersect is None: fsky_intersect = min(fsky_LSS,fsky_CMB)  # full-overlap by default  
   # build covariance matrix
   Clen = npairs*n+nsamples*n+1
   C = np.zeros((Clen,Clen,len(l)))
   # kk,kk component of covariance
   Ckk = fishcast.Ckk_fid
   C[0,0] = 2*(Ckk + Nkk)**2/(2*l+1) / fsky_CMB
   #
   noise = np.zeros((npairs, n, len(l)))
   for i in range(n):
       for j in range(npairs):
            s1, s2 = fishcast.index2sample(j)
            if s1==s2: noise[j,i,:]=compute_lensing_Cell(fishcast,s1,s2,zmin=zes[i],zmax=zes[i+1],noise=True)
            
   def change_noise(Cgigi,noisei):
       Carr = Cgigi+0
       for i1 in range(npairs):
            s1, s2 = fishcast.index2sample(i1)
            if s1==s2: Carr[i1]+= (1/nratio[s1]-1)*noisei[i1]
       return Carr

   for i in range(n): #n=nbins
      Ckgi = fishcast.Ckg_fid[:,i] 
      Cgigi = fishcast.Cgg_fid[:,i]
      Cgigi = change_noise(Cgigi,noise[:,i,:])
      for k in range(npairs):
          if k<nsamples:
              # kk, kg
              C[k*n+i+1,0] = 2*(Ckk + Nkk) * Ckgi[k]/(2*l+1)*constraint[i] / fsky_CMB
              C[0,k*n+i+1] = C[k*n+i+1,0]
          # kk, gg
          sample1,sample2 = fishcast.index2sample(k)
          C[i+1+nsamples*n+k*n,0] = 2*Ckgi[sample1]*Ckgi[sample2]/(2*l+1)*constraint[i] * fsky_intersect / fsky_LSS / fsky_CMB
          C[0,i+1+nsamples*n+k*n] = C[i+1+nsamples*n+k*n,0]
          for j in range(n):
             Ckgj = fishcast.Ckg_fid[:,j]
             Cgjgj = fishcast.Cgg_fid[:,j]
             Cgjgj = change_noise(Cgjgj,noise[:,j,:])

             #gg gg
             for k2 in range(npairs):
                 #if k>k2:continue
                 if k<nsamples and k2<nsamples:
                     ind = fishcast.sample2index(k,k2)
                     # kgi, kgj
                     C[k*n+i+1,k2*n+j+1] = Ckgi[k]*Ckgj[k2]*constraint[i]*constraint[j]
                     if i == j:C[k*n+i+1,k2*n+j+1] += (Ckk + Nkk)*Cgigi[ind]*constraint[i]
                     C[k*n+i+1,k2*n+j+1] /= (2*l+1) * fsky_intersect
                     C[k2*n+j+1,k*n+i+1]=C[k*n+i+1,k2*n+j+1]
                 sample3,sample4=fishcast.index2sample(k2)
                 tuples = np.array([[[sample1,sample3],[sample2,sample4]],[[sample1,sample4],[sample2,sample3]]])
                 indices = np.zeros((2,2))
                 for m1 in range(2):
                    for m2 in range(2):
                        if tuples[m1,m2,0]>=tuples[m1,m2,1]:tuples[m1,m2,:]=np.flip(tuples[m1,m2,:])
                        indices[m1,m2]=fishcast.sample2index(tuples[m1,m2,0],tuples[m1,m2,1])
                 #print(indices)    
                 # gigi, gjgj
                 if i == j: 
                    C[k*n+i+1+n*nsamples,k2*n+j+1+n*nsamples]=(Cgigi[round(indices[0,0])]*Cgigi[round(indices[0,1])]+ Cgigi[round(indices[1,0])]*Cgigi[round(indices[1,1])])/(2*l+1)*constraint[i]/fsky_LSS
                    C[k2*n+j+1+n*nsamples,k*n+i+1+n*nsamples]=C[k*n+i+1+n*nsamples,k2*n+j+1+n*nsamples] 
                 # kgi, gjgj
                 if i == j:
                    if k<nsamples:
                        C[k*n+i+1,k2*n+i+1+n*nsamples]=Ckgi[sample3]*Cgigi[round(fishcast.sample2index(k,sample4))]+                        Ckgi[sample4]*Cgigi[round(fishcast.sample2index(k,sample3))]
                        C[k*n+i+1,k2*n+i+1+n*nsamples]*=constraint[i]/(2*l+1)/fsky_LSS
                        C[k2*n+i+1+n*nsamples,k*n+i+1] = C[k*n+i+1,k2*n+i+1+n*nsamples]
   return C