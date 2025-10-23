from headers import *
from twoPoint import *
from twoPointNoise import *
# from deprecated.castorina import castorinaBias,castorinaPn
from multiprocessing import Pool
from functools import partial
import os, json
from os.path import exists


class fisherForecast(object):
   '''
   Class for computing derivatives of the galaxy (full-shape)
   power spectrum, as well as the CMB lensing cross-/auto spectrum.
   Can build and combine Fisher matrices formed from both of these
   observables.
   '''
   def __init__(self, 
                cosmo=None, 
                cosmo_fid=None,
                experiment=None, 
                kmin=5e-4, 
                kmax=1., 
                Nmu=50, 
                Nk=500, 
                free_params=np.array(['h']),
                fEDE=0., 
                log10z_c=3.56207, 
                thetai_scf=2.83, 
                A_lin=0., 
                omega_lin=0.01, 
                phi_lin=np.pi/2., 
                A_log=0., 
                omega_log=0.01, 
                phi_log=np.pi/2., 
                velocileptors=True,
                linear=False, 
                linear2=False,
                name='toy_model',
                smooth=False,
                AP=True,
                recon=False,
                ell=np.arange(10,1000,1),
                N2cut=0.2,
                setup=True,
                overwrite=False,
                basedir=''):
        
      # some basic stuff for the k-mu grid  
      self.kmin = kmin
      self.kmax = kmax
      self.Nmu = Nmu
      self.Nk = Nk
      # free parameters in the forecast
      self.free_params = free_params
      # parameters for EDE. These are redundant if fEDE has a nonzero fiducial value.
      self.fEDE = fEDE
      self.log10z_c = log10z_c
      self.thetai_scf = thetai_scf
      #
      # parameters for primordial wiggles
      self.A_lin = A_lin 
      self.omega_lin = omega_lin
      self.phi_lin = phi_lin
      self.A_log = A_log 
      self.omega_log = omega_log
      self.phi_log = phi_log
      #
      self.velocileptors = velocileptors
      self.linear = linear
      self.linear2 = linear2
      self.name = name
      self.smooth = smooth
      self.AP = AP
      self.recon = recon
      self.ell = ell
      self.N2cut = N2cut
      self.basedir = basedir
      
      # Set up the k-mu grid 
      k = np.logspace(np.log10(self.kmin),np.log10(self.kmax),self.Nk) # [h/Mpc]
      dk = list(k[1:]-k[:-1])
      dk.insert(0,dk[0])
      dk = np.array(dk)
      self.k = np.repeat(k,self.Nmu)
      self.dk = np.repeat(dk,self.Nmu)
      #
      mu = np.linspace(0.,1.,self.Nmu)
      dmu = list(mu[1:]-mu[:-1])
      dmu.append(dmu[-1])
      dmu = np.array(dmu)
      self.mu = np.tile(mu,self.Nk)
      self.dmu = np.tile(dmu,self.Nk)

      # we set these up later
      self.experiment = None     # 
      self.cosmo = None          # CLASS object
      self.P_fid = None          # fidicual power spectra at the center of each redshift bin
      self.P_recon_fid = None    # fiducial reconstructed power spectra "..."
      self.Vsurvey = None        # comoving volume [Mpc/h]^3 in each redshift bin
      self.params = None         # CLASS parameters
    
      # make directories for storing derivatives and fiducial power spectra
      o,on = basedir+'output/',basedir+'output/'+self.name+'/'
      directories = np.array([o, on, on+'/derivatives/', on+'/derivatives_Cl/', on+'/derivatives_recon/'])
      for directory in directories: 
         if not os.path.exists(directory): os.makedirs(directory,exist_ok=True)

      if (cosmo is None) or (experiment is None):
         print('Attempted to create a forecast without an experiment or cosmology.')     
      else:
         self.set_experiment_and_cosmology_specific_parameters(experiment, cosmo, cosmo_fid)
      
      self.create_json_summary()
      
      if setup or overwrite: 
            self.compute_fiducial_Pk(overwrite=overwrite)
            self.compute_fiducial_Cl(overwrite=overwrite)
            self.compute_fiducial_Precon(overwrite=overwrite)



                                     ##
                                   #### 
                                  #####  
                                    ###
                                    ###
                                    ###
                                    ###
                                    ###
                                  #######  
   
   
   #######################################################################
   #######################################################################
   ## This section of the code is used to set up the forecast object 
   ## Computes (or loads) fiducial power spectra, sets up distances, ...
   
        
   def set_experiment_and_cosmology_specific_parameters(self, experiment, cosmo, cosmo_fid):
      '''
      Set up the experiment- and cosmology-dependent pieces of the forecast
      '''
      self.experiment = experiment
      self.cosmo = cosmo
      params = cosmo.pars
      
      # it's useful when taking derivatives 
      # to have access to the fiducial cosmology
      if cosmo_fid is None:
         cosmo_fid = Class()
         cosmo_fid.set(params)
         cosmo_fid.compute()
      self.cosmo_fid = cosmo_fid 
      
      self.params = params
      self.params_fid = cosmo_fid.pars
      if 'log(A_s)' in self.free_params and 'A_s' not in np.array(list(params.keys())):
         print('Must include A_s in CLASS params if trying to take derivatives w.r.t. log(A_s).')
         return
      if 'fEDE' in np.array(list(params.keys())):
         self.fEDE = params['fEDE']
         self.log10z_c = params['log10z_c']
         self.thetai_scf = params['thetai_scf']

      # Fiducial volume in each redshift bin
      edges = experiment.zedges
      num_bins = len(experiment.zedges) - 1
      self.Vsurvey = np.array([self.comov_vol(edges[i],edges[i+1]) for i in range(num_bins)])
          
      # Fiducial distances as a function of z
      # Da = angular diameter distance [Mpc/h]
      # Hz = hubble [h km/s/Mpc]
      zs = np.linspace(0,60,1000)
      # h = self.params['h']
      h = self.cosmo.h()
      c = 299792.458 # speed of light in km/s
      Da_fid = np.array([self.cosmo_fid.angular_distance(z)*h for z in zs])
      Hz_fid = np.array([self.cosmo_fid.Hubble(z)*c/h for z in zs])
      # interpolate
      self.Da_fid = interp1d(zs,Da_fid,kind='linear')
      self.Hz_fid = interp1d(zs,Hz_fid,kind='linear')

        
   def compute_fiducial_Pk(self, overwrite=False,ell_mult = None):
      '''
      Either compute or load fiducial full-shape power spectra
      '''
      if isinstance(self.experiment.b,list):nsamples = len(self.experiment.b)
      else:nsamples=1
      self.P_fid = np.zeros((int(nsamples*(nsamples-1)/2+nsamples),self.experiment.nbins,self.Nk*self.Nmu))
        
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
         for j in range(nsamples):
             for k in range(nsamples):
                 if j>k:continue
                 samplename1=self.experiment.samples[j]
                 samplename2=self.experiment.samples[k]
                 ind=self.sample2index(j,k)
                 # P(k)
                 fname = self.basedir+'output/'+self.name+'/derivatives/pfid_'+samplename1+samplename2+'_'+str(round(100*z))+'.txt'
                 if not exists(fname) or overwrite:  
                    self.P_fid[ind,i,:] = compute_tracer_power_spectrum(self,j,k,z,ell_mult=ell_mult)
                    np.savetxt(fname,self.P_fid[ind,i,:])
                 else:
                    self.P_fid[ind,i,:] = np.genfromtxt(fname)
            
      # setup the k_par cut  
      self.kpar_cut = np.ones((int(nsamples*(nsamples-1)/2+nsamples),self.experiment.nbins,self.Nk*self.Nmu))
      for i in range(self.experiment.nbins): 
        z = self.experiment.zcenters[i]
        for j in range(nsamples):
            for k in range(nsamples):
                if j>k:continue
                ind = self.sample2index(j,k)
                self.kpar_cut[ind,i,:] = self.compute_kpar_cut(z,i,j,k,ind)
        
        
   def compute_fiducial_Cl(self, overwrite=False):
      '''
      Either compute or load fiducial C_ells (Ckk, Ckg, Cgg)
      '''
      if isinstance(self.experiment.b,list):nsamples = len(self.experiment.b)
      else:nsamples=1
      npairs = int(nsamples*(nsamples-1)/2+nsamples)
      self.Ckk_fid = np.zeros(len(self.ell))
      self.Ckg_fid = np.zeros((nsamples,self.experiment.nbins,len(self.ell)))
      self.Cgg_fid = np.zeros((int(nsamples*(nsamples+1)/2),self.experiment.nbins,len(self.ell)))

      # Ckk    
      fname = self.basedir+'output/'+self.name+'/derivatives_Cl/Ckk_fid.txt'
      if not exists(fname) or overwrite:  
         #self.C_fid[0,:,:] = compute_lensing_Cell(self,'k','k')
         self.Ckk_fid = compute_lensing_Cell(self,'k','k')
         np.savetxt(fname,self.Ckk_fid)
      else:
         self.Ckk_fid = np.genfromtxt(fname)
         
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
         for j in range(nsamples):
             name1=self.experiment.samples[j]
             # Ckg
             fname = self.basedir+'output/'+self.name+'/derivatives_Cl/Ck'+name1+'_fid_'+str(round(100*zmin))+'_'+str(round(100*zmax))+'.txt'
             if not exists(fname) or overwrite:
                self.Ckg_fid[j,i] = compute_lensing_Cell(self,'k',j,zmin,zmax)
                np.savetxt(fname,self.Ckg_fid[j,i])
             else: 
                self.Ckg_fid[j,i] = np.genfromtxt(fname)
             for k in range(nsamples):
                if j>k:continue
                #Cgg
                name2=self.experiment.samples[k]
                ind=self.sample2index(j,k)
                fname= self.basedir+'output/'+self.name+'/derivatives_Cl/C'+name1+name2+ '_fid_'+str(round(100*zmin))+'_'+str(round(100*zmax))+'.txt'
                if not exists(fname) or overwrite:
                   self.Cgg_fid[ind,i] = compute_lensing_Cell(self,j,k,zmin,zmax)
                   np.savetxt(fname,self.Cgg_fid[ind,i])
                else:
                   self.Cgg_fid[ind,i] = np.genfromtxt(fname)


   def compute_fiducial_Precon(self, overwrite=False):
      '''
      Either compute or load reconstructed power spectra
      '''
      if isinstance(self.experiment.b,list):nsamples = len(self.experiment.b)
      else:nsamples=1
      self.P_recon_fid = np.zeros((int(nsamples*(nsamples+1)/2),self.experiment.nbins,self.Nk*self.Nmu))
        
      # Compute fiducial power spectra in each redshift bin
      for i in range(self.experiment.nbins):
         z = self.experiment.zcenters[i]
         zmin = self.experiment.zedges[i]
         zmax = self.experiment.zedges[i+1]
         for j in range(nsamples):
             for k in range(nsamples):
                 if j>k:continue
                 samplename1=self.experiment.samples[j]
                 samplename2=self.experiment.samples[k]
                 ind=self.sample2index(j,k)
                 # P(k)
                 fname = self.basedir+'output/'+self.name+'/derivatives_recon/pfid_'+samplename1+samplename2+'_'+str(round(100*z))+'.txt'
                 if not exists(fname) or overwrite:  
                    self.recon = True
                    self.P_recon_fid[ind,i,:] = compute_tracer_power_spectrum(self,j,k,z)
                    self.recon = False
                    np.savetxt(fname,self.P_recon_fid[ind,i,:])
                 else:
                    self.P_recon_fid[ind,i,:] = np.genfromtxt(fname)
                      
         
   def create_json_summary(self):
      '''
      Creates and stores a .json file in output/forecast_name/
      This file summarizes some of the basic forecast details
      '''
      ze = list(self.experiment.zedges)
      zs = self.experiment.zcenters
      bs = list([[float(compute_b(self,z,i)) for z in zs] for i in range(len(self.experiment.b))])
      # if self.experiment.HI:
      #     ns = list([float(castorinaPn(z)) for z in zs])
      # else:
      ns = list([[float(compute_b(self,z,i)) for z in zs] for i in range(len(self.experiment.b))])
      
      data = {'Forecast name': self.name,
              'Edges of redshift bins': ze,
              'Centers of redshift bins': list(zs),
              'Linear Eulerian bias in each bin': bs,
              'Number density in each bin': ns,
              'fsky': self.experiment.fsky,
              'CLASS default parameters': self.params_fid}

      with open(self.basedir+'output/'+self.name+'/'+'summary.json', 'w') as write_file:
         json.dump(data, write_file, indent=2)
  

   def compute_kpar_cut(self,z,zindex=None,samplenumber1=-1,samplenumber2=-1,sampleindex=-1):
      # Create a "k_parallel cut", removing modes whose N2 term
      # is >= 100*N2cut % of the total power
      def get_N(z):
         # if self.experiment.HI: return 1/castorinaPn(z)
         # else: 
         return self.experiment.fover[sampleindex]/np.sqrt(compute_n(self,z,samplenumber1)*compute_n(self,z,samplenumber2))
      if sampleindex==-1: sampleindex=self.sample2index(samplenumber1,samplenumber2)
      elif samplenumber1 == -1: 
        samplenumber1,samplenumber2 = self.index2sample(sampleindex)
      sigv = self.experiment.sigv
      sn2 = (self.k*self.mu)**2*((1+z)*sigv/self.Hz_fid(z))**2*get_N(z)
      if zindex is not None: Ps = self.P_fid[sampleindex,zindex]
      else: Ps = compute_tracer_power_spectrum(self,z)
      idx = np.where(sn2/Ps >= self.N2cut) ; idx2 = np.where(Ps <= 0)
      kpar_cut = np.ones(self.Nk*self.Nmu)
      kpar_cut[idx] = 0 ; kpar_cut[idx2] = 0 
      return kpar_cut

         
   def comov_vol(self,zmin,zmax):
      '''
      Returns the comoving volume in (Mpc/h)^3 between 
      zmin and zmax, assuming that the universe is flat.
      Includes the fsky of the experiment.
      '''
      chi = lambda z: (1+z)*self.cosmo.angular_distance(z)
      # h = self.params_fid['h']
      h = self.cosmo_fid.h()
      vsmall = (4*np.pi/3) * chi(zmin)**3.
      vbig = (4*np.pi/3) * chi(zmax)**3.
      return self.experiment.fsky*(vbig - vsmall)*h**3.

                                      
                                  ### 
                                #######
                               #########
                               ###   ### 
                               ###   ###
                                     ###
                                    ###
                                  ###
                                 ###
                               #########
                               #########
   
   
   #######################################################################
   #######################################################################
   ## This section contains some helper-functions
   ##
   def index2sample(self,index):
       nsample=len(self.experiment.b)
       if index>=nsample*(nsample+1)/2: print('invalid index')
       count=0
       ind1=0
       ind2=0
       for i in range(nsample-1):         
           if index<count+nsample-i:
               break
           ind1+=1
           count+=nsample-i
       ind2=index-count+ind1
       return ind1,ind2

   def sample2index(self,sample1,sample2):
       if sample1>sample2:
            temp=sample1
            sample1=sample2
            sample2=temp
       nsample=len(self.experiment.b)
       if sample1>=nsample or sample2>=nsample: print('invalid sample')
       count=0
       for i in range(1,sample1+1):
           count+=nsample-i
       return count+sample2

   def get_listparams(self,listparams,lensing=False,auto_only=False):
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples+1)/2)
      biases = ['b','b2','bs']
      Ns = ['N','N2','N4']
      alphas =['alpha0','alpha2','alpha4']
      bidx=None; b2idx=None; bsidx=None
      
      for param in biases: 
          try: idx=listparams.index(param)
          except: idx=None
          if idx is not None: 
              listparams.pop(idx)
              for i in range(nsamples):
                  listparams.insert(idx+i,param+self.experiment.samples[i])
      for param in alphas: 
          try: idx=listparams.index(param)
          except: idx=None
          if idx is not None: 
              listparams.pop(idx)
              for i in range(nsamples):
                  listparams.insert(idx+i,param+self.experiment.samples[i])
              if lensing: listparams.insert(idx+nsamples,param+'k')
      for param in Ns: 
          try: idx=listparams.index(param)
          except: idx=None
          if idx is not None: 
              listparams.pop(idx)
              counter = 0
              for i in range(npairs):
                  s1,s2 = self.index2sample(i)
                  if s1==s2 or not auto_only: 
                        listparams.insert(idx+counter,param+self.experiment.samples[s1]+self.experiment.samples[s2])
                        counter+=1
      return np.array(listparams)  
      
    
    
   def LegendreTrans(self,l,p,mu_max=1.):
      '''
      Returns the l'th multipole of P(k,mu), where P(k,mu) is a 
      vector of length Nk*Nmu. Returns a vector of length Nk.
      '''
      n = self.Nk ; m = self.Nmu
      mu = self.mu.reshape((n,m))[0]
      p_reshaped = p.reshape((n,m))
      result = np.zeros(n)
      for i in range(n):
         integrand = (2*l+1)*p_reshaped[i,:]*scipy.special.legendre(l)(mu)
         result[i] = scipy.integrate.simps(integrand,x=mu)
      return result
      
      
   def LegendreTransInv(self,pls):
      '''
      Given pls = [p0,p2,p4,...], where each pn is an 
      array of length Nk, returns P(k,mu), which is 
      an array of length Nk*Nmu.
      '''
      n = len(pls)
      pls_repeat = np.zeros((n,self.Nk*self.Nmu))
      legendre_polys = pls_repeat.copy()
      for i in range(n): 
         pls_repeat = np.repeat(pls[i],self.Nmu)
         legendre_polys = scipy.special.legendre(2*i)(self.mu)
      return np.sum(pls_repeat*legendre_polys,axis=0)
      
      
   def dPdk(self,P):
      '''
      Given an input array P (length Nk*Nmu), which is 
      defined on the (flattened) k-mu grid, returns 
      dPdk on that same grid. 
      
      assumes that k is log spaced
      '''
      P_reshaped = P.reshape((self.Nk,self.Nmu)) 
      P_low  = np.roll(P_reshaped,1,axis=0)
      P_low2 = np.roll(P_reshaped,2,axis=0)
      P_low3 = np.roll(P_reshaped,3,axis=0) 
      P_low4 = np.roll(P_reshaped,4,axis=0)
      P_hi   = np.roll(P_reshaped,-1,axis=0)
      P_hi2  = np.roll(P_reshaped,-2,axis=0)
      P_hi3  = np.roll(P_reshaped,-3,axis=0)
      P_hi4  = np.roll(P_reshaped,-4,axis=0)
      dP     = (-P_hi2 + 8.*P_hi - 8.*P_low + P_low2) / 12.
      dP_low = -(-3.*P_low4 + 16.*P_low3 - 36.*P_low2 + 48.*P_low - 25.*P_reshaped)/12.
      dP_hi  = (-3.*P_hi4 + 16.*P_hi3 - 36.*P_hi2 + 48.*P_hi - 25.*P_reshaped)/12.
      # correct for "edge effects"
      dP[:5,:] = dP_hi[:5,:]
      dP[-5:,:] = dP_low[-5:,:]  
      ks = self.k.reshape((self.Nk,self.Nmu))[:,0]
      dlnk = np.log(ks)[1]-np.log(ks)[0] 
      dPdk = dP.flatten()/dlnk/self.k
      return dPdk
   
   
   def dPdmu(self,P):
      '''
      Given an input array P (length Nk*Nmu), which is 
      defined on the (flattened) k-mu grid, returns 
      dPdmu on that same grid. 
      
      ssumes that mu is lin spaced
      '''
      P_reshaped = P.reshape((self.Nk,self.Nmu)) 
      P_low  = np.roll(P_reshaped,1,axis=1)
      P_low2 = np.roll(P_reshaped,2,axis=1)
      P_low3 = np.roll(P_reshaped,3,axis=1) 
      P_low4 = np.roll(P_reshaped,4,axis=1)
      P_hi   = np.roll(P_reshaped,-1,axis=1)
      P_hi2  = np.roll(P_reshaped,-2,axis=1)
      P_hi3  = np.roll(P_reshaped,-3,axis=1)
      P_hi4  = np.roll(P_reshaped,-4,axis=1)
      dP     = (-P_hi2 + 8.*P_hi - 8.*P_low + P_low2) / 12.
      dP_low = -(-3.*P_low4 + 16.*P_low3 - 36.*P_low2 + 48.*P_low - 25.*P_reshaped)/12.
      dP_hi  = (-3.*P_hi4 + 16.*P_hi3 - 36.*P_hi2 + 48.*P_hi - 25.*P_reshaped)/12.
      # correct for "edge effects"
      dP[:,:5] = dP_hi[:,:5]
      dP[:,-5:] = dP_low[:,-5:] 
      dmu = self.mu[1] - self.mu[0]
      return dP.flatten()/dmu
    
    
   def compute_dPdp(self, param,X,Y, z, relative_step=-1., absolute_step=-1., 
                    one_sided=False, five_point=False,kwargs=None):
      '''
      Calculates the derivative of the galaxy power spectrum
      with respect to the input parameter around the fidicual
      cosmology and redshift. Returns an array of length Nk*Nmu.
      '''
      def paramsample(param,basestr):
          nsamples=len(self.experiment.b)
          paramarr = [basestr+self.experiment.samples[i] for i in range(nsamples)]
          for i in range(nsamples):
                if paramarr[i]==param:
                    return i
          return None
      def paramindex(param,basestr):
          nsamples=len(self.experiment.b)
          npairs=int(nsamples*(nsamples+1)/2)
          paramarr=[]
          for i in range(npairs):
              s1,s2=self.index2sample(i)
              paramarr.append(basestr+self.experiment.samples[s1]+self.experiment.samples[s2])
              if paramarr[i]==param:
                  return i
          return None
      if paramindex(param,'N')==self.sample2index(X,Y): return np.ones(self.k.shape)
      elif paramindex(param,'N') is not None and paramindex(param,'N')!=self.sample2index(X,Y): return np.zeros(self.k.shape)
      if paramindex(param,'N2')==self.sample2index(X,Y): return self.k**2*self.mu**2
      elif paramindex(param,'N2') is not None and paramindex(param,'N2')!=self.sample2index(X,Y): return np.zeros(self.k.shape)
      if paramindex(param,'N4')==self.sample2index(X,Y): return self.k**4*self.mu**4
      elif paramindex(param,'N4') is not None and paramindex(param,'N4')!=self.sample2index(X,Y): return np.zeros(self.k.shape)
      
      default_step = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002,'A_log':0.002}
        
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01
                 
      def one_sided(param,step):  
         self.cosmo.set({param:step})
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         ap0_hi = self.cosmo.angular_distance(z)*self.cosmo.h() / self.Da_fid(z) 
         ap1_hi = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h() / self.Hz_fid(z)
         #
         self.cosmo.set({param:2.*step})
         self.cosmo.compute()
         P_dummy_hi2 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi2 = self.cosmo.angular_distance(z)*self.cosmo.h() / self.Da_fid(z) 
         ap1_hi2 = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h() / self.Hz_fid(z)
         #
         self.cosmo.set({param:3.*step})
         self.cosmo.compute()
         P_dummy_hi3 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi3 = self.cosmo.angular_distance(z)*self.cosmo.h() / self.Da_fid(z) 
         ap1_hi3 = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h() / self.Hz_fid(z)
         #
         self.cosmo.set({param:4.*step})
         self.cosmo.compute()
         P_dummy_hi4 = compute_tracer_power_spectrum(**kwargs)
         ap0_hi4 = self.cosmo.angular_distance(z)*self.cosmo.h() / self.Da_fid(z) 
         ap1_hi4 = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h() / self.Hz_fid(z)
         #
         self.cosmo = Class()
         self.cosmo.set(self.params_fid)
         self.cosmo.compute() 
         #
         dap0dp = (-3.*ap0_hi4 + 16.*ap0_hi3 - 36.*ap0_hi2 + 48.*ap0_hi - 25.*1)/(12.*step)
         dap1dp = (-3.*ap1_hi4 + 16.*ap1_hi3 - 36.*ap1_hi2 + 48.*ap1_hi - 25.*1)/(12.*step)
         result = (-3.*P_dummy_hi4 + 16.*P_dummy_hi3 - 36.*P_dummy_hi2 + 48.*P_dummy_hi - 25.*P_fid)/(12.*step)
         K,MU = self.k,self.mu
         if self.AP: result += (dap1dp-2*dap0dp)*P_fid + MU*(1-MU**2)*(dap1dp+dap0dp)*self.dPdmu(P_fid) +\
                                K*(dap1dp*MU**2 - dap0dp*(1-MU**2))*self.dPdk(P_fid)
         return result
      def set_param(value):
         self.cosmo.set({param : value})
         #self.params[param] = value
      sample1=X
      sample2=Y
      ba_fid=compute_b(self,z,sample1)
      bb_fid=compute_b(self,z,sample2)
      b_fid = 0.5*(ba_fid+bb_fid) 
      f_fid = self.cosmo_fid.scale_independent_growth_factor_f(z)  
      alpha0a_fid = self.experiment.alpha0[X](z)
      alpha0b_fid = self.experiment.alpha0[Y](z)
      alpha0_fid = (alpha0a_fid*bb_fid/ba_fid+alpha0b_fid*ba_fid/bb_fid)/2
#       alpha0_fid = self.experiment.alpha0[self.sample2index(sample1,sample2)](z)
      Hz = self.Hz_fid(z)
      N_fid = self.experiment.fover[self.sample2index(sample1,sample2)]/np.sqrt(compute_n(self,z,sample1)*compute_n(self,z,sample2))
      noise = self.experiment.fover[self.sample2index(sample1,sample2)]/np.sqrt(compute_n(self,z,sample1)*compute_n(self,z,sample2))
      # if self.experiment.HI: noise = castorinaPn(z)
      sigv = self.experiment.sigv
      N2_fid = -noise*((1+z)*sigv/Hz)**2.
      #N2_fid = self.experiment.N2[self.sample2index(sample1,sample2)]
      if self.experiment.bs is None: 
            bsa_fid =-2*(ba_fid-1)/7
            bsb_fid =-2*(bb_fid-1)/7
      else: 
            bsa_fid = self.experiment.bs[sample1](z)
            bsb_fid = self.experiment.bs[sample2](z)

      if kwargs is None: 
         kwargs = {'fishcast':self, 'Xind':sample1, 'Yind':sample2, 'z':z, 'ba':ba_fid, 'bb':bb_fid, 'b2a':self.experiment.b2[sample1](z),
                   'b2b':self.experiment.b2[sample2](z), 'bsa':bsa_fid,'bsb':bsb_fid,
                   'alpha0a':alpha0a_fid,'alpha0b':alpha0b_fid, 'alpha2a':0,'alpha2b':0, 'alpha4a':0.,'alpha4b':0., 
                   'alpha6':0, 'N':N_fid, 'N2':N2_fid, 'N4':0.,
                   'f':-1, 'A_lin':self.A_lin, 'omega_lin':self.omega_lin, 'phi_lin':self.phi_lin,'A_log':self.A_log, 
                   'omega_log':self.omega_log,'phi_log':self.phi_log,'kIR':0.2}

      # if self.experiment.HI: kwargs['N'] = noise # ignores thermal HI noise in deriavtives
      
      nsamples=len(self.experiment.b)
      bidx=None; b2idx=None; bsidx=None
      bidx = paramsample(param,'b')
      b2idx = paramsample(param,'b2')
      bsidx = paramsample(param,'bs')
      if bidx is not None:
          if bidx!=sample1 and bidx!=sample2: return np.zeros(self.k.shape)
          elif bidx==sample1 and bidx==sample2:
                del kwargs['ba']
                del kwargs['bb']
                kwargs['b'] = b_fid
                param = 'b'
          elif bidx==sample1: param='ba'
          elif bidx==sample2: param='bb'
          else: print('error b', X, Y, param)
      elif b2idx is not None:
          if b2idx!=sample1 and b2idx!=sample2: return np.zeros(self.k.shape)
          elif b2idx==sample1 and b2idx==sample2:
                del kwargs['b2a']
                del kwargs['b2b']
                kwargs['b2'] =self.experiment.b2[sample1](z)
                param = 'b2'
          elif b2idx==sample1: param='b2a'
          elif b2idx==sample2: param='b2b'
          else: print('error b2', X, Y, param)
      elif bsidx is not None:
          if bsidx!=sample1 and bsidx!=sample2: return np.zeros(self.k.shape)
          elif bsidx==sample1 and bsidx==sample2:
                del kwargs['bsa']
                del kwargs['bsb']
                if self.experiment.bs is not None: kwargs['bs'] = self.experiment.bs[sample1](z)
                else: kwargs['bs'] = -2*(b_fid-1)/7
                param = 'bs'
          elif bsidx==sample1: param='bsa'
          elif bsidx==sample2: param='bsb'
          else: print('error bs', X, Y, param)
      alpha0idx=paramsample(param,'alpha0')
      alpha2idx=paramsample(param,'alpha2')
      alpha4idx=paramsample(param,'alpha4')
      if alpha0idx is not None:
          if alpha0idx!=sample1 and alpha0idx!=sample2: return np.zeros(self.k.shape)
          elif alpha0idx==sample1 and alpha0idx==sample2:
                del kwargs['alpha0a']
                del kwargs['alpha0b']
                kwargs['alpha0'] = alpha0_fid
                param = 'alpha0'
          elif alpha0idx==sample1: param='alpha0a'
          elif alpha0idx==sample2: param='alpha0b'
          else: print('error alpha0', X, Y, param)
      if alpha2idx is not None:
          if alpha2idx!=sample1 and alpha2idx!=sample2: return np.zeros(self.k.shape)
          elif alpha2idx==sample1 and alpha2idx==sample2:
                del kwargs['alpha2a']
                del kwargs['alpha2b']
                kwargs['alpha2'] = 0#alpha2_fid
                param = 'alpha2'
          elif alpha2idx==sample1: param='alpha2a'
          elif alpha2idx==sample2: param='alpha2b'
          else: print('error alpha2', X, Y, param)
      if alpha4idx is not None:
          if alpha4idx!=sample1 and alpha4idx!=sample2: return np.zeros(self.k.shape)
          elif alpha4idx==sample1 and alpha4idx==sample2:
                del kwargs['alpha4a']
                del kwargs['alpha4b']
                kwargs['alpha4'] = 0#alpha4_fid
                param = 'alpha4'
          elif alpha4idx==sample1: param='alpha4a'
          elif alpha4idx==sample2: param='alpha4b'
          else: print('error alpha4', X, Y, param)
      if param in kwargs or param == 'f_NL': 
         fNL_flag = False
         if param == 'f_NL': 
            param = 'b' ; fNL_flag = True
         if param == 'f': default_value = f_fid
         else: default_value = kwargs[param]
         #
         up = default_value*(1+relative_step)
         if default_value == 0: up = absolute_step
         down = default_value*(1-relative_step)
         if default_value == 0: down = -absolute_step
         upup = default_value*(1+2*relative_step)
         if default_value == 0: upup = 2*absolute_step
         downdown = default_value*(1-2*relative_step)
         if default_value == 0: downdown = -2*absolute_step
         step = up - default_value
         kwargs[param] = up
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = down
         P_dummy_low = compute_tracer_power_spectrum(**kwargs)
         if five_point:
            kwargs[param] = upup
            P_dummy_higher = compute_tracer_power_spectrum(**kwargs)
            kwargs[param] = downdown
            P_dummy_lower = compute_tracer_power_spectrum(**kwargs)
         kwargs[param] = default_value
         if five_point: dPdtheta = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
         else: dPdtheta = (P_dummy_hi - P_dummy_low) / (2. * step)
         if fNL_flag:
            D = 0.76 * self.cosmo.scale_independent_growth_factor(z) # normalized so D(a) = a in the MD era
            # brute force way of getting the transfer function, normalized to 1 at kmin
            pmatter = compute_matter_power_spectrum(self, z, linear=True)
            T = np.sqrt(pmatter/self.k**self.params['n_s'])
            T /= T[0]
            fNL_factor = 3.*1.68*(b_fid-1.)*((self.params['omega_cdm']+self.params['omega_b'])/\
                                              self.cosmo.h()**2.)*100.**2.
            fNL_factor /= D * self.k**2. * T * 299792.458**2.
            dPdtheta *= fNL_factor
         return dPdtheta
 
      if param == 'RT':
         K,MU = self.k,self.mu
         # h = self.params['h']
         h = self.cosmo.h()
         plin = np.array([self.cosmo_fid.pk_cb_lin(k*h,z)*h**3. for k in K]) 
         if X!=0 and Y!=0: return np.zeros(len(plin))
         if X==0 and X==Y: return plin*MU**2*(2*ba_fid+2*f_fid*MU**2)
         if X==0 and Y!=X: return plin*MU**2*(bb_fid+f_fid*MU**2)
         if Y==0 and Y!=X: return plin*MU**2*(ba_fid+f_fid*MU**2)
         print('error') 

      P_fid = compute_tracer_power_spectrum(**kwargs) 

      if param == 'norm': return 2*P_fid  

      if param == 'Tb': 
         Ez = self.cosmo.Hubble(z)/self.cosmo.Hubble(0)
         Ohi = 4e-4*(1+z)**0.6
         Tb = 188e-3*(self.cosmo.h())/Ez*Ohi*(1+z)**2
         return 2. * ( P_fid - noise  + castorinaPn(z)) / Tb  

      if param == 'alpha_parallel':
         K,MU = self.k,self.mu
         return -P_fid - MU*(1-MU**2)*self.dPdmu(P_fid) - K*MU**2*self.dPdk(P_fid)

      if param == 'alpha_perp':
         K,MU = self.k,self.mu
         return -2*P_fid + MU*(1-MU**2)*self.dPdmu(P_fid) - K*(1-MU**2)*self.dPdk(P_fid)       
            
      # derivative of early dark energy parameters (Hill+2020)
      if param == 'fEDE' and self.fEDE == 0.:
         EDE_params = {'log10z_c': self.log10z_c,'fEDE': absolute_step,'thetai_scf': self.thetai_scf,
                       'Omega_Lambda':0.0,'Omega_fld':0,'Omega_scf':-1,
                       'n_scf':3,'CC_scf':1,'scf_tuning_index':3,
                       'scf_parameters':'1, 1, 1, 1, 1, 0.0',
                       'attractor_ic_scf':'no'}
         self.cosmo.set(EDE_params)
         return one_sided('fEDE',absolute_step)

      if param == 'xi_idr' and self.xi_idr == 0.: 
         idr_params = {'xi_idr':0.,'a_idm_dr':1e4,'f_idm_dr':1.}
         self.cosmo.set(idr_params)
         return one_sided('xi_idr',absolute_step)

      if (param == 'log10z_c' or param == 'thetai_scf') and self.fEDE == 0.:
         print('Attempted to marginalize over log10z_c or thetai_scf when fEDE has a fiducial value of 0.')
         return

      result = np.zeros(len(self.k))

      # brute force numerical differentiation
      flag = False 
      if param == 'log(A_s)' : 
         flag = True
         param = 'A_s'  

      default_value = self.params[param] 

      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         # CLASS takes a string as an input when there is more than one massless neutrino
         default_value_float = np.array(list(map(float,list(default_value.split(',')))))
         Mnu = sum(default_value_float)
         up = ','.join(list(map(str,list(default_value_float+relative_step*Mnu/self.params['N_ncdm']))))
         upup = ','.join(list(map(str,list(default_value_float+2.*relative_step*Mnu/self.params['N_ncdm']))))
         down = ','.join(list(map(str,list(default_value_float-relative_step*Mnu/self.params['N_ncdm']))))
         downdown = ','.join(list(map(str,list(default_value_float-2.*relative_step*Mnu/self.params['N_ncdm']))))
         step = Mnu*relative_step
      else:
         up = default_value * (1. + relative_step)
         if default_value == 0.: up = default_value + absolute_step
         upup = default_value * (1. + 2.*relative_step)
         if default_value == 0.: upup = default_value + 2.*absolute_step
         down = default_value * (1. - relative_step)
         if default_value == 0.: down = default_value - absolute_step
         downdown = default_value * (1. - 2.*relative_step)
         if default_value == 0.: downdown = default_value - 2.*absolute_step
         step = default_value * relative_step
         if default_value == 0.: step = absolute_step

      if five_point:
         set_param(up)
         self.cosmo.compute()
         P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
         aperp_hi = self.cosmo.angular_distance(z)*self.cosmo.h() / self.Da_fid(z) 
         apar_hi = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h())
         #
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_tracer_power_spectrum(**kwargs)
         aperp_higher = self.cosmo.angular_distance(z)*self.cosmo.h()/ self.Da_fid(z) 
         apar_higher = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h())
         #
         set_param(down)
         self.cosmo.compute()
         P_dummy_low = compute_tracer_power_spectrum(**kwargs)
         aperp_low = self.cosmo.angular_distance(z)*self.cosmo.h()/ self.Da_fid(z) 
         apar_low = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h())
         #
         set_param(downdown)
         self.cosmo.compute()
         P_dummy_lower = compute_tracer_power_spectrum(**kwargs)
         aperp_lower = self.cosmo.angular_distance(z)*self.cosmo.h()/ self.Da_fid(z) 
         apar_lower = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h())
         #
         set_param(default_value)
         self.cosmo.compute()
         #
         result += (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
         daperpdp = (-aperp_higher + 8.*aperp_hi - 8.*aperp_low + aperp_lower) / (12. * step)
         dapardp = (-apar_higher + 8.*apar_hi - 8.*apar_low + apar_lower) / (12. * step)
         K,MU = self.k,self.mu
         if self.AP: result += -(dapardp+2*daperpdp)*P_fid - MU*(1-MU**2)*(dapardp-daperpdp)*self.dPdmu(P_fid) -\
                               K*(dapardp*MU**2 + daperpdp*(1-MU**2))*self.dPdk(P_fid)
         if flag: result *= self.params['A_s']
         return result

      # defaults to a two sided derivative
      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_tracer_power_spectrum(**kwargs)
      aperp_hi = self.cosmo.angular_distance(z)*self.cosmo.h()/ self.Da_fid(z) 
      apar_hi = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h()) 
      #
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_tracer_power_spectrum(**kwargs)
      aperp_low = self.cosmo.angular_distance(z)*self.cosmo.h()/ self.Da_fid(z) 
      apar_low = self.Hz_fid(z)/(self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h())
      #
      set_param(default_value)
      self.cosmo.compute()
      result += (P_dummy_hi - P_dummy_low) / (2. * step)
      daperpdp = (aperp_hi - aperp_low) / (2. * step)
      dapardp = (apar_hi - apar_low) / (2. * step)
      K,MU = self.k,self.mu
      if self.AP: result += -(dapardp+2*daperpdp)*P_fid - MU*(1-MU**2)*(dapardp-daperpdp)*self.dPdmu(P_fid) -\
                               K*(dapardp*MU**2 + daperpdp*(1-MU**2))*self.dPdk(P_fid)
      if flag: result *= self.params['A_s']
      #if param=='sigma8': result*=self.params['sigma8']
      return result


   def compute_dCdp(self, param, X, Y, zmin=None, zmax=None, 
                    relative_step=-1., absolute_step=-1., five_point=False):
      '''
      '''
      def paramsample(param,basestr):
          nsamples=len(self.experiment.b)
          paramarr = [basestr+self.experiment.samples[i] for i in range(nsamples)]
          for i in range(nsamples):
                if paramarr[i]==param:
                    return i
          return None
      def paramindex(param,basestr):
          nsamples=len(self.experiment.b)
          npairs=int(nsamples*(nsamples+1)/2)
          paramarr=[]
          for i in range(npairs):
              s1,s2=self.index2sample(i)
              paramarr.append(basestr+self.experiment.samples[s1]+self.experiment.samples[s2])
              if paramarr[i]==param:
                  return i
          return None
      if paramindex(param,'N') is not None and (X=='k' or Y=='k' or paramindex(param,'N')!=self.sample2index(X,Y)): 
            return np.zeros(self.ell.shape)
      elif paramindex(param,'N') is not None:
        #return np.ones(self.ell.shape)
        param='N'
      default_step = {'tau_reio':0.3,'m_ncdm':0.05,'A_lin':0.002,'A_log':0.002}
      if relative_step == -1.: 
         try: relative_step = default_step[param]
         except: relative_step = 0.01
      if absolute_step == -1.: 
         try: absolute_step = default_step[param]
         except: absolute_step = 0.01
      
      if zmin is not None and zmax is not None: zmid = (zmin+zmax)/2
      else: zmid = self.experiment.zcenters[0] # Ckk, where b and stuff don't matter
      ba_fid=compute_b(self,zmid,X)
      bb_fid=compute_b(self,zmid,Y)
      alpha0a_fid = self.experiment.alpha0[X](zmid) if X!='k' else self.experiment.alphak(zmid)
      alpha0b_fid = self.experiment.alpha0[Y](zmid) if Y!='k' else self.experiment.alphak(zmid)
      alpha0_fid = (alpha0a_fid*bb_fid/ba_fid+alpha0b_fid*ba_fid/bb_fid)/2
      if X!='k' and Y!='k' and X==Y: 
          noise = self.experiment.fover[self.sample2index(X,Y)]/np.sqrt(compute_n(self,zmid,X)*compute_n(self,zmid,Y))
      else: noise = 0
      # if self.experiment.HI: noise = castorinaPn(zmid)
      b2a_fid = self.experiment.b2[X](zmid) if X!='k' else 8*(ba_fid-1)/21
      b2b_fid = self.experiment.b2[Y](zmid) if Y!='k' else 8*(bb_fid-1)/21
          
      kwargs = {'fishcast':self, 'X':X, 'Y':Y, 'zmin':zmin, 'zmax':zmax,
                'zmid':zmid,'gamma':1, 'ba':ba_fid, 'bb':bb_fid, 'b2a':b2a_fid,
                'b2b':b2b_fid, 'bsa':-2*(ba_fid-1)/7,'bsb':-2*(bb_fid-1)/7, 
                'alpha0a':alpha0a_fid,'alpha0b':alpha0b_fid,'N':noise}

      bidx = paramsample(param,'b')
      b2idx = paramsample(param,'b2')
      bsidx = paramsample(param,'bs')
      if bidx is not None:
          if bidx!=X and bidx!=Y: return np.zeros(self.ell.shape)
          elif (bidx==X and bidx==Y):
                del kwargs['ba']
                del kwargs['bb']
                kwargs['b'] = ba_fid
                param = 'b'
          elif bidx==X: param='ba'
          elif bidx==Y: param='bb'
      elif b2idx is not None:
          if b2idx!=X and b2idx!=Y: return np.zeros(self.ell.shape)
          elif (b2idx==X and b2idx==Y):
                del kwargs['b2a']
                del kwargs['b2b']
                kwargs['b2'] = b2a_fid
                param = 'b2'
          elif b2idx==X: param='b2a'
          elif b2idx==Y: param='b2b'
      elif bsidx is not None:
          if bsidx!=X and bsidx!=Y: return np.zeros(self.ell.shape)
          elif (bsidx==X and bsidx==Y):
                del kwargs['bsa']
                del kwargs['bsb']
                kwargs['bs'] = -2*(ba_fid-1)/7 #this is ok because there is only one sample, ie b_fid = 0.5(ba+bb)=ba
                param = 'bs'
          elif bsidx==X: param='bsa'
          elif bsidx==Y: param='bsb'
      alpha0idx=paramsample(param,'alpha0')
      if alpha0idx is not None:
          if alpha0idx!=X and alpha0idx!=Y: return np.zeros(self.ell.shape)
          elif (alpha0idx==X and alpha0idx==Y):
                del kwargs['alpha0a']
                del kwargs['alpha0b']
                kwargs['alpha0'] = alpha0_fid 
                param = 'alpha0'
          elif alpha0idx==X: param='alpha0a'
          elif alpha0idx==Y: param='alpha0b'
      if param=='alpha0k':
          alpha0idx ='k'
          if alpha0idx!=X and alpha0idx!=Y: return np.zeros(self.ell.shape)
          elif (alpha0idx==X and alpha0idx==Y):
                return np.zeros(self.ell.shape)
          elif alpha0idx==X: param='alpha0a'
          elif alpha0idx==Y: param='alpha0b'
        
      if param in kwargs: 
         default_value = kwargs[param]
         up = default_value*(1+relative_step)
         if default_value == 0: up = absolute_step
         down = default_value*(1-relative_step)
         if default_value == 0: down = -absolute_step
         upup = default_value*(1+2*relative_step)
         if default_value == 0: upup = 2*absolute_step
         downdown = default_value*(1-2*relative_step)
         if default_value == 0: downdown = -2*absolute_step
         step = up - default_value
         kwargs[param] = up
         P_dummy_hi = compute_lensing_Cell(**kwargs)
         kwargs[param] = down
         P_dummy_low = compute_lensing_Cell(**kwargs)
         if five_point:
            kwargs[param] = upup
            P_dummy_higher = compute_lensing_Cell(**kwargs)
            kwargs[param] = downdown
            P_dummy_lower = compute_lensing_Cell(**kwargs)
         kwargs[param] = default_value
         if five_point: return (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. *  step) 
         return (P_dummy_hi-P_dummy_low) / (2. * step)
       
      P_fid = compute_lensing_Cell(**kwargs)
              
      # brute force numerical differentiation
      flag = False 
      if param == 'log(A_s)': 
         flag = True
         param = 'A_s'  
        
      default_value = self.params_fid[param] 
        
      if param == 'm_ncdm' and self.params['N_ncdm']>1:
         # CLASS takes a string as an input when there is more than one massless neutrino
         default_value_float = np.array(list(map(float,list(default_value.split(',')))))
         Mnu = sum(default_value_float)
         up = ','.join(list(map(str,list(default_value_float+relative_step*Mnu/self.params['N_ncdm']))))
         upup = ','.join(list(map(str,list(default_value_float+2.*relative_step*Mnu/self.params['N_ncdm']))))
         down = ','.join(list(map(str,list(default_value_float-relative_step*Mnu/self.params['N_ncdm']))))
         downdown = ','.join(list(map(str,list(default_value_float-2.*relative_step*Mnu/self.params['N_ncdm']))))
         step = Mnu*relative_step
      else:
         up = default_value * (1. + relative_step)
         if default_value == 0.: up = default_value + absolute_step
         upup = default_value * (1. + 2.*relative_step)
         if default_value == 0.: upup = default_value + 2.*absolute_step
         down = default_value * (1. - relative_step)
         if default_value == 0.: down = default_value - absolute_step
         downdown = default_value * (1. - 2.*relative_step)
         if default_value == 0.: downdown = default_value - 2.*absolute_step
         step = (up-down)/2
         if default_value == 0.: step = absolute_step
            
      def set_param(value):
         self.cosmo.set({param : value})
         #self.params[param] = value

      set_param(up)
      self.cosmo.compute()
      P_dummy_hi = compute_lensing_Cell(**kwargs)
      set_param(down)
      self.cosmo.compute()
      P_dummy_low = compute_lensing_Cell(**kwargs)
      if five_point:
         set_param(upup)
         self.cosmo.compute()
         P_dummy_higher = compute_lensing_Cell(**kwargs)
         set_param(downdown)
         self.cosmo.compute()
         P_dummy_lower = compute_lensing_Cell(**kwargs)
      set_param(default_value)
      self.cosmo.compute()
      if five_point: result = (-P_dummy_higher + 8.*P_dummy_hi - 8.*P_dummy_low + P_dummy_lower) / (12. * step)
      else: result = (P_dummy_hi-P_dummy_low)/(2.*step)
      if flag: result *= self.params['A_s']
      return result

   
   def Sigma2(self, z): return sum(compute_matter_power_spectrum(self,z,linear=True)*self.dk*self.dmu) / (6.*np.pi**2.)

    
   def kmax_constraint(self, z, kmax_knl=1.): return self.k < kmax_knl/np.sqrt(self.Sigma2(z))
    
    
   def compute_wedge(self, z, kmin=0.003):
      '''
      Returns the foreground wedge. If not an HI experiment, just returns a kmin constraint.
      The object that is returned is an array of bools of length Nk*Nmu.
      '''
      return self.k > kmin
      # if not self.experiment.HI: return self.k > kmin
      # #
      # kparallel = self.k*self.mu
      # kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
      # #
      # chi = (1.+z)*self.cosmo.angular_distance(z)*self.cosmo.h() # Mpc/h
      # Hz = self.cosmo.Hubble(z)*(299792.458)/self.cosmo.h() # h km / s Mpc
      # c = 299792.458 # km/s
      # lambda21 = 0.21 * (1+z) # meters
      # D_eff = self.experiment.D * np.sqrt(0.7) # effective dish diameter, in meters
      # theta_w = self.experiment.N_w * 1.22 * lambda21 / (2.*D_eff)
      # #
      # wedge = kparallel > chi*Hz*np.sin(theta_w)*kperpendicular/(c*(1.+z))
      # kparallel_constraint = kparallel > self.experiment.kparallel_min
      # return wedge*kparallel_constraint
    

    
   def compute_derivatives(self, five_point=True, parameters=None, z=None, overwrite=False):
      '''
      Calculates all the derivatives and saves them to the 
      output/forecast name/derivatives directory
      '''
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples+1)/2)
      
      if parameters is not None: 
         for i,p in enumerate(parameters):
            for j in range(npairs):
                if p == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(round(100*z[i]))+'.txt'
                elif p == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(round(100*z[i]))+'.txt'
                elif p == 'A_log': filename = 'A_log_'+str(int(100.*self.omega_log))+'_'+str(round(100*z[i]))+'.txt'
                else: filename = p+'_'+str(round(100*z[i]))+'.txt'
                folder = '/derivatives/'
                if self.recon: folder = '/derivatives_recon/'
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename='P'+name1+name2+'_'+filename
                fname = self.basedir+'output/'+self.name+folder+filename
                if not exists(fname) or overwrite: 
                   dPdp = self.compute_dPdp(param=p,X=s1,Y=s2,z=z[i], five_point=five_point)
                   np.savetxt(fname,dPdp)
                else:
                   continue
         return
      self.free_params=self.get_listparams(list(self.free_params))
      zs = self.experiment.zcenters
      for z in zs:
         for free_param in self.free_params:
            for j in range(npairs):
                if free_param == 'fEDE': filename = 'fEDE_'+str(int(1000.*self.log10z_c))+'_'+str(round(100*z))+'.txt'
                elif free_param == 'A_lin': filename = 'A_lin_'+str(int(100.*self.omega_lin))+'_'+str(round(100*z))+'.txt'
                elif free_param == 'A_log': filename = 'A_log_'+str(int(100.*self.omega_log))+'_'+str(round(100*z))+'.txt'
                else: filename = free_param+'_'+str(round(100*z))+'.txt'
                folder = '/derivatives/'
                if self.recon: folder = '/derivatives_recon/'
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename='P'+name1+name2+'_'+filename
                fname = self.basedir+'output/'+self.name+folder+filename
                if not exists(fname) or overwrite:
                   dPdp = self.compute_dPdp(param=free_param,X=s1,Y=s2, z=z, five_point=five_point)
                   np.savetxt(fname,dPdp)
                else:
                   continue
       
    
   def compute_Cl_derivatives(self, five_point=True, overwrite=False):
      '''
      Calculates the derivatives of Ckk, Ckg, and Cgg with respect to 
      each of the free_params. 
      '''
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples+1)/2)
      zs = self.experiment.zedges
      self.free_params=self.get_listparams(list(self.free_params))
      for free_param in self.free_params:
         #
         if free_param != 'gamma':
            filename = 'Ckk_'+free_param+'.txt'
            fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
            if not exists(fname) or overwrite: 
               dCdp = self.compute_dCdp(free_param, 'k', 'k', five_point=five_point)
               np.savetxt(fname,dCdp)
            #else:
            #   continue
         else:
            for i,z in enumerate(zs[:-1]):
               filename = 'Ckk_'+free_param+'_'+str(round(100*zs[i]))+'_'+str(round(100*zs[i+1]))+'.txt'
               fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
               if not exists(fname) or overwrite: 
                  dCdp = self.compute_dCdp(free_param, 'k', 'k',zmin=zs[i],zmax=zs[i+1], five_point=five_point)
                  np.savetxt(fname,dCdp)
               else:
                  continue
         #
         for i,z in enumerate(zs[:-1]):
            for j in range(npairs):
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename = 'C'+name1+name2+'_'+free_param+'_'+str(round(100*zs[i]))+'_'+str(round(100*zs[i+1]))+'.txt'
                fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
                if not exists(fname) or overwrite: 
                   dCdp = self.compute_dCdp(free_param, s1, s2, zmin=zs[i], zmax=zs[i+1], five_point=five_point)
                   np.savetxt(fname,dCdp) 
                #else:
                #   continue
                if j<nsamples:
                    name1=self.experiment.samples[j]
                    filename = 'Ck'+name1+'_'+free_param+'_'+str(round(100*zs[i]))+'_'+str(round(100*zs[i+1]))+'.txt'
                    fname = self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename
                    if not exists(fname) or overwrite: 
                       dCdp = self.compute_dCdp(free_param, 'k', j, zmin=zs[i], zmax=zs[i+1], five_point=five_point)
                       np.savetxt(fname,dCdp)
                    #else:
                    #   continue
            
            
   def check_derivatives(self):
      '''
      Plots all the derivatives in the output/forecast name/derivatives directory
      '''
      directory = self.basedir+'output/'+self.name+'/derivatives'
      for root, dirs, files in os.walk(directory, topdown=False):
         for file in files:
            filename = os.path.join(directory, file)
            dPdp = np.genfromtxt(filename)
            plt.figure(figsize=(6,6))
            k = np.linspace(0.008,1.,1000)
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.)(k),color='b')
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.3)(k),color='g')
            plt.semilogx(k,self.get_f_at_fixed_mu(dPdp,0.7)(k),color='r')
            plt.xlabel(r'$k$ [h/Mpc]')
            plt.show()
            plt.clf()
            print(file)
            
            
   def load_derivatives(self, basis, log10z_c=-1.,omega_lin=-1,omega_log=-1,polys=True,auto_only=False,return_auto=False):
      '''
      Let basis = [p1, p2, ...], and denote the centers of the
      redshift bins by z1, z2, ... This returns a matrix of the
      form:
      
      derivatives = [[dPdp1, dPdp2, ...],  (z=z1)
                     [dPdp1, dPdp2, ...],  (z=z2)
                     ...]
                     
      Where dPdpi is the derivative with respect to the ith basis
      parameter (an array of length Nk*Nmu). 
      '''
      if return_auto: auto_only=True
      if log10z_c == -1. : log10z_c = self.log10z_c  
      if omega_lin == -1. : omega_lin = self.omega_lin
      if omega_log == -1. : omega_log = self.omega_log
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples-1)/2+nsamples)
      nbins = self.experiment.nbins
      folder = '/derivatives/'
      if self.recon: folder = '/derivatives_recon/'
      directory = self.basedir+'output/'+self.name+folder
      basis=self.get_listparams(list(basis),auto_only=auto_only)
      
      N = len(basis)
      if self.recon and polys: N += 15*npairs
      derivatives = np.zeros((npairs,nbins,N,self.Nk*self.Nmu))
      for zbin_index in range(nbins):
         z = self.experiment.zcenters[zbin_index]
         for i,param in enumerate(basis):
            for j in range(npairs):
                if param == 'fEDE': filename = 'fEDE_'+str(int(1000.*log10z_c))+'_'+str(round(100*z))+'.txt'
                elif param == 'A_lin': filename = 'A_lin_'+str(int(100.*omega_lin))+'_'+str(round(100*z))+'.txt'
                elif param == 'A_log': filename = 'A_log_'+str(int(100.*omega_log))+'_'+str(round(100*z))+'.txt'
                else: filename = param+'_'+str(round(100*z))+'.txt'
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename='P'+name1+name2+'_'+filename
                if param!='RT':
                    try:
                       dPdp = np.genfromtxt(directory+filename)
                    except:
                       print(directory+filename)
                       print('Have not calculated derivative of ' + param)
                else: dPdp = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives/'+filename)

                derivatives[j,zbin_index,i,:] = dPdp
         if self.recon and polys:
            for m in range(15):
                for j in range(npairs):
                    derivatives[j,zbin_index,npairs*m+j+len(basis)] = self.mu**(2*(m//5)) * self.k**(m%5)
      if return_auto: 
        idx=[]
        for j in range(npairs):
            s1,s2=self.index2sample(j)
            if s1!=s2:idx+=[j+0]
        derivatives=np.delete(derivatives,idx,axis=0)
      return derivatives
            

   def shuffle_fisher(self,F,globe,Nz=None):
      N = len(F)
      if Nz is None: Nz = self.experiment.nbins
      loc = (N-globe)//Nz
      result = np.zeros(F.shape)
      mapping = {}
      for i in range(N):
         if i<globe: 
            mapping[i] = i
         else: 
            z = i-globe 
            y = globe + (z%Nz)*loc + z//Nz
            mapping[i] = y
      for i in range(N):
         for j in range(N):
            result[i,j] = F[mapping[i],mapping[j]]
      return result


   def combine_fishers(self,Fs,globe):
      N = len(Fs)
      result = Fs[0]
      for i in range(1,N): result = self.combine_2fishers(result,Fs[i],globe)
      return result


   def combine_2fishers(self,F1,F2,globe):
      '''
      helper function for combine_fishers
      '''
      N1 = int(len(F1)-globe)
      N2 = int(len(F2)-globe)
      N = N1+N2
      F = np.zeros((N+globe,N+globe))
      for i in range(N+globe):
         for j in range(N+globe):
            if i<globe+N1 and j<globe+N1: F[i,j] = F1[i,j]
            if i<globe and j<globe: F[i,j] = F1[i,j] + F2[i,j]
            if j>=globe+N1 and i<globe: F[i,j] = F2[i,j-N1]
            if i>=globe+N1 and j<globe: F[i,j] = F2[i-N1,j]
            if i>=globe+N1 and j>=globe+N1: F[i,j] = F2[i-N1,j-N1]
      return F


   def gen_fisher(self,basis,globe,log10z_c=-1.,omega_lin=-1.,omega_log=-1.,kmax_knl=1.,
                  kmin=0.003,kmax=-10.,kpar_min=-1.,mu_min=-1,derivatives=None,
                  zbins=None,polys=True,simpson=False,nratio=1.,auto_only=False,fskyratio=1.):
      '''
      Computes an array of Fisher matrices, one for each redshift bin.
      '''
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples+1)/2)
      if log10z_c == -1. : log10z_c = self.log10z_c
      if omega_lin == -1. : omega_lin = self.omega_lin
      if omega_log == -1. : omega_log = self.omega_log
      basis=self.get_listparams(list(basis),auto_only=auto_only)
        
      if derivatives is None: derivatives = self.load_derivatives(basis,log10z_c=log10z_c,
                                                                  omega_lin=omega_lin,omega_log=omega_log,polys=polys,auto_only=auto_only)   
      if zbins is None: zbins = range(self.experiment.nbins)

      autoidx=[]
      for i in range(npairs):
            s1,s2=self.index2sample(i)
            if auto_only:
                if s1==s2: autoidx+=[i]
            else: autoidx+=[i]
      def fish(zbin_index):
         n = len(basis)
         if self.recon: n += 15*npairs
         F = np.zeros((n,n))
         z = self.experiment.zcenters[zbin_index]
         dPdvecp = derivatives[autoidx,:,:][:,zbin_index,:,:]
         Cinv = np.linalg.inv(np.einsum('ijk->kij',compute_covariance_matrix(self,zbin_index,nratio=nratio,fskyratio=fskyratio))[:,autoidx,:][:,:,autoidx])
         constraints = self.compute_wedge(z,kmin=kmin)*self.kmax_constraint(z,kmax_knl)
         if kmax > 0: constraints = self.compute_wedge(z,kmin=kmin)*(self.k<kmax)
         constraints *= (self.k > kmin)
         kpar = self.k*self.mu
         kperp = self.k*np.sqrt(1-self.mu**2)
         if kpar_min > 0: constraints *= (kpar>kpar_min)
         if mu_min > 0: constraints *= (kpar > kperp*mu_min/np.sqrt(1-mu_min**2))
         F=np.einsum('c,aic,cab,bjc,ac,bc->ij',constraints,dPdvecp,Cinv,dPdvecp,self.kpar_cut[autoidx,:,:][:,zbin_index,:], self.kpar_cut[autoidx,:,:][:,zbin_index,:])
         return F

      fishers = [fish(zbin_index) for zbin_index in zbins]
      result = self.combine_fishers(fishers,globe)
      result = self.shuffle_fisher(result,globe,Nz=len(zbins))
      return result
  
   def load_lensing_derivatives(self,param,param_index,globe,zbin_index):
      '''
      Loads the derivative of (Ckk, Ckgi, Cgigi), i = 1 , 2, ..., Nz
      with respect to param.
      '''
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples-1)/2+nsamples)
      n = self.experiment.nbins
      zs = self.experiment.zedges
      result = np.zeros((npairs*n+nsamples*n+1,len(self.ell)))
    
      # if param is a local parameter (param_index >= globe)
      # then only take derivatives wrt C^{\kappa g_m} and 
      # C^{g_m g_m}, where m = zbin_index+1
      # alphax and alphaa are always assumed to be local
      if param_index >= globe:
         m = zbin_index+1
         for i in range(npairs):
             if i<nsamples:
                 name1=self.experiment.samples[i]
                 filename = 'Ck'+name1+'_'+param+'_'+str(round(100*zs[m-1]))+'_'+str(round(100*zs[m]))+'.txt'   
                 result[m+n*i] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
             s1,s2=self.index2sample(i)
             name1=self.experiment.samples[s1]
             name2=self.experiment.samples[s2]
             filename = 'C'+name1+name2+'_'+param+'_'+str(round(100*zs[m-1]))+'_'+str(round(100*zs[m]))+'.txt' 
             result[m+n*nsamples+i*n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
         return result
    
      if not 'gamma' in param:
         result[0] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/Ckk_'+param+'.txt')
      else:
         idx = int(param[-1])
         filename = 'Ckk_gamma_'+str(round(100*zs[idx-1]))+'_'+str(round(100*zs[idx]))+'.txt'
         result[0] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename) 
      for i in range(1,n+1):
         if 'gamma' in param: 
            idx = int(param[-1])
            for j in range(npairs):
                if j<nsamples:
                    name1=self.experiment.samples[j]
                    filename = 'Ck'+name1+'_gamma_'+str(round(100*zs[idx-1]))+'_'+str(round(100*zs[idx]))+'.txt'
                    result[idx+j*n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename = 'C'+name1+name2+'_gamma_'+str(round(100*zs[idx-1]))+'_'+str(round(100*zs[idx]))+'.txt'
                result[idx+n*nsamples+j*n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
         else:
            for j in range(npairs):
                if j<nsamples:
                    name1=self.experiment.samples[j]
                    filename = 'Ck'+name1+'_'+param+'_'+str(round(100*zs[i-1]))+'_'+str(round(100*zs[i]))+'.txt'
                    result[i+j*n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
                s1,s2=self.index2sample(j)
                name1=self.experiment.samples[s1]
                name2=self.experiment.samples[s2]
                filename = 'C'+name1+name2+'_'+param+'_'+str(round(100*zs[i-1]))+'_'+str(round(100*zs[i]))+'.txt'
                result[i+n*nsamples+j*n] = np.genfromtxt(self.basedir+'output/'+self.name+'/derivatives_Cl/'+filename)
      return result

    
   def gen_lensing_fisher(self,basis,globe,ell_min=30,ell_max=None,kmax_knl=1,
                          CMB='SO',kk=True,only_kk=False,bins=None,fsky_CMB=0.4,
                          fsky_intersect=None,auto_only=False,nratio=1.,fskyratio=1.,no_kg=False):
      '''
      '''
      
      nsamples=len(self.experiment.b)
      npairs=int(nsamples*(nsamples+1)/2)
              
      basis=self.get_listparams(list(basis),auto_only=auto_only)
      n = len(basis)
      C = covariance_Cls(self,kmax_knl=kmax_knl,CMB=CMB,fsky_CMB=fsky_CMB,fsky_intersect=fsky_intersect,nratio=nratio,fskyratio=fskyratio)
      Nz = self.experiment.nbins
      loc = n - globe
      N = globe + loc*Nz
      Np = 1 + Nz*(npairs+nsamples)
    
      if not isinstance(ell_min,list): ell_min = ell_min*np.ones(Np)
      if ell_max == None:
          chi = lambda z: (1.+z)*self.cosmo_fid.angular_distance(z)*self.cosmo.h()
          ell_max= np.zeros(Np)
          ell_max[0]=500 #baryonic feedback
          for i in range(Np-1):
                ell_max[1+i]=chi(self.experiment.zcenters[i%Nz])*kmax_knl/np.sqrt(self.Sigma2(self.experiment.zcenters[i%Nz]))
      elif not isinstance(ell_max,list): ell_max = ell_max*np.ones(Np)
      if bins is None: bins = np.arange(0,Nz,1)
      idx_bins = []
      if kk: idx_bins = idx_bins + [0]
      for binn in bins:
        for i in range(nsamples+npairs):
            if no_kg:
                if i>=nsamples: idx_bins = idx_bins + [binn+1+i*Nz] #don't include kg
            elif auto_only:
                if i<nsamples: idx_bins = idx_bins + [binn+1+i*Nz] #always include kg
                else:
                    s1,s2 = self.index2sample(int(i-nsamples))
                    if s1==s2:idx_bins = idx_bins + [binn+1+i*Nz] #include gagb if ga=gb
            elif not auto_only:idx_bins = idx_bins + [binn+1+i*Nz]
      result = np.zeros((N,N))
      derivs = np.zeros((N,Np,len(self.ell)))
      for i in range(globe): 
         derivs[i] = self.load_lensing_derivatives(basis[i],i,globe,0)
      for i in range(globe,N):
         param_index = globe + (i-globe)//Nz
         zbin_index = (i-globe)%Nz
         derivs[i] = self.load_lensing_derivatives(basis[param_index],param_index,globe,zbin_index)
      
      constraints = np.zeros((Np,len(self.ell)))
      for i in range(Np):
            constraints[i,:]=(self.ell>=ell_min[i])*(self.ell<ell_max[i])
      
      if only_kk: idx_bins=[0]
      Cinv = np.linalg.inv(np.einsum('ijk->kij',C[idx_bins,:,:][:,idx_bins,:]))
      result = np.einsum('ac,bc,iac,cab,jbc->ij',constraints[idx_bins,:],constraints[idx_bins,:],
                         derivs[:,idx_bins,:],Cinv,derivs[:,idx_bins,:])
      return result

        
   def get_f_at_fixed_mu(self,f,mu):
      '''
      For a function f(k,mu), which is represented as an array of length Nmu*Nk,
      return a function f(k)
      '''
      closest_index = np.where(self.mu >= mu)[0][0]
      indices = np.array([closest_index+n*self.Nmu for n in np.linspace(0,self.Nk-1,self.Nk)])
      f_fixed = [f[i] for i in indices.astype(int)]
      k = [self.k[i] for i in indices.astype(int)]
      f = interp1d(k,f_fixed,kind='linear',bounds_error=False, fill_value=0.)
      return f
    
    
   def Nmodes(self,zmin,zmax,nbins,kpar=-1.,kmax=-1,alpha0=-1,alpha2=0,linear=False,halofit=False):
    
      def G(z):
         Sigma2 = self.Sigma2(z)
         f = self.cosmo.scale_independent_growth_factor_f(z)
         kparallel = self.k*self.mu
         kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
         return np.exp(-0.5 * (kperpendicular**2. + kparallel**2. * (1.+f)**2.) * Sigma2)
      def I1(z):
         f = self.cosmo.scale_independent_growth_factor_f(z)      
         K,MU,b = self.k,self.mu,compute_b(self,z)
         P_L = compute_matter_power_spectrum(self,z,linear=True) * (b+f*MU**2.)**2.
         P_F = compute_tracer_power_spectrum(self,z,alpha0=alpha0,alpha2=alpha2)
         if linear: P_F = P_L + 1/compute_n(self,z)
         if halofit: P_F = compute_matter_power_spectrum(self,z) * (b+f*MU**2.)**2. + 1/compute_n(self,z)
         integrand = ( G(z)**2 * P_L / P_F )**2. 
         integrand *= self.compute_wedge(z) 
         if kpar > 0.: integrand *= (self.k*self.mu > kpar)
         if kmax > 0.: integrand *= (self.k < kmax)
         return sum(integrand * self.k**2. * self.dk * self.dmu / (2. * np.pi**2.))
         # we are dividing by 2 pi^2 (and not 4 pi^2) since we integrate from mu = 0 to 1
    
      zedges = np.linspace(zmin,zmax,nbins+1)
      zs = (zedges[1:]+zedges[:-1])/2.
      dV = np.array([self.comov_vol(zedges[i],zedges[i+1]) for i in range(nbins)])
      I = np.array([I1(z) for z in zs])
      return sum(I*dV) 
    
    
   def Nmodes_fixed_k(self,k,zmin,zmax,nbins,Deltak=0.1):
    
      def G(z):
         Sigma2 = self.Sigma2(z)
         f = self.cosmo.scale_independent_growth_factor_f(z)
         kparallel = self.k*self.mu
         kperpendicular = self.k*np.sqrt(1.-self.mu**2.)
         return np.exp(-0.5 * (kperpendicular**2. + kparallel**2. * (1.+f)**2.) * Sigma2)
      def I1(z):
         f = self.cosmo.scale_independent_growth_factor_f(z)      
         K,MU,b = self.k,self.mu,compute_b(self,z)
         P_L = compute_matter_power_spectrum(self,z,linear=True) * (b+f*MU**2.)**2.
         P_F = compute_tracer_power_spectrum(self,z)
         integrand = ( G(z)**2. * P_L / P_F )**2. 
         integrand *= self.k**2. * Deltak * self.dmu / (2. * np.pi**2.)
         # we are dividing by 2 pi^2 (and not 4 pi^2) since we integrate from mu = 0 to 1
         ks = self.k.reshape((self.Nk,self.Nmu))[:,0]
         integrand = integrand.reshape((self.Nk,self.Nmu))
         integrand = np.sum(integrand, axis=1)
         answer = interp1d(ks,integrand)    
         return answer(k)
    
      zedges = np.linspace(zmin,zmax,nbins+1)
      zs = (zedges[1:]+zedges[:-1])/2.
      dV = np.array([self.comov_vol(zedges[i],zedges[i+1]) for i in range(nbins)])
      I = np.array([I1(z) for z in zs])
      return sum(I*dV)
