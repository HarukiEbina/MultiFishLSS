from headers import *
import types
import copy
from scipy.integrate import quad
from scipy.optimize import fsolve
import json
from definitions import MFISHLSS_BASE

class experiment(object):
   '''
   An object that contains all the information related to the experiment
   '''
   def __init__(self, 
                zmin=0.8,               # Minimum redshift of survey
                zmax=1.2,               # Maximum redshift of survey
                nbins=1,                # Number of redshift bins
                zedges=None,            # Optional: Edges of redshift bins. Default is evenly-spaced bins.
                fsky=0.5,               # Fraction of sky observed
                sigma_z=0.0,            # Redshift error sz/(1+z)
                n=1e-3,                 # Galaxy number density, float (constant n) or function of z
                b=1.5,                  # Galaxy bias, float (constant b) or function of z
                b2=None,                # 
                bs=None,
                N2=None,
                alpha0=None,            #
                custom_n=False,         #
                custom_b=False,         #
                pesimistic=False,       # HI survey: specifies the k-wedge 
                Ndetectors=256**2.,     # HI survey: number of detectors
                fill_factor=0.5,        # HI survey: the array's fill factor 
                tint=5,                 # HI survey: oberving time [years]
                sigv=100,               # comoving velocity dispersion for FoG contribution [km/s]
                D = 6,
                HI_ideal=False,
                samples=None,
                fover=None):

      self.zmin = zmin
      self.zmax = zmax
      self.nbins = nbins
      self.zedges = np.linspace(zmin,zmax,nbins+1)
      if zedges is not None: 
         self.zedges = zedges
         self.nbins = len(zedges)-1
      self.zcenters = (self.zedges[1:]+self.zedges[:-1])/2.
      self.fsky = fsky
      self.sigma_z = sigma_z
      # If the number density is not a float, assumed to be a function of z
      if not isinstance(n, float): self.n = n
      else: self.n = lambda z: n + 0.*z
      # If the bias is not a float, assumed to be a function of z
      if not isinstance(b, float): self.b = b
      else: self.b = lambda z: b + 0.*z
      if not isinstance(n,list): self.n=[n]
      if not isinstance(b,list): self.b=[b]
      self.samples=samples
      if self.samples is None: self.samples = ['g']
      if len(self.samples)!=len(self.b) or len(self.b)!=len(self.n): print('invalid experiment due to sample number contradiction')
      nsample=len(self.b)
      npair=int(nsample*(nsample+1)/2)
      def index2sample(index):
          nsample=len(self.b)
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
      def find_b2_ST(z,i):
          '''
          Use Sheth-Tormen peak-background split to find b2 from b1
          '''
          bL1 = self.b[i](z)-1
          a=.707; p=.3; dc=1.686
          def get_ST_nu2(nu2,bL1):
              return a*nu2-1+2*p/(1+(a*nu2)**p)-dc*bL1
          nu2_ST = fsolve(get_ST_nu2, 5, args=(bL1))[0]
          bL2_ST = (a**2*nu2_ST**2-3*a*nu2_ST+2*p*(2*a*nu2_ST+2*p-1)/(1+(a*nu2_ST)**p))/dc**2
          return bL2_ST+8/21*bL1
      self.b2 = b2
      if self.b2 is None:
            self.b2=[]
            for i in range(nsample):
#                 self.b2.append(lambda z,i=i:8*(b[i](z)-1)/21)
                self.b2.append(lambda z,i=i:find_b2_ST(z,i))
      self.bs = bs
      '''if self.bs is None:
            self.bs=[]
            for i in range(nsample):
                self.bs.append(lambda z:-2*(self.b[i](z)-1)/7)'''
      self.alpha0 = alpha0
#       if self.alpha0 is None:
#             self.alpha0=[]
#             for i in range(npair):
#                 s1,s2=index2sample(i)
#                 self.alpha0.append(lambda z,s1=s1,s2=s2: 1.22 + 0.24*self.b[s1](z)*self.b[s2](z)*(z-5.96) if z<6 else 0.)  
      if self.alpha0 is None:
            self.alpha0=[]
            for i in range(nsample):
                self.alpha0.append(lambda z,i=i: 1.22 + 0.24*self.b[i](z)**2*(z-5.96) if z<6 else 0.)  
      if self.b2 is not None:
        if not isinstance(self.b2,list):self.b2=[self.b2]
        if len(self.b2)!=nsample: print('invalid experiment due to sample number contradiction') 
      if self.bs is not None:
        if not isinstance(self.bs,list):self.bs=[self.bs]
        if len(self.bs)!=nsample: print('invalid experiment due to sample number contradiction') 
      if self.alpha0 is not None:
        if len(self.alpha0)!=nsample: print('invalid experiment due to sample number contradiction') 
      self.fover=fover
      if self.fover is not None:
        if len(self.fover)!=npair: print('invalid experiment due to sample number contradiction')     
      if self.fover is None: 
        self.fover=np.ones(npair)
        for i in range(npair): 
          s1,s2=index2sample(i)
          if s1!=s2:
              self.fover[i]=0.
      for i in range(nsample):
          idx=0
          for j in range(i):
              idx+=(len(self.b)-i+1)
          if self.fover[idx]!=1:print('fover not unity for auto spectrum')
      self.N2=N2
      self.custom_n = custom_n
      self.custom_b = custom_b
      self.Ndetectors = Ndetectors
      self.fill_factor = fill_factor
      self.tint = tint
      self.sigv = sigv
      self.D = D  
      if pesimistic: 
         self.N_w = 3.
         self.kparallel_min = 0.1
      else: 
         self.N_w = 1.
         self.kparallel_min = 0.01
      with open(os.path.join(MFISHLSS_BASE, "input/alpha0k_fit.json"), "r") as read_file:
         alphak_fit = json.load(read_file)

      self.alphak = lambda z: float(interp1d(alphak_fit['z'], alphak_fit['alpha0'],kind='linear',bounds_error=False,fill_value='extrapolate')(z))
      