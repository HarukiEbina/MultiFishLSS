from headers import *
from velocileptors.LPT.cleft_fftw import CLEFT
from bao_recon.zeldovich_rsd_recon_fftw import Zeldovich_Recon
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.signal import savgol_filter
from scipy.integrate import simpson as simps
from math import ceil
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.special import legendre


from bao_recon.wiggle_split_recon import WiggleSplit_Recon


def compute_b(fishcast,z,X=0):
   '''
   Quick way of getting the bias. This is what 
   FishLSS always calls to get the bias from
   a forecast.
   '''
   exp = fishcast.experiment
   custom = exp.custom_b
   #treatment of k ignores gamma
   if X=='k': return 1
   return exp.b[X](z)

def compute_n(fishcast, z,samplenumber=0):
   '''
   Returns the relevant number density h^3/Mpc^3. For HI surveys
   returns an array of length Nk*Nmu, for all other surveys
   return a float.
   '''
   return fishcast.experiment.n[samplenumber](z)
   #return fishcast.experiment.n(z)

def recon_efficiency(fishcast, z, n, ba, bb, f, k0=0.14, mu0=0.6):
   """
   Computes reconstruction efficiency according to 
   "DESI and other Dark Energy experiments in the era of neutrino mass measurements"
   By interpolating a table across x = n*P_g(k0,mu0)/0.1734.
   """
   h = fishcast.cosmo.h()

   Pg_k0 = fishcast.cosmo.pk_cb_lin(k0*h,z)*h**3.
   Pg_k0_mu0 = (ba + f * mu0**2. ) * (bb + f * mu0**2.) * Pg_k0 

   x = n * Pg_k0_mu0/0.1734 

   if x > 10.:
      return 0.5 
   
   elif x < 0.2:
      return 1.

   else:
      rs = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.5])
      xs = np.array([0.2,0.3,0.5,1.,2.,3.,6.,10.])

      r = interp1d(xs,rs,kind='linear',bounds_error=False, fill_value=(1.0, 0.5))

      return r(x)
   
def compute_biaspoly(bvec1,bvec2,f=-1,fishcast=None):
    '''
    bvec=(b1,b2,bs,b3 ...,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4), biases are Lagrangian
    '''
    #not sure what to do for alpha and sn terms since I need new parameters anyways
    if f==-1 and fishcast == None: raise Exception('Must provide f or forecast object')
    if f == -1: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
    b1a,b2a,bsa,b3a,alpha0a,alpha2a,alpha4a,alpha6a,sna,sn2a,sn4a=bvec1
    b1b,b2b,bsb,b3b,alpha0b,alpha2b,alpha4b,alpha6b,snb,sn2b,sn4b=bvec2
    bE1a = b1a+1
    bE1b = b1b+1
    return np.array([1,(b1a+b1b)/2,b1a*b1b,(b2a+b2b)/2,(b1a*b2b+b1b*b2a)/2,b2a*b2b,(bsa+bsb)/2,(b1a*bsb+b1b*bsa)/2,
                     (b2a*bsb+b2b*bsa)/2,bsa*bsb,(b3a+b3b)/2,(b1a*b3b+b1b*b3a)/2, 
                     (alpha0a*bE1b/bE1a+alpha0b*bE1a/bE1b)/2,
                     (alpha2a*bE1b/bE1a+alpha2b*bE1a/bE1b+alpha0a*f/bE1a*(1-bE1b/bE1a)+alpha0b*f/bE1b*(1-bE1a/bE1b))/2,
                     (alpha4a*bE1b/bE1a+alpha4b*bE1a/bE1b+alpha2a*f/bE1a*(1-bE1b/bE1a)+alpha2b*f/bE1b*(1-bE1a/bE1b)
                     +alpha0a*f**2/bE1a**2*(bE1b/bE1a-1)+alpha0b*f**2/bE1b**2*(bE1a/bE1b-1))/2,
                     0,np.sqrt(sna*snb),np.sqrt(sn2a*sn2b),np.sqrt(sn4a*sn4b)])
    
def get_biaspoly(fishcast,X,Y,z,b=-1, b2=-1, bs=-1, 
                  alpha0=-1, alpha2=-1, alpha4=-1,alpha6=0.,
                  N=None,N2=-1,N4=0.,     
                  bL1=None,bL2=None,bLs=None,
                  ba=-1.,bb=-1.,b2a=-1.,b2b=-1.,bsa=-1.,bsb=-1.,
                  bL1a=None,bL1b=None,bL2a=None,bL2b=None,bLsa=None,bLsb=None,
                  alpha0a=-1,alpha0b=-1,alpha2a=0,alpha2b=0,alpha4a=0,alpha4b=0,f=-1,get_bvec=False,get_EPT=False):
   exp = fishcast.experiment
   if X=='k':
        N=0; N2=0; N4=0; alpha2=0; alpha4=0 #if marginalizing over these, remove
   if not(N==0 and N2==0): 
      fover = fishcast.experiment.fover[fishcast.sample2index(X,Y)]
      if N is None: N = fover/np.sqrt(compute_n(fishcast,z,X)*compute_n(fishcast,z,Y))
      noise = fover/np.sqrt(compute_n(fishcast,z,X)*compute_n(fishcast,z,Y))
      sigv = exp.sigv
      Hz = fishcast.Hz_fid(z)
      if N2 == -1: 
           if exp.N2 is not None: N2=exp.N2[fishcast.sample2index(X,Y)](z)
           else: N2 = -noise*((1+z)*sigv/fishcast.Hz_fid(z))**2
   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
   if b!=-1:
      ba=b
      bb=b
   if b2!=-1:
      b2a=b2
      b2b=b2
   if bs != -1:
      bsa = bs
      bsb = bs
   if ba==-1: ba = compute_b(fishcast,z,X)
   if bb==-1: bb = compute_b(fishcast,z,Y)
   if b2a==-1 and X!='k': b2a = exp.b2[X](z)
   elif b2a == -1: b2a = 8*(ba-1)/21 
   if b2b==-1 and Y!='k': b2b = exp.b2[Y](z)
   elif b2b == -1: b2b = 8*(bb-1)/21 
   #Y==k
   if bsa==-1 and exp.bs is not None: bsa = exp.bs[X](z)
   elif bsa == -1: bsa = -2*(ba-1)/7
   if bsb==-1 and exp.bs is not None: bsb = exp.bs[Y](z)
   elif bsb == -1: bsb = -2*(bb-1)/7
      
   if bL1a is None: bL1a = ba-1
   if bL1b is None: bL1b = bb-1
   if bL2a is None: bL2a = b2a - 8*(ba-1)/21
   if bL2b is None: bL2b = b2b - 8*(bb-1)/21
   if bLsa is None: bLsa = bsa + 2*(ba-1)/7
   if bLsb is None: bLsb = bsb + 2*(bb-1)/7 
   
   if alpha0!=-1: 
      alpha0a=alpha0; alpha0b=alpha0 
   if alpha0a == -1.:alpha0a = exp.alpha0[X](z) if X!='k' else exp.alphak(z)
   if alpha0b == -1.:alpha0b = exp.alpha0[Y](z) if Y!='k' else exp.alphak(z)
   
   if alpha2!=-1:
      alpha2a=alpha2; alpha2b=alpha2
   if alpha4!=-1:
      alpha4a=alpha4; alpha4b=alpha4
   bL3a = 0 #- (ba-1)/9
   bL3b = 0 #- (bb-1)/9

   bveca = [bL1a,bL2a,bLsa,bL3a,alpha0a,alpha2a,alpha4a,0,0,0,0]
   bvecb = [bL1b,bL2b,bLsb,bL3b,alpha0b,alpha2b,alpha4b,0,0,0,0]
   if get_bvec:
        return bveca,bvecb
   bpoly = compute_biaspoly(bveca,bvecb,f=f)
   if get_EPT: 
        barr1=[ba,b2a,bsa,0] #fix this later... b3 isn't 0 if bL3=0
        barr2=[bb,b2b,bsb,0]
        aNarr = bpoly[12:16].tolist()+[N,N2,N4]
        return barr1,barr2,aNarr


#    bpoly[12:] = np.array([alpha0,alpha2,alpha4,alpha6,N,N2,N4])
   bpoly[16:] = np.array([N,N2,N4])
   return bpoly
    

#################################################################################################
#################################################################################################
# scale-dependent growth rate (I'm not using this function anywhere)

# def compute_f(fishcast, z, step=0.01):
#    '''
#    Returns the scale-dependent growth factor.
#    '''
#    p_hi = compute_matter_power_spectrum(fishcast,z=z+step)
#    p_higher = compute_matter_power_spectrum(fishcast,z=z+2.*step)
#    p_fid = compute_matter_power_spectrum(fishcast,z=z)
#    dPdz = (p_fid - (4./3.) * p_hi + (1./3.) * p_higher) / ((-2./3.)*step)
#    return -(1.+z) * dPdz / (2. * p_fid)

#################################################################################################
# functions for calculating P_{mm}(k,mu), P_{gg}(k,mu), P_{XY}(k), C^{XY}_\ell, and the smoothed
# power spectrum

def compute_matter_power_spectrum(fishcast, z, linear=False):
   '''
   Computes the cdm + baryon power spectrum for a given cosmology
   at redshift z. By default returns the linear power spectrum, with
   an option to return the Halofit guess for the nonlinear power
   spectrum.
   Returns an array of length Nk*Nmu. 
   '''
   kk = np.logspace(np.log10(fishcast.kmin),np.log10(fishcast.kmax),fishcast.Nk)
   if linear: pmatter = np.array([fishcast.cosmo.pk_cb_lin(k*fishcast.cosmo.h(),z)*fishcast.cosmo.h()**3. for k in kk])
   else: pmatter = np.array([fishcast.cosmo.pk_cb(k*fishcast.cosmo.h(),z)*fishcast.cosmo.h()**3. for k in kk])
   return np.repeat(pmatter,fishcast.Nmu)


def get_smoothed_p(fishcast,klin,plin,division_factor=2.):
   '''
   Returns a power spectrum without wiggles, given by:
      P_nw = P_approx * F[P/P_approx]
   where P is the linear power spectrum, P_approx is given by Eisenstein & Hu (1998),
   and F is an SG low-pass filter.
   '''
   def Peh(k,p):
      '''
      Returns the smoothed power spectrum Eisenstein & Hu (1998).
      '''
      k = k.copy() * fishcast.params['h']
      Obh2      = fishcast.params['omega_b'] 
      Omh2      = fishcast.params['omega_b'] + fishcast.params['omega_cdm']
      f_baryon  = Obh2 / Omh2
      theta_cmb = fishcast.cosmo_fid.T_cmb() / 2.7
      k_eq = 0.0746 * Omh2 * theta_cmb ** (-2)
      sound_horizon = fishcast.params['h'] * 44.5 * np.log(9.83/Omh2) / \
                            np.sqrt(1 + 10 * Obh2** 0.75) 
      alpha_gamma = 1 - 0.328 * np.log(431*Omh2) * f_baryon + \
                0.38* np.log(22.3*Omh2) * f_baryon ** 2
      ks = k * sound_horizon / fishcast.params['h']
      q = k / (13.41*k_eq)
      gamma_eff = Omh2 * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
      q_eff = q * Omh2 / gamma_eff
      L0 = np.log(2*np.e + 1.8 * q_eff)
      C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
      Teh = L0 / (L0 + C0 * q_eff**2)
      t_with_wiggles = np.sqrt(p/k**fishcast.params['n_s'])
      t_with_wiggles /= t_with_wiggles[0]
      return p * (Teh/t_with_wiggles)**2.

   p_approx = Peh(klin,plin)
   psmooth = savgol_filter(plin/p_approx,int(fishcast.Nk/division_factor)+1-\
                                         int(fishcast.Nk/division_factor)%2, 6)*p_approx
   return psmooth



def compute_tracer_power_spectrum(fishcast, Xind, Yind, z, b=-1., b2=-1, bs=-1, 
                                  alpha0=-1, alpha2=-1, alpha4=-1,alpha6=0.,
                                  N=None,N2=-1,N4=0.,f=-1., A_lin=-1., 
                                  omega_lin=-1., phi_lin=-1.,A_log=-1., 
                                  omega_log=-1., phi_log=-1.,kIR=0.2,     
                                  moments=False,bL1=None,bL2=None,bLs=None,
                                  one_loop=True,ba=-1.,bb=-1.,b2a=-1.,b2b=-1.,bsa=-1.,bsb=-1.,
                                  bL1a=None,bL1b=None,bL2a=None,bL2b=None,bLsa=None,bLsb=None,
                                  alpha0a=-1,alpha0b=-1,alpha2a=0,alpha2b=0,alpha4a=0,alpha4b=0,
                                  alpha_perp=None, alpha_parallel=None, ap_deriv=False, sigmaS=None, sigmaPar=None, sigmaPerp=None,
                                  ell_mult = None):
   '''
   Computes the nonlinear redshift-space power spectrum P(k,mu) [Mpc/h]^3 
   of the matter tracer. Returns an array of length Nk*Nmu. 
   '''
   if b!=-1 and (ba!=-1 or bb!=-1):print('both b and ba, bb provided')
   if b2!=-1 and (b2a!=-1 or b2b!=-1):print('both b2 and b2a, b2b provided')
   if bs!=-1 and (bsa!=-1 or bsb!=-1):print('both bs and bsa, bsb provided')
   exp = fishcast.experiment
   bpoly=get_biaspoly(fishcast,Xind,Yind,z,b,b2,bs,alpha0,alpha2,alpha4,alpha6,
                   N,N2,N4,bL1,bL2,bLs,ba,bb,b2a,b2b,bsa,bsb,bL1a,bL1b,bL2a,bL2b,bLsa,bLsb,
                   alpha0a,alpha0b,alpha2a,alpha2b,alpha4a,alpha4b)
   
   if f == -1.: f = fishcast.cosmo.scale_independent_growth_factor_f(z)
   
   if fishcast.recon:
      model_params = {'alpha_perp': alpha_perp, 'alpha_parallel':alpha_parallel, 'ap_deriv': ap_deriv, 
                   'sigmaS': sigmaS, 'sigmaPar': sigmaPar, 'sigmaPerp': sigmaPerp, 
                   'f':f, 'ba': None, 'bb': None, 'r': None}
      
      print('b, ba,bb = {},{},{}'.format(b,ba,bb))
      if b != -1: 
         model_params['ba'] = b
         model_params['bb'] = b
      else:
         if ba==-1: ba = compute_b(fishcast,z,Xind)
         if bb==-1: bb = compute_b(fishcast,z,Yind)

         model_params['ba'] = ba 
         model_params['bb'] = bb

      print('WS ba,bb = {},{}'.format(model_params['ba'],model_params['bb']))

      return compute_recon_power_spectrum(fishcast,z,Xind, Yind, bpoly, moments, model_params)
   
   if A_lin == -1.: A_lin = fishcast.A_lin
   if omega_lin == -1.: omega_lin = fishcast.omega_lin
   if phi_lin == -1.: phi_lin = fishcast.phi_lin 
   if A_log == -1.: A_log = fishcast.A_log
   if omega_log == -1.: omega_log = fishcast.omega_log

   K = fishcast.k
   MU = fishcast.mu
   # h = fishcast.params['h']
   h = fishcast.cosmo.h()
   klin = np.array([K[i*fishcast.Nmu] for i in range(fishcast.Nk)])
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin]) 
   prim_feat_fac =  1. + A_lin * np.sin(omega_lin * klin + phi_lin)
   prim_feat_fac += A_log * np.sin(omega_log * np.log(klin*h/0.05)  + phi_log)
   plin *= prim_feat_fac


   if fishcast.smooth: plin = get_smoothed_p(fishcast,z)
    
   if fishcast.linear2:
      print('forecast.linear2 deprecated')
      pmatter = np.repeat(plin,fishcast.Nmu)
      result = pmatter * (b+f*MU**2.)**2.
      result /= 1 - N2*(K*MU)**2/noise 
      result += N   
      return result


   if fishcast.linear:
      print('forecast.linear deprecated')
      # If not using velocileptors, use linear theory
      # and approximate RSD with Kaiser.
      if b!=-1:
            ba=b; bb=b;
      if ba==-1:ba = compute_b(fishcast,z,Xind)
      if bb==-1:bb = compute_b(fishcast,z,Yind)
      if N is None: N = bpoly[16]
      if N2 == -1: N2 = bpoly[17]
      
      if alpha0 ==-1: alpha0=bpoly[12]
      pmatter = np.repeat(plin,fishcast.Nmu)
      result = pmatter * (ba+f*MU**2.)*(bb+f*MU**2)
      result += N+N2*(K*MU)**2+N4*(K*MU)**4
      result += pmatter*K**2*(alpha0+alpha2*MU**2+alpha4*MU**4)
      return result

   lpt = LPT_RSD(klin,plin,kIR=kIR,one_loop=one_loop,cutoff=2)
   lpt.make_pltable(f,kmin=min(klin),kmax=max(klin),nk=len(klin))
   k = lpt.kv
   p0 = np.sum(lpt.p0ktable * bpoly,axis=1)# + sn + 1./3 * kv**2 * sn2 + 1./5 * kv**4 * sn4
   p2 = np.sum(lpt.p2ktable * bpoly,axis=1)# + 2 * kv**2 * sn2 / 3 + 4./7 * kv**4 * sn4
   p4 = np.sum(lpt.p4ktable * bpoly,axis=1)# + 8./35 * kv**4 * sn4
   
   if moments: return k,p0,p2,p4
   p0 = np.repeat(p0,fishcast.Nmu) 
   p2 = np.repeat(p2,fishcast.Nmu) 
   p4 = np.repeat(p4,fishcast.Nmu)
   pkmu = p0+0.5*(3*MU**2-1)*p2+0.125*(35*MU**4-30*MU**2+3)*p4
   if ell_mult == 0: pkmu = p0
   elif ell_mult == 2: pkmu = 0.5*(3*MU**2-1)*p2
   elif ell_mult == 4: pkmu = 0.125*(35*MU**4-30*MU**2+3)*p4
   del lpt
   return pkmu

def compute_real_space_cross_power(fishcast, X, Y, z, gamma=1., b=-1., 
                                   b2=-1,bs=-1,alpha0=-1,alphax=0,N=None,
                                   ba=-1,bb=-1,b2a=-1,b2b=-1,bsa=-1,bsb=-1,bpoly=None,alpha0a=-1,alpha0b=-1,useHF=True):
   '''
   Wrapper function for CLEFT. Returns P_XY where X,Y = k or g
   as a function of k.
   '''
   if X!='k' and Y=='k': print('invert input please')
   # bk = (1+gamma)/2-1 #gamma functionality left behind for now
   if X!='k' and Y!='k' and bpoly==None:
        bpoly=get_biaspoly(fishcast,X,Y,z,b=b,b2=b2,bs=bs,alpha0=alpha0,alpha2=0,alpha4=0,alpha6=0,
                       N=N,N2=0,N4=0,ba=ba,bb=bb,b2a=b2a,b2b=b2b,bsa=bsa,bsb=bsb,
                       alpha0a=alpha0a,alpha0b=alpha0b,alpha2a=0,alpha2b=0,alpha4a=0,alpha4b=0)
   elif Y!='k' and bpoly==None:
        bpoly=get_biaspoly(fishcast,X,Y,z,b=b,b2=b2,bs=bs,alpha0=alpha0,alpha2=0,alpha4=0,alpha6=0,
                       N=N,N2=0,N4=0,ba=ba,bb=bb,b2a=b2a,b2b=b2b,bsa=0,bsb=bsb,
                       alpha0a=alpha0a,alpha0b=alpha0b,alpha2a=0,alpha2b=0,alpha4a=0,alpha4b=0)
   h = fishcast.cosmo.h()
   
   K = fishcast.k
    
   klin = np.array([K[i*fishcast.Nmu] for i in range(fishcast.Nk)])
   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin])
    
   ######################################################################################################## 
   if fishcast.linear:
      print('forecast.linear deprecated')
    
   if X == Y and X == 'k': 
      plin = np.array([fishcast.cosmo.pk_lin(k*h,z)*h**3. for k in klin])
      cleft = CLEFT(klin,plin,cutoff=2.)
      cleft.make_ptable(kmin=min(klin),kmax=max(klin),nk=fishcast.Nk)
      kk,pmm = cleft.combine_bias_terms_pk(0,0,0,0,0,0) 
      if z>10: pmm=np.array([fishcast.cosmo.pk(k*h,z)*h**3. for k in klin])
      return interp1d(kk, pmm, kind='linear', bounds_error=False, fill_value=0.)
      
   cleft = CLEFT(klin,plin,cutoff=2.)
   cleft.make_ptable(kmin=min(klin),kmax=max(klin),nk=fishcast.Nk)
   
   if bpoly is not None: #ga gb ### and k g
      kk = cleft.pktable[:,0]
      pktemp = np.copy(cleft.pktable)[:,1:-1]
      pgg = np.sum(pktemp*bpoly[:12],axis=1)+bpoly[12]*kk**2* cleft.pktable[:,-1] +bpoly[16]
      return interp1d(kk, pgg, kind='linear', bounds_error=False, fill_value=0.) 
   else: 
      print('bias polynomial undefined')
      return None   
   
def compute_lensing_Cell(fishcast, X, Y, zmin=None, zmax=None,zmid=None,gamma=1., 
                         b=-1,b2=-1,bs=-1, alpha0=-1,alphax=0.,N=-1,
                         noise=False,Nzsteps=100,Nzeff='auto',maxDz=0.2,
                         ba=-1,bb=-1,b2a=-1,b2b=-1,bsa=-1,bsb=-1,alpha0a=-1,alpha0b=-1,useHF=True):
   '''
   Calculates C^XY_l using the Limber approximation where X,Y = 'k' or 'g'. 
   Returns an array of length len(fishcast.ell). If X=Y=k, use CLASS with 
   HaloFit to calculate the lensing convergence. 
   ---------------------------------------------------------------------
   noise: if True returns the projected shot noise. 
   (replaces P -> 1/n)

   Nzsteps: number of integration points
   ---------------------------------------------------------------------
   '''
   if X == Y and X == 'k' and zmin is None and zmax is None: 
      lmin,lmax = int(min(fishcast.ell)),int(max(fishcast.ell))
      Cphiphi = fishcast.cosmo.raw_cl(lmax)['pp'][lmin:]
      return 0.25*fishcast.ell**2*(fishcast.ell+1)**2*Cphiphi*((1+gamma)/2)**2

   if zmin is None or zmax is None: raise Exception('Must specify zmin and zmax')
   if X!='k' and Y=='k': print('invert input please')
   if zmid is None: zmid = (zmin+zmax)/2
   exp = fishcast.experiment
       
   if b!=-1: ba=b; bb=b
   if b2!=-1: b2a=b2; b2b=b2
   if bs!=-1: bsa=bs; bsb=bs
   
   ba_fid = fishcast.experiment.b[X](zmid) if X!='k' else 1
   bb_fid = fishcast.experiment.b[Y](zmid) if Y!='k' else 1
   b2a_fid = fishcast.experiment.b2[X](zmid) if X!='k' else 0
   b2b_fid = fishcast.experiment.b2[Y](zmid) if Y!='k' else 0
   bsa_fid = -2*(ba_fid-1)/7
   bsb_fid = -2*(bb_fid-1)/7
   alpha0a_fid = fishcast.experiment.alpha0[X](zmid) if X!='k' else fishcast.experiment.alphak(zmid)
   alpha0b_fid = fishcast.experiment.alpha0[Y](zmid) if Y!='k' else fishcast.experiment.alphak(zmid)
   # if fishcast.experiment.HI: N_fid = castorinaPn(zmid)
   N_fid = fishcast.experiment.fover[fishcast.sample2index(X,Y)]/np.sqrt(fishcast.experiment.n[X](zmid)*fishcast.experiment.n[Y](zmid)) if (X!='k' and Y!='k') else 0

   if ba==-1: ba=ba_fid
   if bb==-1: bb=bb_fid
   if b2a==-1: b2a=b2a_fid
   if b2b==-1: b2b=b2b_fid
   if bsa==-1: bsa=bsa_fid
   if bsb==-1: bsb=bsb_fid
   
   if alpha0!=-1: alpha0a=alpha0; alpha0b=alpha0
   if alpha0a ==-1: alpha0a = alpha0a_fid
   if alpha0b ==-1: alpha0b = alpha0b_fid

   if N == -1: N = N_fid
       
   z_star = 1098
   chi = lambda z: (1.+z)*fishcast.cosmo.angular_distance(z)*fishcast.cosmo.h()
   def dchidz(z): 
      if z <= 0.02: return (chi(z+0.01)-chi(z))/0.01
      return (-chi(z+0.02)+8*chi(z+0.01)-8*chi(z-0.01)+chi(z-0.02))/0.12 
   chi_star = chi(z_star)  

   # CMB lensing convergence kernel
   W_k = lambda z: 1.5*(fishcast.params['omega_cdm']+fishcast.params['omega_b'])/\
                   fishcast.cosmo.h()**2*(1/2997.92458)**2*(1+z)*chi(z)*\
                   (chi_star-chi(z))/chi_star
   
   # Galaxy kernel (arbitrary normalization)
   def nonnorm_Wg(z,samplenumber=0):
      result = fishcast.cosmo.Hubble(z)
      try:
         number_density = compute_n(fishcast,z,samplenumber)
      except:
         if X == Y and X == 'k': number_density = 10
         else: raise Exception('Attemped to integrate outside of \
                                specificed n(z) range')
      # if fishcast.experiment.HI: number_density = castorinaPn(z)
      result *= number_density * dchidz(z) * chi(z)**2 
      return result
  
   zs = np.linspace(zmin,zmax,Nzsteps)
   chis = np.array([chi(z) for z in zs])
   if X!='k' and Y!='k':
       WgX = np.array([nonnorm_Wg(z,X) for z in zs])
       WgX /= simps(WgX,x=chis)
       WgY = np.array([nonnorm_Wg(z,Y) for z in zs])
       WgY /= simps(WgY,x=chis)
       kern = WgX*WgY/chis**2
   elif Y!='k' and X=='k':
       Wk = np.array([W_k(z) for z in zs])
       WgY = np.array([nonnorm_Wg(z,Y) for z in zs])
       WgY /= simps(WgY,x=chis)
       kern = WgY*Wk/chis**2
   else: 
       Wk = np.array([W_k(z) for z in zs])
       kern = Wk**2/chis**2
   if Nzeff == 'auto': Nzeff = ceil((zmax-zmin)/maxDz)
   if Nzeff > Nzsteps: Nzeff = Nzsteps
   mask = [(zs < np.linspace(zmin,zmax,Nzeff+1)[1:][i])*\
           (zs >= np.linspace(zmin,zmax,Nzeff+1)[:-1][i]) for i in range(Nzeff)]
   mask[-1][-1] = True
   zeff = np.array([simps(kern*zs*m,x=chis)/simps(kern*m,x=chis) for m in mask])
   if X=='k' and Y=='k':
       if not noise: 
            P = [compute_real_space_cross_power(
                 fishcast, X, Y, zz, gamma=gamma, b=0, 
                 b2=0,bs=0,alpha0=0,alphax=0,N=0,useHF=useHF) for zz in zeff]
       else: 
            P = [np.zeros(fishcast.ell) for i in range(len(zeff))]
       P = np.array(P)

       def result(ell):
          kval = (ell+0.5)/chis
          integrands = np.array([kern*P[i](kval) for i in range(Nzeff)])*mask
          integrand = np.sum(integrands,axis=0)
          return simps(integrand,x=chis)

       return np.array([result(l) for l in fishcast.ell])

   baz = lambda z: compute_b(fishcast,z,X)*ba/ba_fid
   bbz = lambda z: compute_b(fishcast,z,Y)*bb/bb_fid
   bsaz = lambda z: -2*(compute_b(fishcast,z,X)-1)/7 * bsa/bsa_fid if bsa_fid!=0 else bsa
   bsbz = lambda z: -2*(compute_b(fishcast,z,Y)-1)/7 * bsb/bsb_fid if bsb_fid!=0 else bsb
   b2az = lambda z: exp.b2[X](z) * b2a/b2a_fid if b2a_fid!=0 else b2a
   b2bz = lambda z: exp.b2[Y](z) * b2b/b2b_fid if b2b_fid!=0 else b2b
   def alpha0az(z):
       if alpha0a_fid ==0: return alpha0a
       if X=='k': return fishcast.experiment.alphak(z) * alpha0a/alpha0a_fid
       return fishcast.experiment.alpha0[X](z) * alpha0a/alpha0a_fid
   def alpha0bz(z):
       if alpha0b_fid ==0: return alpha0b
       if Y=='k': return fishcast.experiment.alphak(z) * alpha0b/alpha0b_fid
       return fishcast.experiment.alpha0[Y](z) * alpha0b/alpha0b_fid
   def Nz(z):
       if N_fid ==0: return N #any k case goes here
#        if X=='k' or Y=='k': return forecast.experiment.alphak(z) * alpha0b/alpha0b_fid
       return fishcast.experiment.fover[fishcast.sample2index(X,Y)]/np.sqrt(fishcast.experiment.n[X](z)*fishcast.experiment.n[Y](z))

   # calculate P_XY 
   if not noise: 
        P = [compute_real_space_cross_power(
             fishcast, X, Y, zz, gamma=gamma, ba=baz(zz), bb=bbz(zz), b2a=b2az(zz), b2b=b2bz(zz),
             bsa=bsaz(zz),bsb=bsbz(zz),alpha0a=alpha0az(zz),alpha0b=alpha0bz(zz),N=Nz(zz)) for zz in zeff]
   if noise: 
        #this ignores fover - assumes fover==0 for cross
        if X!='k' and X==Y:
            P = [fishcast.get_f_at_fixed_mu(np.ones(fishcast.Nk*fishcast.Nmu)/\
                  np.sqrt(compute_n(fishcast,zz,X)*compute_n(fishcast,zz,Y)),0) for zz in zeff]
        else: return np.zeros(len(fishcast.ell))
   P = np.array(P)

   def result(ell):
      kval = (ell+0.5)/chis
      integrands = np.array([kern*P[i](kval) for i in range(Nzeff)])*mask
      integrand = np.sum(integrands,axis=0)
      return simps(integrand,x=chis)
   
   return np.array([result(l) for l in fishcast.ell])
    

#def compute_recon_power_spectrum(fishcast,z,b=-1.,b2=-1.,bs=-1.,N=None):
def compute_recon_power_spectrum(fishcast,z,Xind, Yind, bpoly, moments, model_params):

   '''
   Returns reconstructed power spectrum based on the DESI wiggle-split method 
   or Stephen's LPT reconstruction.

   From Eq. 3.20 of Noah's paper P_obs = a_{||}^{-1} a_\perp^{-2} * P_recon + N, N=1/shot noise. Here N = c_00 is technically 
   one term from the broadband polynomials but all other c_ij are zero. The a_{||}^{-1} a_\perp^{-2} prefactor enters into the derivative calculation in fisherForecast.py

   From Eq. 6.13 of DESI 2024 BAO comparison paper, P_obs_ell is exactly what I output in wiggle_split_recon_power_spectrum. 

   '''

   K,MU = fishcast.k,fishcast.mu
   h = fishcast.cosmo.h()

   klin = np.logspace(np.log10(min(K)),np.log10(max(K)),fishcast.Nk)
   #klin = np.array([K[i*fishcast.Nmu] for i in range(fishcast.Nk)]) #Why were we using a different k grid than the pre-reconstruction case?

   mulin = MU.reshape((fishcast.Nk,fishcast.Nmu))[0,:]

   plin = np.array([fishcast.cosmo.pk_cb_lin(k*h,z)*h**3. for k in klin])

   if fishcast.recon == 'LPT':
      p0, p2, p4 = LPT_recon_power_spectrum(fishcast,z,bpoly, klin, plin, f=model_params['f'])
      
   elif fishcast.recon == 'wigglesplit':

      ba = model_params['ba']
      bb = model_params['bb']

      n = np.sqrt(compute_n(fishcast,z,Xind) * compute_n(fishcast,z,Yind))

      model_params['r'] = recon_efficiency(fishcast,z=z,n=n,ba=ba,bb=bb,f=model_params['f'])
      
      sigmaS, sigmaPar, sigmaPerp = fishcast.compute_recon_sigmas(z, model_params)

      model_params['sigmaS'] = sigmaS
      model_params['sigmaPar'] = sigmaPar 
      model_params['sigmaPerp'] = sigmaPerp

      p0, p2, p4 = wiggle_split_recon_power_spectrum(fishcast,z, klin, mulin, plin, model_params)
   
   else:
      raise Exception('recon method not recognized')

   if moments:
      return p0, p2, p4

   l0,l2,l4 = legendre(0),legendre(2),legendre(4)

   Pk = lambda mu: p0*l0(mu) + p2*l2(mu) + p4*l4(mu)

   result = np.array([Pk(mu) for mu in mulin]).T

   N = bpoly[16]

   return result.flatten() + N
      
   
def wiggle_split_recon_power_spectrum(fishcast,z, klin, mulin, plin, model_params):
   '''
   Returns the reconstructed power spectrum, following the DESI wiggle-split method.
   '''   
   wsplit = WiggleSplit_Recon(fishcast=fishcast,z=z,k=klin,mu=mulin,plin=plin, model_params=model_params)

   return wsplit.compute()


def LPT_recon_power_spectrum(fishcast,z,bpoly, klin, plin, f):
   '''
   Returns the reconstructed power spectrum, following Stephen's paper.
   '''
   bias_factors = bpoly[:13]

   bias_factors[10]=0;bias_factors[11]=0;bias_factors[12]=0;

   K = fishcast.k

   # if fishcast.experiment.HI: print('not set up yet')#noise = castorinaPn(z)
   #if N is None: N = 1/compute_n(fishcast,z)
   #f = fishcast.cosmo.scale_independent_growth_factor_f(z) 
 
   zelda = Zeldovich_Recon(klin,plin,R=15,N=2000,jn=5)

   kSparse,p0ktable,p2ktable,p4ktable = zelda.make_pltable(f,ngauss=3,kmin=min(K),kmax=max(K),nk=200,method='RecSym')
   #bias_factors = np.array([1, bL1, bL1**2, bL2, bL1*bL2, bL2**2, bLs, bL1*bLs, bL2*bLs, bLs**2,0,0,0])
   p0Sparse = np.sum(p0ktable*bias_factors, axis=1)
   p2Sparse = np.sum(p2ktable*bias_factors, axis=1)
   p4Sparse = np.sum(p4ktable*bias_factors, axis=1)
   p0,p2,p4 = Spline(kSparse,p0Sparse)(klin),Spline(kSparse,p2Sparse)(klin),Spline(kSparse,p4Sparse)(klin)
   
   return p0, p2, p4
