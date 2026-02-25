import numpy as np
import sys, os
sys.path.append('./MultiFishLSS/')
from headers import *
from twoPoint import *
from twoPointNoise import *
from scipy.interpolate import interp1d
from classy import Class

# compute fiducial cosmology
params = {'output': 'mPk lCl','P_k_max_h/Mpc': 40.,'non linear':'halofit', 
          'z_pk': '0.0,10','A_s': 2.10732e-9,'n_s': 0.96824,
          'alpha_s': 0.,'h': 0.6770, 'N_ur': 1.0196,
          'N_ncdm': 2,'m_ncdm': '0.01,0.05','tau_reio': 0.0568,
          'omega_b': 0.02247,'omega_cdm': 0.11923,'Omega_k': 0.}
cosmo = Class()
cosmo.set(params)
cosmo.compute()

# set up the forecast for DESI ELGs 
print('setting up the forecast for DESI ELGs...')

bd = './' # replace with where you want to save derivatives (will create bdir/output/bfn/)
nbins = 4

zs = np.array([0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65])
dNdz = np.array([309,2269,1923,2094,1441,1353,1337,523,466,329,126])
N = 41252.96125 * dNdz * 0.1 # number of emitters in dz=0.1 across the whole sky
volume = np.array([((1.+z+0.05)*cosmo.angular_distance(z+0.05))**3. for z in zs])
volume -= np.array([((1.+z-0.05)*cosmo.angular_distance(z-0.05))**3. for z in zs])
volume *= 4.*np.pi*cosmo.pars['h']**3./3. # volume in Mpc^3/h^3
n = list(N/volume)
zs = np.array([0.6,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55,1.65,1.7])
n = [n[0]] + n
n = n + [n[-1]]
n = np.array(n)
n_interp = interp1d(zs, n, kind='linear', bounds_error=False, fill_value=0.)

b = lambda z: 0.84/cosmo.scale_independent_growth_factor(z)

exp = experiment(zmin=0.6, zmax=1.7, nbins=nbins, n=n_interp, b=b, fsky=0.34)

recon = 'wigglesplit'

bfn = 'desi_elg_{}'.format(recon)

forecast = fisherForecast(experiment=exp,cosmo=cosmo,name=bfn,basedir=bd, recon=recon, overwrite=False)

# compue derivatives of post-recon Pk
print('computing derivatives of post-recon Pk...')
basis = np.array(['alpha_perp','alpha_parallel','b'])
forecast.free_params = basis
forecast.compute_derivatives(overwrite=False)
derivs = forecast.load_derivatives(basis)

# compute the errors on alpha_perp and alpha_parallel in each redshift bin
for i in range(nbins):
    F = forecast.gen_fisher(basis, 100, derivatives=derivs, zbins=np.array([i]))
    print(f'bin {i}: sigma_alpha_perp = {np.sqrt(np.linalg.inv(F)[0,0]):.5f}, sigma_alpha_parallel = {np.sqrt(np.linalg.inv(F)[1,1]):.5f}')