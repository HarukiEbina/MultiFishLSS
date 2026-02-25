import sys, os
sys.path.append('../MultiFishLSS/')
from headers import *
from twoPoint import *
from twoPointNoise import *
from classy import Class

default_cosmo = {
        'output': 'tCl lCl mPk',
        'non linear':'halofit',
        'l_max_scalars': 4000,
        'lensing': 'yes',
        'P_k_max_h/Mpc': 2.,
        'z_pk': '0.0,1087',
        'A_s': 2.10732e-9,
        'n_s': 0.96824,
        'alpha_s': 0.,
        'h': 0.6736,
        'N_ur': 2.0328,
        'N_ncdm': 1,
        'm_ncdm': 0.06,
        'tau_reio': 0.0544,
        'omega_b': 0.02237,
        'omega_cdm': 0.1200,
        'Omega_k': 0.,
        'Omega_Lambda': 0.,
        'w0_fld':-1,
        'wa_fld':0}
cosmo = Class()
cosmo.set(default_cosmo)
cosmo.compute()
h = cosmo.h()

chi = lambda zz: cosmo.comoving_distance(zz)*h

fsky = 5000/41253 # 5000 deg2

# two redshift bins from z = 2.7 to 3.3
zmin, zmax = 2.7, 3.3
nbins = 2

# 2 galaxy samples 
# name, bias, number density as lists
samples=['ga','gb']
b = [lambda z: 2.5, lambda z: 3.5] 
n = [lambda z: 2e-4, lambda z: 2e-4]

# overlap between stochastic terms
# index 0 and 2 are auto-correlations, so they must be 1
# index 1 is the cross-correlation between the two samples, set to 0 by default for no overlap
fover=[1,0,1]

# experiment name and basedir
bd='./'

recon = 'wigglesplit'
broadband = 'splines'

surveyname='bao_setup_{}_{}'.format(recon, broadband)

exp = experiment(zmin=zmin, zmax=zmax, nbins=nbins, fsky=fsky, b=b, n=n,samples=samples,fover=fover)

# May need to turn recon off here and turn it back on later otherwise the fiducial Pk is the Precon?

forecast = fisherForecast(experiment=exp,cosmo=cosmo,name=surveyname,basedir=bd,ell=np.arange(10,3000,1), 
                          recon_fid=recon, broadband=broadband, overwrite=True)    

basis = np.array(['alpha_perp','alpha_parallel','b'])

forecast.recon = recon

forecast.free_params = basis
forecast.compute_derivatives(overwrite=True) # will not overwrite unless specified

derivs = forecast.load_derivatives(basis)
F = lambda i: forecast.gen_fisher(basis, 100, derivatives=derivs, zbins=np.array([i]))
Fs = [F(i) for i in range(nbins)]
Fs = np.array(Fs)

Finvs = [np.linalg.inv(Fs[i]) for i in range(nbins)]
saperp = [np.sqrt(Finvs[i][0,0]) for i in range(nbins)]
saparr = [np.sqrt(Finvs[i][1,1]) for i in range(nbins)]
print('Relative error on DA/rd:',saperp)
print('Relative error on H*rd:',saparr)