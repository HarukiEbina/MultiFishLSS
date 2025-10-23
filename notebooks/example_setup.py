#!/usr/bin/env python
import numpy as np
from classy import Class
from scipy.interpolate import interp1d
import sys, os, json
sys.path.append('../MultiFishLSS/')

from headers     import *
from twoPoint     import *
from twoPointNoise import *

# Set the default cosmological parameters.
default_cosmo = {'P_k_max_h/Mpc': 2.,
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

def make_forecast(cosmo,):
    """Generate an appropriate forecast instance."""
    fsky = 5000/41253 # 5000deg2
    zedges = np.linspace(2.7,3.3,3)
    nbins = 2

    b = [lambda z: 2.5, lambda z: 3.5]    
    n = [lambda z: 2e-4, lambda z: 2e-4]

    bd='./'
    fover=[1,0,1]
    samples=['ga','gb']
    surveyname='example'
    # exp = experiment(zmin=zmin, zmax=zmax, nbins=nbins, fsky=fsky, b=b, n=n,samples=samples,fover=fover)
    exp = experiment(zmin=np.min(zedges), zmax=np.max(zedges),zedges=zedges, nbins=nbins, fsky=fsky, b=b, n=n,samples=samples,fover=fover)
    if typ == 'setup':
        setup = True
    else:
        setup = False
    forecast = fisherForecast(experiment=exp,cosmo=cosmo,name=surveyname,setup=setup,basedir=bd,ell=np.arange(10,3000,1))    
    return forecast


def do_task(typ,):
    """Does the work, performing task "typ" on survey file base name "sfb"."""
    # When taking derivatives of P(k,mu) you don't need to get the lensing
    # Cell's from class. So to speed things up we'll use a different CLASS
    # object depending on the derivatives being calculated. 
    if typ == 'setup':
       params = {'output': 'tCl lCl mPk',\
                 'non linear':'halofit', \
                 'l_max_scalars': 4000,\
                 'lensing': 'yes'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       #
    elif typ == 'rec':
       params = {'output': 'mPk',\
                'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['alpha_perp','alpha_parallel','b'])
       forecast.recon = True
       forecast.free_params = basis
       forecast.compute_derivatives()
       forecast.recon = False

    elif typ == 'fs1':
       params = {'output': 'mPk',\
                'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['N','alpha0','b','b2','bs','N2','N4','alpha2','alpha4','RT',])
       forecast.free_params = basis
       forecast.compute_derivatives()
    elif typ == 'fs2':
       params = {'output': 'mPk',\
                'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['log(A_s)','omega_cdm','omega_b','tau_reio','h',])
       forecast.free_params = basis
       forecast.compute_derivatives()
    elif typ == 'fs3':
       params = {'output': 'mPk',\
                'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['n_s','m_ncdm','N_ur','alpha_s','Omega_k'])
       forecast.free_params = basis
       forecast.compute_derivatives()
       # 
    elif typ == 'lens1':
       params = {'output': 'tCl lCl mPk',\
              'l_max_scalars': 4000,\
              'lensing': 'yes',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['N','alpha0','b','b2','bs','alpha0k',])
       forecast.free_params = basis
       forecast.compute_Cl_derivatives()
    elif typ == 'lens2':
       params = {'output': 'tCl lCl mPk',\
              'l_max_scalars': 4000,\
              'lensing': 'yes',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['log(A_s)','h','n_s','omega_cdm','omega_b',])
       forecast.free_params = basis
       forecast.compute_Cl_derivatives()
    elif typ == 'lens3':
       params = {'output': 'tCl lCl mPk',\
              'l_max_scalars': 4000,\
              'lensing': 'yes',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       basis = np.array(['tau_reio','m_ncdm','N_ur','alpha_s','Omega_k'])
       forecast.free_params = basis
       forecast.compute_Cl_derivatives()
    elif typ == 'Alin':
       params = {'output': 'mPk',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       omega_lin_list = np.linspace(10,154,17).tolist()+np.linspace(175,300,6).tolist()
       for omega_lin in omega_lin_list:
           forecast.omega_lin = omega_lin
           basis = ['A_lin']
           forecast.free_params = basis
           forecast.compute_derivatives()
           forecast.omega_lin = 0.01
    elif typ == 'Alog':
       params = {'output': 'mPk',\
              'non linear':'halofit'}
       for k in default_cosmo.keys():
           params[k] = default_cosmo[k]
       #
       cosmo = Class(); cosmo.set(params); cosmo.compute();
       forecast = make_forecast(cosmo,)
       omega_log_list = [1]+[3]+np.arange(5,105,5).tolist()
       for omega_log in omega_log_list:
           forecast.omega_log = omega_log
           basis = ['A_log']
           forecast.free_params = basis
           forecast.compute_derivatives()
           forecast.omega_log = 0.01
    else:
        raise RuntimeError("Unknown task "+str(typ))
    cosmo.struct_cleanup()
    # print(sfb,typ)
    #

    
if __name__=="__main__":
    if len(sys.argv)<2:
        outstr = "Usage: "+sys.argv[0]+" <survey-filename> <task-name>"
        raise RuntimeError(outstr)
    # Extract the arguments.
    typ = sys.argv[1]
    # Do the actual work.
    do_task(typ)#,nbins,fsky)
