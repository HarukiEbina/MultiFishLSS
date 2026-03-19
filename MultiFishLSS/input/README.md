The following files contain the lensing convergence noise curves: 
   nlkk_v3_1_0deproj0_SENS2_fsky0p4_it_lT30-3000_lP30-5000.dat
   nlkk_planck.dat
   S4_kappa_deproj0_sens0_16000_lT30-3000_lP30-5000.dat
See twoPointNoise.py for how we load the noise curves from these files.

The remaining files are CMB Fisher matrices in the basis
   'h','log(A_s)','n_s','omega_cdm','omega_b','tau_reio','m_ncdm','N_ur','alpha_s','Omega_k' 
Planck_SO combines low-ell Planck (ell < 30) with high-ell SO (ell > 30), and so on.

`alpha0k_fit.json` provides fits to the nonlinear matter power spectrum, which can be used as fiducial values for the matter counterterm.

`b2_ST.json` provides the second-order bias as a function of the linear bias, evaluated using the Sheth-Tormen peak-background split. The function is additionally extrapolated to lower bias than allowed by peak-background splits to forecast low-bias tracers. This is used as a fiducial relation between b1 and b2 in the forecasts.

If necessary, the extrapolation range of $b_2$ can be expanded by modifying and re-running `sheth_tormen.py`.