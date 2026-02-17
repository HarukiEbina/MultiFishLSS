#### Notebooks to run basic functions of MultiFishLSS.

- `BAO.ipynb` and `sigma8.ipynb` show how to make BAO and sigma8 forecasts
- `Mnu.ipynb` shows how to make neutrino mass forecasts
- `Nmodes_Veff.ipynb` shows how to compute the number of modes and effective volume for a given survey
- `example_setup.sh` and `example_setup.py` sets up the forecast object, including the derivatives necessary for Fisher calculations. These scripts parallelize the computation, making the computation time significantly shorter. Note that `MFISHLSS_BASE` has to be set to the local path in `example_setup.sh`.