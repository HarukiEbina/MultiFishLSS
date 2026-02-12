### Code structure 

#### Scripts
* `experiment.py` defines the experiment object (redshift range, sky coverage, linear bias, number density, stochastic overlap etc.).
* `twoPoint.py` contains all the code relevant for computing power spectra ($P(k)$ and $C_\ell$)
* `twoPointNoise.py` contains all the code relevant for computing covariance matrices
* `fisherForecast.py` defines a forecast object, compute derivatives and Fisher matrices. All modules in the scripts above can be called through a forecast object. 

#### Subdirectories
* `input/` contains CMB fisher matrices, CMB lensing noise curves, assumed fiducial reionization history, and fiducial values for the matter counterterms
* `bao_recon/` is a copy of [ZeldovichReconPk](https://github.com/sfschen/ZeldovichReconPk), is used to compute the reconstructed power spectrum, and is wrapped in `twoPoint.py`
