import numpy as np

from scipy.signal import savgol_filter
from scipy import special
from scipy import fftpack, interpolate
from scipy.special import legendre
from scipy.integrate import simpson
from bao_recon.loginterp import loginterp


class DESI2024_Recon:
    """
    Class to perform wiggle/no-wiggle split of a linear power spectrum
    using the DESI 2024 reconstruction method. For more details please see 
    arXiv:2402.14070v2. Generalised for multiple tracers by modifying the 
    Kaiser term to be 
    
    (b_1 + fmu^2[1-S(k)]) x (b_2 + fmu^2[1-S(k)])

    Which reduces to the standard Kaiser term for the auto-spectra. 
    
    Notes
    --------------------
    Defaults to the RecSym convention where S(k)=0 per accompanying DESI theory 
    paper.

    """
    def __init__(
        self,
        fishcast,
        z,
        k: np.ndarray,
        mu: np.ndarray,
        plin: np.ndarray,
        model_params : dict,        
        method = 'RecSym',
        smoothing_scale = 15.0,
        integration = 'simps',
        baofilter = 'eh_savgol',
        ells = [0,2,4]
    ):
        self.fishcast = fishcast
        self.k = k
        self.mu = mu
        self.plin = plin
        self.z = z

        if model_params['alpha_perp'] == None:
            self.alpha_perp = 1.
        else:
            self.alpha_perp = model_params['alpha_perp']
        
        if model_params['alpha_parallel'] == None:
            self.alpha_parallel = 1.
        else:
            self.alpha_parallel = model_params['alpha_parallel']

        self.sigmaS = model_params['sigmaS']
        self.sigmaPar = model_params['sigmaPar']
        self.sigmaPerp = model_params['sigmaPerp']
        self.ba = model_params['ba']
        self.bb = model_params['bb']
        self.f = model_params['f']
        self.ap_deriv = model_params['ap_deriv']

        print(
        'sigmaS = {} \n ' \
        'sigmaPar = {} \n ' \
        'sigmaPerp = {} \n ' \
        'r = {} \n ' \
        'ba = {} \n ' \
        'bb = {} \n ' \
        'f = {} \n ' \
        'alpha_parallel = {} \n ' \
        'alpha_perp = {} \n ' \
        'ap_deriv = {}'.format(self.sigmaS,self.sigmaPar,self.sigmaPerp,
                               model_params['r'],self.ba,self.bb,self.f, 
                               self.alpha_parallel, self.alpha_perp, self.ap_deriv))

        self.ells = ells

        self.integration = integration

        if method == 'RecSym':
            self.Sk = 0 
        elif method == 'RecIso':
            self.Sk = np.exp(-1/2 * (k * smoothing_scale) ** 2)
        else:
            raise Exception("Invalid reconstruction method. Choose 'RecSym' or 'RecIso'.")
        
        if baofilter == 'wallish':
            self.pnw = self.get_smoothed_p_wallish2018(k, plin)
        elif baofilter == 'eh_savgol':
            self.pnw = self.get_smoothed_p_eh_savgol(fishcast,k,plin)
        else:
            raise Exception("Invalid BAO Filter. Choose 'wallish' or 'eh_savgol'.")
        
        self.pw = self.plin - self.pnw

        if self.integration == 'leggauss':
            self.set_leggauss_mu_wmu(mu=50)


    def compute(self):
        '''
        Computes the reconstructed power spectrum components.

        The model here is

        P_\ell(k,mu) = (2\ell+1)/2 * \int dmu [ B(k,mu)P_nw(k) + C(k,mu)P_w(k')] + D_\ell(k)

        For more information, consult arXiv:2402.14070v2 and arXiv:2411.19738v1
        '''
        pnw = self.pnw

        k = self.k[None,:]
        mu = self.mu[:,None]

        k_ap, mu_ap = self.get_kmu_ap(k, mu, alpha_perp=self.alpha_perp, alpha_parallel=self.alpha_parallel)

        pap = loginterp(self.k, self.plin)(k_ap) 
        pnwap = loginterp(self.k, pnw)(k_ap)

        pwap = pap - pnwap 

        B = self.compute_B(k, mu)
        C = self.compute_C(k_ap, mu_ap) # Compute C at AP corrected k and mu according to companion DESI 2024 paper https://arxiv.org/pdf/2411.19738 .
        D = np.sum(self.fishcast.splines_array, axis=0) # Compute spline array in forecast to use here and for marginalisation.

        if self.ap_deriv:
            # returns Pkmu = C(k_AP,mu_AP) * P_w(k_AP) so that 
            # dP/dAP = d/dAP( C(k_AP,mu_AP) * P_w(k_AP) )
            B = 0.
            D = 0. 
            #C = self.compute_C(k, mu) # Turn on to make dP/dAP = C(k,mu) * dP_w(k_AP)/dAP

        Pkmu = B * pnw + C * pwap

        Pkell = self.get_multipoles(Pkmu.T)

        Pkell += D

        p0, p2, p4 = Pkell[0], Pkell[1], Pkell[2]

        return p0, p2, p4

    def get_multipoles(self, pkmu):
        """
        Integrate P(k,mu) over mu. Default is Simpson integration.
        Can also use Gauss-Legendre quadrature as done in cosmodesi. 
        Simpson integration is not noticeably slower and is more interpretible.  
        """

        if self.integration == 'simps':
            print('Using simpson integration')
            pkell = np.zeros((len(self.ells), len(self.k)))

            for i, ell in enumerate(self.ells):

                L = legendre(ell)(self.mu)
                pkell[i] = (2*ell + 1) * simpson(pkmu * L[None,:], self.mu, axis=1)
            
        elif self.integration == 'leggauss':  
            pkell = np.sum( pkmu * self.wmu[:, None,:], axis=-1)

        return pkell

    def set_leggauss_mu_wmu(self, mu=50):
        '''
        Copied from cosmodesi/desilike/theories/galaxy_clustering/base.py  
        '''
        self.mu, wmu = self.weights_leggauss(mu, sym=True)

        self.wmu = np.array([wmu * (2 * ell + 1) * special.legendre(ell)(self.mu) for ell in self.ells])
 
    def weights_leggauss(self, nx, sym=False):
        '''
        Copied from cosmodesi/desilike/utils.
        '''
        x, wx = np.polynomial.legendre.leggauss((1 + sym) * nx)
        if sym:
            x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
        return x, wx

    def compute_C(self, k, mu):        
        '''
        Computes the BAO damping factor C(k,mu) following the DESI 2024 reconstruction method.
        '''
        Sk = self.Sk

        exp_fac = np.exp( -0.5 * k**2 * (mu**2 * self.sigmaPar**2 + (1 - mu**2) * self.sigmaPerp**2) )
        kaiser = (self.ba + self.f * mu**2 * (1-Sk)) *  (self.bb + self.f * mu**2 * (1-Sk))  # Modified Multitracer Kaiser term

        C = kaiser * exp_fac

        return C 

    def compute_B(self, k, mu):
        '''
        Computes the BAO damping factor B(k,mu) following the DESI 2024 reconstruction method.
        '''
        Sk = self.Sk
        
        FOG = 1. / ( 1. + 0.5 * (k * mu * self.sigmaS)**2 )**2.
        kaiser = (self.ba + self.f * mu**2 * (1-Sk)) *  (self.bb + self.f * mu**2 * (1-Sk))  # Modified Multitracer Kaiser term

        B = kaiser * FOG 

        return B

    def get_kmu_ap(self, k, mu, alpha_perp=None, alpha_parallel=None):
        '''
        Returns the redefined k and mu after applying the Alcock-Paczynski effect.
        '''

        a_iso = (alpha_parallel * alpha_perp**2) ** (1./3)
        a_ap = alpha_parallel / alpha_perp 

        k_ap = a_ap ** (1./3) / a_iso * (1 + mu**2 * ( 1 / a_ap**2 - 1 ) ) ** (1./2) * k 

        mu_ap = mu/a_ap * (1 + mu**2 * ( 1 / a_ap**2 - 1 ) ) ** (-1./2)

        return k_ap, mu_ap



    def get_smoothed_p_eh_savgol(self, fishcast,k,plin,division_factor=2.):
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

        p_approx = Peh(k,plin)
        psmooth = savgol_filter(plin/p_approx,int(fishcast.Nk/division_factor)+1-\
                                                int(fishcast.Nk/division_factor)%2, 6)*p_approx
        return psmooth
    
    def get_smoothed_p_wallish2018(self, klin, plin):
        """
        Copied from cosmodesi/cosmoprimo. Assumes klin is sampled densely across wide range in k, won't work 
        if it is not the same grid DESI uses because this method has hardcoded empirical factors relating 
        to where the BAO bump is in real space.

        Filter BAO wiggles by sine-transforming the power spectrum to real space (where the BAO is better localized),
        cutting the peak and interpolating with a spline.

        References
        ----------
        https://arxiv.org/pdf/1810.02800.pdf, Appendix D (thanks to Stephen Chen for the reference)
        https://arxiv.org/pdf/1003.3999.pdf

        Note
        ----
        Cosmodesi/cosmoprimo have hand-tuned parameters w.r.t. a reference.
        """
        
        k = klin 
        pk = plin

        kpk = np.log(k[:, None] * pk)
        kpkffted = fftpack.dst(kpk, type=2, axis=0, norm='ortho', overwrite_x=False)
        even = kpkffted[::2]
        odd = kpkffted[1::2]

        xeven, xodd = 1 + np.arange(even.shape[0]), 1 + np.arange(odd.shape[0])
        spline_even = interpolate.CubicSpline(xeven, even, axis=0, bc_type='clamped', extrapolate=False)
        # dd_even = ndimage.uniform_filter1d(spline_even(xeven,nu=2), 3, axis=0, mode='reflect')
        dd_even = spline_even(xeven, nu=2)
        spline_odd = interpolate.CubicSpline(xodd, odd, axis=0, bc_type='clamped', extrapolate=False)
        # dd_odd = ndimage.uniform_filter1d(spline_odd(xodd,nu=2), 3, axis=0, mode='reflect')
        dd_odd = spline_odd(xodd, nu=2)
        
        margin_first = 20
        margin_second = 5
        offset_even = offset_odd = (-10, 20)

        def smooth_even_odd(even, odd, dd_even, dd_odd):
            argmax_even = dd_even[margin_first:-margin_first].argmax() + margin_first
            argmax_odd = dd_odd[margin_first:-margin_first].argmax() + margin_first
            ibox_even = (argmax_even + offset_even[0], argmax_even + margin_second + dd_even[argmax_even + margin_second:-margin_first].argmax() + offset_even[1])
            ibox_odd = (argmax_odd + offset_odd[0], argmax_odd + margin_second + dd_odd[argmax_odd + margin_second:-margin_first].argmax() + offset_odd[1])
            mask_even = np.ones_like(even, dtype=np.bool_)
            mask_even[ibox_even[0]:ibox_even[1] + 1] = False
            mask_odd = np.ones_like(odd, dtype=np.bool_)
            mask_odd[ibox_odd[0]:ibox_odd[1] + 1] = False
            spline_even = interpolate.CubicSpline(xeven[mask_even], even[mask_even] * xeven[mask_even]**2, axis=-1, bc_type='clamped', extrapolate=False)
            spline_odd = interpolate.CubicSpline(xodd[mask_odd], odd[mask_odd] * xodd[mask_odd]**2, axis=-1, bc_type='clamped', extrapolate=False)
            return spline_even(xeven) / xeven**2, spline_odd(xodd) / xodd**2

        for iz in range(self.plin.shape[-1]):
            even[:, iz], odd[:, iz] = smooth_even_odd(even[:, iz], odd[:, iz], dd_even[:, iz], dd_odd[:, iz])

        merged = np.empty_like(kpkffted)
        merged[::2] = even
        merged[1::2] = odd
        kpknow = fftpack.idst(merged, type=2, axis=0, norm='ortho', overwrite_x=False)
        pknow = np.exp(kpknow) / k[..., None]

        mask = (k > 1e-2) & (k < 1.5)
        k, pknow = k[mask], pknow[mask]
        
        mask_left, mask_right = self.k < 5e-4, self.k > 2.
        k = np.concatenate([self.k[mask_left], k, self.k[mask_right]], axis=0)
        pknow = np.concatenate([self.plin[mask_left], pknow, self.plin[mask_right]], axis=0)
        pknow = interpolate.CubicSpline(k, pknow, axis=0, bc_type='clamped', extrapolate=False)(self.k)
        tophat = self._tophat(self.k, kmax=1., scale=20.)[..., None]
        wiggles = (self.plin / pknow - 1.) * tophat + 1.
        
        pnow = self.plin / wiggles

        return pnow

        
    def _tophat(self, k, kmax=1, scale=1):
        """
        Tophat Gaussian kernel.
        """
        tophat = np.ones_like(k)
        mask = k > kmax
        tophat[mask] *= np.exp(-scale**2 * (k[mask] / kmax - 1.)**2)
        return tophat
