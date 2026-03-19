import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import make_interp_spline
import os

'''
Calculate b2 from b1 using Sheth-Tormen peak-background split. 
For extremely low b1, peak-background split can fail, so extrapolate using a spline.
'''


def tame_extrapolator(x_data, y_data, decay_length=0.3):
    cs = make_interp_spline(x_data, y_data, k=2)
    x_lo, x_hi = x_data[0], x_data[-1]

    def evaluate(x):
        x = np.asarray(x, dtype=float)
        result = np.empty_like(x)

        interior = (x >= x_lo) & (x <= x_hi)
        result[interior] = cs(x[interior])

        for mask, x_bnd in [(x < x_lo, x_lo), (x > x_hi, x_hi)]:
            if not np.any(mask):
                continue
            dx = x[mask] - x_bnd
            result[mask] = cs(x[mask]) * np.exp(-np.abs(dx) / decay_length)

        return result

    return evaluate

def find_b2_ST(b,basis='Eulerian'):
    '''
    Use Sheth-Tormen peak-background split to find b2 from b1

    b: linear Eulerian bias
    '''

    # Lagrangian bias
    bL1 = b-1
    a=.707; p=.3; dc=1.686
    def get_ST_nu2(nu2,bL1):
        return a*nu2-1+2*p/(1+(a*nu2)**p)-dc*bL1
    nu2_ST = fsolve(get_ST_nu2, 5, args=(bL1))[0]
    bL2_ST = (a**2*nu2_ST**2-3*a*nu2_ST+2*p*(2*a*nu2_ST+2*p-1)/(1+(a*nu2_ST)**p))/dc**2

    if basis == 'Eulerian':
        # convert back to Eulerian bias
        return bL2_ST+8/21*bL1
    else:
        # return Lagrangian bias for interpolation/extrapolation since it should be suppressed at low b1
        return bL2_ST

b_interpolate = np.linspace(0.7,10,500)
b2_interpolate = np.zeros_like(b_interpolate)
for i in range(len(b_interpolate)):
    try: b2_interpolate[i] = find_b2_ST(b_interpolate[i], basis='Lagrangian')
    except: b2_interpolate[i] = np.nan


# Remove any NaN values first
mask = ~np.isnan(b2_interpolate)
b_clean = b_interpolate[mask]
bL2_clean = b2_interpolate[mask]

b2_spline = tame_extrapolator(b_clean, bL2_clean, decay_length=0.2)

# plot interpolation and extrapolation regions to verify behavior
import matplotlib.pyplot as plt

# Evaluate the spline over a wider range to show extrapolation
b_eval = np.linspace(-1., 10, 500)
bL2_eval = b2_spline(b_eval)
b2_eval = bL2_eval + 8/21*(b_eval-1) # convert back to Eulerian bias for plotting

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the direct Sheth-Tormen calculation
ax.plot(b_clean, bL2_clean, 'o', ms=4, label='Sheth-Tormen (direct)', zorder=3)

# Plot the spline (including extrapolated regions)
ax.plot(b_eval, bL2_eval, '-', label='Quadratic Spline + Exponential Decay ($b^L_2$)')
ax.plot(b_eval, b2_eval, '-', label='Corresponding $b_2$')

# Shade the extrapolation regions
ax.axvspan(b_eval[0], b_clean[0], alpha=0.15, color='gray', label='Extrapolation region')
ax.axvspan(b_clean[-1], b_eval[-1], alpha=0.15, color='gray')



ax.set_xlabel(r'$b_1$', fontsize=14)
ax.set_ylabel(r'$b_2$', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-1, 3.5)
ax.set_ylim(-2,5)

plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/b2_extrapolation.png', dpi=150)
plt.close()


# output to json for use in forecasts
import json

b2_data = {
    'b1': b_eval.tolist(),
    'b2': b2_eval.tolist()
}

json.dump(b2_data, open('b2_ST.json', 'w'), indent=2)