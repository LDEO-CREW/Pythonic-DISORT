try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
import scipy as sc
from math import pi

def transform_interval(arr, c, d, a=-1, b=1):
    """Transforms a vector in interval [a, b] to a similar vector in [c, d]

    :Input:
     - *arr* (vector) - The vector to transform
     - *c* (float) - The beginning of interval [c, d]
     - *d* (float) - The end of interval [c, d]
     :Optional:
     - *a* (float) - The beginning of interval [a, b]
     - *b* (float) - The end of interval [a, b]


    :Output:
     - (vector) - The transformed vector
    """
    return (((arr - a) * (d - c)) / (b - a)) + c


def transform_weights(weights, c, d, a=-1, b=1):
    """Transform quadrature weights in interval [a, b] to similar weights in [c, d]

    :Input:
     - *weights* (float vector) - The weights to transform
     - *c* (float) - The beginning of interval [c, d]
     - *d* (float) - The end of interval [c, d]
     :Optional:
     - *a* (float) - The beginning of interval [a, b]
     - *b* (float) - The end of interval [a, b]

    :Output:
     - *weights* (vector) - The transformed weights
    """
    return weights * (d - c) / (b - a)
	
	
def calculate_nu(mu, phi, mu_p, phi_p):
    """Calculates the scattering angle nu between incident angle (mu_p, phi_p) and scattering angle (mu, phi)
    
    :Input:
     - *mu* (float vector) - Cosine of scattering polar angles
     - *phi* (float vector) - Scattering azimuthal angles
     - *mu_p* (float vector) - Cosine of incident polar angles
     - *phi_p* (float vector) - Incident azimuthal angles
     
     
    :Output:
     - *nu* (tensor / float) - The transformed weights 
    """
    mu, phi, mu_p, phi_p = np.atleast_1d(mu, phi, mu_p, phi_p)
    nu = mu_p[None, None, :, None] * mu[:, None, None, None] + np.sqrt(1 - mu_p**2)[
        None, None, :, None
    ] * np.sqrt(1 - mu**2)[:, None, None, None] * np.cos(
        phi_p[None, None, None, :] - phi[None, :, None, None]
    )
    return np.squeeze(nu)


def generate_Ds(m, Leg_coeffs, mu_arr_pos, w0, ells, degree_tile):
    """Generates the D term in the system of ODEs for each Fourier mode, see Section 3.2
    
    :Input:
     - *m* (integer) - The Fourier mode
     - *Leg_coeffs* (float vector) - Vector of weighted phase function Legendre coefficients
     - *mu_arr_pos* (float vector) - Positive mu (quadrature) values
     - *w0* (float) - Single-scattering albedo
     - *ells* (float vector) - Vector of Legendre polynomial orders
     - *degree_tile* (float matrix) - Matrix of Associated Legendre polynomials orders and degrees
     
     
    :Output:
     - *D_pos* (float matrix) - D with only elements that correspond to positive mu values
     - *D_neg* (float matrix) - D with only elements that correspond to negative mu values
    """
    Dm_term = Leg_coeffs[ells] * (
        sc.special.factorial(ells - m) / sc.special.factorial(ells + m)
    )

    asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
    asso_leg_term_neg = sc.special.lpmv(m, degree_tile, -mu_arr_pos)
    D_temp = (Dm_term)[None, :] * asso_leg_term_pos.T

    D_pos = (w0 / 2) * D_temp @ asso_leg_term_pos
    D_neg = (w0 / 2) * D_temp @ asso_leg_term_neg

    return D_pos, D_neg
	

def generate_Xs(m, Leg_coeffs, w0, mu0, I0, mu_arr_pos, ells, degree_tile):
    """Generates the X term in the system of ODEs for each Fourier mode, see Section 3.2
    
    :Input:
     - *m* (integer) - The Fourier mode
     - *Leg_coeffs* (vector) - WEIGHTED phase function Legendre coefficients
     - *w0* (float) - Single-scattering albedo
     - *mu0* (float) - Polar angle of the direct beam
     - *I0* (float) - Intensity of the direct beam
     - *mu_arr_pos* (vector) - Positive mu (quadrature) values
     - *ells* (vector) - Vector of Legendre polynomial orders
     - *degree_tile* (matrix) - Matrix of Associated Legendre polynomial orders and degrees
     
     
    :Output:
     - *X_pos* (float vector) - X with only elements that correspond to positive mu values
     - *X_neg* (float vector) - X with only elements that correspond to negative mu values
    """
    if m == 0:
        prefactor = w0 * I0 / (4 * pi)
    else:
        prefactor = w0 * I0 / (2 * pi)

    Xm_term = Leg_coeffs[ells] * (
        sc.special.factorial(ells - m) / sc.special.factorial(ells + m)
    )
    Xm_term2 = sc.special.lpmv(m, ells, -mu0)
    X_temp = prefactor * Xm_term * Xm_term2

    X_pos = X_temp @ sc.special.lpmv(m, degree_tile, mu_arr_pos)
    X_neg = X_temp @ sc.special.lpmv(m, degree_tile, -mu_arr_pos)

    return X_pos, X_neg
	
	
def generate_flux_functions(
    I0, mu0, tau0,
    GC_pos, GC_neg, 
    eigenvals, N,
    B_pos, B_neg, 
    mu_arr_pos, weights_mu, 
    scale_tau,
): 
    """Generates the flux functions with respect to the radiative transfer equation
    
    :Input:
     - *I0* (float) - Intensity of the direct beam
     - *mu0* (float) - Polar angle of the direct beam
     - *GC_pos / neg* (float matrix) - Product of eigenvectors and coefficients that correspond to positive / negative mu values
     - *eigenvals* (float vector) - Eigenvalues
     - *B_pos / neg* (float vector) - Coefficients of the inhomogenity that correspond to positive / negative mu values
     - *mu_arr_pos / neg* (float vector) - Positive / negative mu (quadrature) values
     - *scale_tau* (float) - Delta-M scale factor for tau
     
     
    :Output:
     - *flux_up* (function) - Flux function with argument tau for positive (upward) mu values
     - *flux_down* (function) - Flux function with argument tau for negative (downward) mu values
    """
    def flux_up(tau):
        """Returns the magnitude of the upwards flux at the specified tau levels

        :Input:
        - *tau* (float vector) - Optical depth levels

        :Output:
        - (float) Magnitude of diffuse upwards flux, which is also the total upwards flux
        """
        tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
        exponent = np.vstack(
            (
                eigenvals[:N, None] * tau[None, :],
                eigenvals[N:, None] * (tau - tau0)[None, :],
            )
        )
        um = GC_pos @ np.exp(exponent) + B_pos[:, None] * np.exp(-tau[None, :] / mu0)
        return np.squeeze(2 * pi * (mu_arr_pos * weights_mu) @ um)[()]

    def flux_down(tau):
        """Returns the magnitude of the downwards fluxes at the specified tau levels

        :Input:
        - *tau* (float vector) - Optical depth levels

        :Output:
        - (float) Magnitude of diffuse downwards flux
        - (float) Magnitude of direct downwards flux
        """
        direct_beam = I0 * mu0 * np.exp(-tau / mu0)

        tau = scale_tau * np.atleast_1d(tau)  # Delta-M scaling
        direct_beam_scaled = I0 * mu0 * np.exp(-tau / mu0)
        exponent = np.vstack(
            (
                eigenvals[:N, None] * tau[None, :],
                eigenvals[N:, None] * (tau - tau0)[None, :],
            )
        )
        um = GC_neg @ np.exp(exponent) + B_neg[:, None] * np.exp(-tau[None, :] / mu0)
        return (
            np.squeeze(2 * pi * (mu_arr_pos * weights_mu) @ um + direct_beam_scaled - direct_beam)[()],
            direct_beam,
        )

    return flux_up, flux_down
    

# The following function is exactly NumPy's `atleast_2d` function but altered
# to add dimensions to the back of the shape tuple rather than to the front.
# Documentation for `np.atleast_2d` taken from https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html.
def atleast_2d_append(*arys):
    """
    View inputs as arrays with at least two dimensions.
    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.
    Returns
    -------
    res, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.
    """
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[:, np.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res