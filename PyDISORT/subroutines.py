try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
import scipy as sc
from math import pi
from numpy.polynomial.legendre import leggauss


def transform_interval(arr, c, d, a=-1, b=1):
    """Affine transformation of an array from interval [a, b] to [c, d]

    Args:
        arr (array): The 1D array to transform
        c (float): The beginning of interval [c, d]
        d (float): The end of interval [c, d]
        a (float, optional): The beginning of interval [a, b]
        b (float, optional): The end of interval [a, b]
        
    Returns:
        array: The transformed 1D array
       
    """
    return (((arr - a) * (d - c)) / (b - a)) + c

def transform_weights(weights, c, d, a=-1, b=1):
    """Transforms quadrature weights from interval [a, b] to [c, d]

    Args:
        weights (array): The weights to transform
        c (float): The beginning of interval [c, d]
        d (float): The end of interval [c, d]
        a (float, optional): The beginning of interval [a, b]
        b (float, optional): The end of interval [a, b]
        
    Returns:
        array: The transformed weights
        
    """
    return weights * (d - c) / (b - a)
	
	
def calculate_nu(mu, phi, mu_p, phi_p):
    """Calculates the scattering angle nu between incident angle (mu_p, phi_p) and scattering angle (mu, phi)

    Args:
        mu (array): Cosine of outgoing polar angles
        phi (array): Outgoing azimuthal angles
        mu_p (array): Cosine of incident polar angles
        phi_p (array): Incident azimuthal angles
        
    Returns:
        ndarray: Cosine of scattering angles
        
    """
    mu, phi, mu_p, phi_p = np.atleast_1d(mu, phi, mu_p, phi_p)
    nu = mu_p[None, None, :, None] * mu[:, None, None, None] + np.sqrt(1 - mu_p**2)[
        None, None, :, None
    ] * np.sqrt(1 - mu**2)[:, None, None, None] * np.cos(
        phi_p[None, None, None, :] - phi[None, :, None, None]
    )
    return np.squeeze(nu)
	
    
def generate_flux_functions(
    tau_arr, I0, mu0,
    GC_pos, GC_neg, 
    eigenvals, N,
    B_pos, B_neg, 
    mu_arr_pos, weights_mu, 
    scale_tau,
): 
    """Generates the flux functions with respect to the radiative transfer equation

    Args:
        tau_arr (array): Optical depth of the lower boundary of each atmospheric layer
        I0 (float): Intensity of the incident beam
        mu0 (float): Polar angle of the incident beam
        GC_pos (2darray): Product of eigenvectors and coefficients that correspond to positive mu values
        GC_neg (2darray): Product of eigenvectors and coefficients that correspond to negative mu values
        eigenvals (array): Eigenvalues arranged negative then positive, from largest to smallest magnitude
        N (int): Half the number of quadrature points
        B_pos (array): Coefficients of the incident beam inhomogeneity that correspond to positive mu values
        B_neg (array): Coefficients of the incident beam inhomogeneity that correspond to negative mu values
        mu_arr_pos (array): Positive mu quadrature nodes
        weights_mu (array): mu quadrature weights for positive mu nodes
        scale_tau (array): Delta-M scale factors for tau
        
    Returns:
        function: Flux function with argument tau (array) for positive (upward) mu values
        function: Flux function with argument tau (array) for negative (downward) mu values
        
    """
    def flux_up(tau):
        """Returns the magnitude of the upwards flux at the specified tau levels

        Args:
            tau (array): Optical depth levels
            
        Returns:
            array: Magnitude of diffuse upwards flux, which is also the total upwards flux
            
        """
        tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
        exponent = np.vstack(
            (
                eigenvals[:N, None] * tau[None, :],
                eigenvals[N:, None] * (tau - taul)[None, :],
            )
        )
        u0_pos = GC_pos @ np.exp(exponent) + B_pos[:, None] * np.exp(-tau[None, :] / mu0)
        return np.squeeze(2 * pi * (mu_arr_pos * weights_mu) @ u0_pos)[()]

    def flux_down(tau):
        """Returns the magnitude of the downwards flux at the specified tau levels

        Args:
            tau (array): Optical depth levels
            
        Returns:
            array: Magnitude of diffuse downwards flux
            array: Magnitude of direct downwards flux
            
        """
        direct_beam = I0 * mu0 * np.exp(-tau / mu0)

        tau = scale_tau * np.atleast_1d(tau)  # Delta-M scaling
        direct_beam_scaled = I0 * mu0 * np.exp(-tau / mu0)
        exponent = np.vstack(
            (
                eigenvals[:N, None] * tau[None, :],
                eigenvals[N:, None] * (tau - taul)[None, :],
            )
        )
        u0_neg = GC_neg @ np.exp(exponent) + B_neg[:, None] * np.exp(-tau[None, :] / mu0)
        return (
            np.squeeze(2 * pi * (mu_arr_pos * weights_mu) @ u0_neg + direct_beam_scaled - direct_beam)[()],
            direct_beam,
        )

    return flux_up, flux_down
    

def Gauss_Legendre_quad(N, c=0, d=1):
    """Generates Gauss-Legendre quadrature weights and points for integration from c to d

    Args:
        N (int): Number of quadrature nodes
        c (float, optional): Start of integration interval
        d (float, optional): End of integration interval
        
    Returns:
        array: Quadrature points 
        array: Quadrature weights
        
    """
    mu_arr_pos, weights_mu = leggauss(N)

    return transform_interval(
        mu_arr_pos, c, d
    ),transform_weights(weights_mu, c, d)


def Clenshaw_Curtis_quad(Nphi, c=0, d=(2 * pi)):
    """Generates Clenshaw-Curtis quadrature weights and points for integration from c to d

    Args:
        Nphi (int): Number of quadrature nodes
        c (float, optional): Start of integration interval
        d (float, optional): End of integration interval
        
    Returns:
        array: Quadrature points 
        array: Quadrature weights
        
    """
    # Ensure that the number of nodes is odd and greater than 2
    assert Nphi > 2
    assert Nphi % 2 == 1

    Nphi -= 1  # The extra index corresponds to the point 0 which we will add later
    Nphi_pos = Nphi // 2
    phi_arr_pos = np.cos(pi * np.arange(Nphi_pos) / Nphi)
    phi_arr = np.hstack((-phi_arr_pos, 0, np.flip(phi_arr_pos)))
    diff = np.hstack((2, 2 / (1 - 4 * np.arange(1, Nphi_pos + 1) ** 2)))
    weights_phi_pos = sc.fft.idct(diff, type=1)
    weights_phi_pos[0] /= 2
    full_weights_phi = np.hstack((weights_phi_pos, np.flip(weights_phi_pos[:-1])))

    return transform_interval(
        phi_arr, c, d
    ), transform_weights(full_weights_phi, c, d)
    
 
def generate_FD_mat(Ntau, a, b):
    """Generates diagonal storage sparse first derivative matrix with second-order accuracy 
    on [a,b] with Ntau grid points. We use second order forward and backward differences at the edges

    Args:
        Ntau (int): Number of grid nodes
        a (float, optional): Start of diffentiation interval
        b (float, optional): End of diffentiation interval
        
    Returns:
        array: The differentiation grid
        sparse 2darray: The finite difference matrix
        
    """
    tau_arr = np.linspace(a, b, Ntau)
    h = tau_arr[1] - tau_arr[0]

    diagonal = np.ones(Ntau) / (2 * h)
    first_deriv = sc.sparse.diags(diagonal[:-1], 1, format = "lil")
    first_deriv.setdiag(-diagonal[:-1], -1)
    
    first_deriv[0, 0] = -3 / (2 * h)
    first_deriv[0, 1] = 2 / h
    first_deriv[0, 2] = -1 / (2 * h)
    first_deriv[-1, -1] = 3 / (2 * h)
    first_deriv[-1, -2] = -2 / h
    first_deriv[-1, -3] = 1 / (2 * h)
    
    return tau_arr, first_deriv.asformat("dia")
    
    
# The following function is exactly NumPy's `atleast_2d` function but altered
# to add dimensions to the back of the shape tuple rather than to the front.
# Documentation for `np.atleast_2d` taken from https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html but reformatted
def atleast_2d_append(*arys):
    """View inputs as arrays with at least two dimensions. Appends axes.

    Args:
        Ntau (tuple of array_likes): One or more array-like sequences.  
            Non-array inputs are converted to arrays.  
            Arrays that already have two or more dimensions are preserved.
        
    Returns:
        ndarray or list of ndarray: An array, or list of arrays, each with ``a.ndim >= 2``.
            Copies are avoided where possible, and views with two or more dimensions are returned.
        
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
    
    
    
    # Sample NumPy docstring
    """
    View inputs as arrays with at least two dimensions.
    Parameter(s)
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.
    Return(s)
    -------
    res, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.
        
    """