import numpy as np
import scipy as sc
from math import pi



def transform_interval(arr, c, d, a, b):
    """Affine transformation of an array from interval [a, b] to [c, d].

    Parameters
    ----------
    arr : array
        The 1D array to transform.
    c : float
        Lower bound of interval [c, d].
    d : float
        Upper bound of interval [c, d].
    a : float, optional
        Lower bound of interval [a, b].
    b : float, optional
        Upper bound of interval [a, b].

    Returns
    -------
    array
        The transformed 1D array.

    """
    return (arr - a) * (d - c) / (b - a) + c



def transform_weights(weights, c, d, a, b):
    """Transforms an array of quadrature weights from interval [a, b] to [c, d].

    Parameters
    ----------
    weights : array
        The weights to transform.
    c : float
        Lower bound of interval [c, d].
    d : float
        Upper bound of interval [c, d].
    a : float, optional
        Lower bound of interval [a, b].
    b : float, optional
        Upper bound of interval [a, b].

    Returns
    -------
    array
        The transformed weights.

    """
    return weights * (d - c) / (b - a)



def calculate_nu(mu, phi, mu_p, phi_p):
    """Calculates the cosine of the scattering angle ``nu`` 
    between incident angle ``(mu_p, phi_p)`` and scattering angle ``(mu, phi)``.

    Parameters
    ----------
    mu : array
        Cosine of outgoing polar angles.
    phi : array
        Outgoing azimuthal angles.
    mu_p : array
        Cosine of incident polar angles.
    phi_p : array
        Incident azimuthal angles.

    Returns
    -------
    nu : ndarray
        Cosine of scattering angles which axes capture variation with ``mu, phi, mu_p, phi_p`` respectively.

    """
    mu, phi, mu_p, phi_p = np.atleast_1d(mu, phi, mu_p, phi_p)
    nu = mu_p[None, None, :, None] * mu[:, None, None, None] + np.sqrt(1 - mu_p**2)[
        None, None, :, None
    ] * np.sqrt(1 - mu**2)[:, None, None, None] * np.cos(
        phi_p[None, None, None, :] - phi[None, :, None, None]
    )
    return np.squeeze(nu)



def Gauss_Legendre_quad(N, c=0, d=1):
    """Generates Gauss-Legendre quadrature zero points and weights for integration from c to d.

    Parameters
    ----------
    N : int (will be converted to int)
        Number of quadrature nodes.
    c : float, optional
        Lower bound of integration interval.
    d : float, optional
        Upper bound of integration interval.

    Returns
    -------
    array
        Quadrature zero points.
    array
        Quadrature weights.

    """
    mu_arr_pos, W = np.polynomial.legendre.leggauss(int(N))

    return transform_interval(mu_arr_pos, c, d, -1, 1), transform_weights(W, c, d, -1, 1)



def Clenshaw_Curtis_quad(Nphi, c=0, d=(2 * pi)):
    """Generates Clenshaw-Curtis quadrature zero points and weights for integration from c to d.

    Parameters
    ----------
    Nphi : int
        Number of quadrature nodes.
    c : float, optional
        Start of integration interval.
    d : float, optional
        End of integration interval.

    Returns
    -------
    array
        Quadrature zero points.
    array
        Quadrature weights.

    """
    # Ensure that the number of nodes is odd and greater than 2
    if not (Nphi > 2 and Nphi % 2 == 1):
        raise ValueError("The number of quadrature nodes must be odd and greater than 2.")

    Nphi -= 1  # The extra index corresponds to the point 0 which we will add later
    Nphi_pos = Nphi // 2
    phi_arr_pos = np.cos(pi * np.arange(Nphi_pos) / Nphi)
    phi_arr = np.concatenate([-phi_arr_pos, [0], np.flip(phi_arr_pos)])
    diff = np.insert(2 / (1 - 4 * np.arange(1, Nphi_pos + 1) ** 2), 0, 2)
    weights_phi_pos = sc.fft.idct(diff, type=1)
    weights_phi_pos[0] /= 2
    full_weights_phi = np.hstack((weights_phi_pos, np.flip(weights_phi_pos[:-1])))

    return transform_interval(phi_arr, c, d, -1, 1), transform_weights(full_weights_phi, c, d, -1, 1)



def generate_FD_mat(Ntau, a, b):
    """Generates a sparse first derivative central difference (second-order accuracy) 
    matrix on [a,b] in ``csr`` format with ``Ntau`` grid points.
    Second order forward or backward differences are used at the boundaries.

    Parameters
    ----------
    Nphi : int
        Number of grid nodes.
    a : float, optional
        Start of diffentiation interval.
    b : float, optional
        End of diffentiation interval.

    Returns
    -------
    array
        The differentiation grid.
    sparse 2darray
        The finite difference matrix.

    """
    tau_arr = np.linspace(a, b, Ntau)
    h = tau_arr[1] - tau_arr[0]

    diagonal = np.ones(Ntau) / (2 * h)
    first_deriv = sc.sparse.diags(diagonal[:-1], 1, format="lil")
    first_deriv.setdiag(-diagonal[:-1], -1)

    first_deriv[0, 0] = -3 / (2 * h)
    first_deriv[0, 1] = 2 / h
    first_deriv[0, 2] = -1 / (2 * h)
    first_deriv[-1, -1] = 3 / (2 * h)
    first_deriv[-1, -2] = -2 / h
    first_deriv[-1, -3] = 1 / (2 * h)

    return tau_arr, first_deriv.tocsr()
  


def atleast_2d_append(*arys):
    """View inputs as arrays with at least two dimensions. 
    Dimensions are added as necessary to the back of the shape tuple rather than to the front.
        
        This is exactly NumPy's ``numpy.atleast_2d`` function but altered to add dimensions to the back of the shape tuple rather than to the front.
        View the documentation for ``numpy.atleast_2d`` at https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html.

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
       
       
        
def generate_diff_act_flux_funcs(u0):
    """Generates the up and down diffuse actinic flux functions respectively.
    This is a use case of the ``u0`` function which is an output of ``pydisort``.
    The reclassification of delta-scaled actinic flux is automatically performed.
    The computation of actinic fluxes is similar to that of energetic fluxes
    and the latter is discussed in section 3.8 of the Comprehensive Documentation.

    Parameters
    ----------
    u0 : func
        Zeroth Fourier mode of the intensity.
        See the fourth "return" of the ``pydisort`` function.

    Returns
    -------
    Fp_act(tau) : function
        Actinic flux function with argument ``tau`` (type: array) for positive (upward) ``mu`` values.
        Returns the diffuse flux magnitudes (type: array).
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).
    Fm_act(tau) : function
        Actinic flux function with argument ``tau`` (type: array) for negative (downward) ``mu`` values.
        Returns the diffuse flux magnitudes (type: array).
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).

    """
    N = len(u0(0)) // 2
    GL_weights = Gauss_Legendre_quad(N)[1]

    # Note that the zeroth axis of the array u0(tau) captures variation with mu
    def flux_act_up(tau, is_antiderivative_wrt_tau=False, return_tau_arr=False):
        if return_tau_arr:
            (
                u0_cache, 
                tau_arr,
            ) = u0(tau, is_antiderivative_wrt_tau, True)
            return np.squeeze(2 * pi * GL_weights @ u0_cache[:N])[()], tau_arr
        else:
            return np.squeeze(2 * pi * GL_weights @ u0(tau, is_antiderivative_wrt_tau)[:N])[()]
        
    def flux_act_down_diffuse(tau, is_antiderivative_wrt_tau=False, return_tau_arr=False):
        if return_tau_arr:
            (
                u0_cache, 
                tau_arr,
                act_dscale_reclassification,
            ) = u0(tau, is_antiderivative_wrt_tau, True, _return_act_dscale_for_reclass=True)
            result_without_reclassification = 2 * pi * GL_weights @ u0_cache[N:]
            return np.squeeze(result_without_reclassification + act_dscale_reclassification)[()], tau_arr
        else:
            (
                u0_cache, 
                act_dscale_reclassification,
            ) = u0(tau, is_antiderivative_wrt_tau, False, _return_act_dscale_for_reclass=True)
            result_without_reclassification = 2 * pi * GL_weights @ u0_cache[N:]
            return np.squeeze(result_without_reclassification + act_dscale_reclassification)[()]
    
    return flux_act_up, flux_act_down_diffuse



def Planck(T, WVNM):
    """The Planck function for intensity leaving a blackbody surface,
    with units W / m^2 to match Stamnes' DISORT.

    Parameters
    ----------
    T : float or array
        Temperatures in kelvin.
    WVNM : float
        Wavenumber with units m^-1.

    Returns
    -------
    float or array
        Emitted power per unit area from a blackbody surface with units W / m^2.

    """
    T = np.atleast_1d(T)
    not_zeros_ind = T != 0
    
    results = np.zeros(len(T))
    if np.sum(not_zeros_ind) > 0:
        results[~not_zeros_ind] = 0
        
        # Coded in this way to prevent overflow
        expterm = np.exp(-100 * sc.constants.h * sc.constants.c * WVNM / (sc.constants.k * T[not_zeros_ind]))
        results[not_zeros_ind] = (2e8 * sc.constants.h * sc.constants.c**2 * WVNM**3 * expterm) / (1 - expterm)
        
    return np.squeeze(results)[()]
    
    

def blackbody_contrib_to_BCs(T, WVNMLO, WVNMHI, **kwargs):
    """Compute blackbody contribution to each BC (``b_pos`` or ``b_neg``), 
    i.e. the blackbody emission of that boundary, with units W / m^2.
    This convenience function is provided to help match the inputs for Stamnes' DISORT to those for PythonicDISORT.
    Users will have to manually adjust emissivities but ``PythonicDISORT.subroutines.generate_emissivity_from_BDRF``
    can help with that.
    
    Parameters
    ----------
    T : float or array
        Temperatures in kelvin.
    WVNMLO : float
        Lower bound of wavenumber interval with units m^-1. This variable is identically named in Stamnes' DISORT.
    WVNMHI : float
        Upper bound of wavenumber interval with units m^-1. This variable is identically named in Stamnes' DISORT.
    **kwargs
        Keyword arguments to pass to ``scipy.integrate.quad_vec``.
        
    Returns
    -------
    float or array
        The blackbody emission of each boundary with units W / m^2.
    """ 
    return np.squeeze(sc.integrate.quad_vec(lambda WVNM: Planck(T, WVNM), WVNMLO, WVNMHI, **kwargs)[0])



def linear_spline_coefficients(x, y, check_inputs=True):
    """Compute the coefficients of a linear spline interpolation.

    Parameters
    ----------
    x : array
        Array of `x` data points.
    y : array
        Array of `y` data points.

    Returns
    -------
    2darray
        The coefficients with axes (segment, ascending polynomial order) which is required by ``pydisort``.

    """
    # Input checks
    if check_inputs:
        if not len(x) > 1:
            raise ValueError("At least 2 points are required.")
        if not len(x) == len(y):
            raise ValueError("The number of x and y points must be equal.")
        if not np.all(np.diff(x) > 0):
            raise ValueError("The x values must be sorted in ascending order.")
    
    slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    intercepts = y[:-1] - slopes * x[:-1]
    
    return np.array((intercepts, slopes)).T



def generate_s_poly_coeffs(tau_arr, TEMPER, WVNMLO, WVNMHI, **kwargs):
    """Generate DISORT-equivalent ``s_poly_coeffs`` for input into ``pydisort``.
    This convenience function is provided to help match the inputs for Stamnes' DISORT to those for PythonicDISORT.
    Note that the coefficients will be multiplied by emissivity factors equal to ``1 - omega_arr`` inside
    `pydisort`, i.e. Kirchoff's law of thermal radiation is enforced, just like in Stamnes' DISORT. 
    
    Parameters
    ----------
    tau_arr : array or float
        Optical depth of the lower boundary of each atmospheric layer.
    TEMPER : array
        Temperature in kelvin at each boundary / interface from top to bottom.
        This variable is identically named in Stamnes' DISORT.
    WVNMLO : float
        Lower bound of wavenumber interval with units m^-1. This variable is identically named in Stamnes' DISORT.
    WVNMHI : float
        Upper bound of wavenumber interval with units m^-1. This variable is identically named in Stamnes' DISORT.
    **kwargs
        Keyword arguments to pass to ``scipy.integrate.quad_vec``.
        
    Returns
    -------
    s_poly_coeffs : 2darray
        Polynomial coefficients of isotropic internal sources.
        Each row pertains to an atmospheric layer (from top to bottom).
        The coefficients are arranged from lowest order term to highest.

    """
    tau_arr = np.atleast_1d(tau_arr)
    if not len(TEMPER) == len(tau_arr) + 1:
        raise ValueError(
            "Missing temperature specification at some boundaries / interfaces."
        )

    tau_arr_with_0 = np.insert(tau_arr, 0, 0)
    blackbody_emission_at_each_boundary = sc.integrate.quad_vec(
        lambda WVNM: Planck(TEMPER, WVNM), WVNMLO, WVNMHI, **kwargs
    )[0]

    return linear_spline_coefficients(
        tau_arr_with_0, blackbody_emission_at_each_boundary, check_inputs=False
    )




def generate_emissivity_from_BDRF(N, zeroth_BDRF_Fourier_mode):
    """Use Kirchoff's law of thermal radiation to determine the (directional) emissivity 
    of the surface given the zeroth Fourier mode of the Bi-Directional Reflectance Function (BDRF).
    This computation is automatic and internal to Stamnes' DISORT.
    This function supplements ``PythonicDISORT.subroutines.blackbody_contrib_to_BCs``.
    
    Parameters
    ----------
    N : int
        Number of upper hemisphere quadrature nodes. Equal to ``NQuad // 2``.
    zeroth_BDRF_Fourier_mode : function or float
        Zeroth BDRF Fourier mode with arguments ``mu, -mu_p`` of type array
        and which output has the same dimensions as the outer product of the two arrays.
        A scalar input represents a constant function and in that case the output 
        is simply one minus the scalar input.

    Returns
    -------
    array or float
        Emissivity for the blackbody contribution to the lower boundary source ``b_neg``.
    """
    if np.isscalar(zeroth_BDRF_Fourier_mode):
        return 1 - zeroth_BDRF_Fourier_mode
    else:
        mu_arr_pos, W = Gauss_Legendre_quad(N)
        return (
            1 - 2 * zeroth_BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos) * mu_arr_pos[None, :] @ W
        )



def cache_BDRF_Fourier_modes(N, mu0, BDRF_Fourier_modes):
    """If the same BDRF, number of streams and ``mu0`` will be used repeatedly,
    consider using this function to cache ``BDRF_Fourier_modes`` and instead input
    ``cached_BDRF_Fourier_modes`` into ``pydisort``.

    Parameters
    ----------
    N : int
        Number of upper hemisphere quadrature nodes. Equal to ``NQuad // 2``.
    mu0 : float
        Cosine of polar angle of the incident beam.
    BDRF_Fourier_modes : list of functions and scalars
        BDRF Fourier modes, each a scalar representing a constant function, 
        or a function with arguments ``mu, -mu_p`` of type array
        which output has the same dimensions as the outer product of the two arrays.

    Returns
    -------
    cached_BDRF_Fourier_modes : list of functions
        Cached BDRF Fourier modes, each a function with arguments ``mu, -mu_p`` of type array
        and which output is a scalar (if the Fourier mode was a scalar) 
        or has the same dimensions as the outer product of the two arrays.
    """
    NBDRF = len(BDRF_Fourier_modes)
    mu_arr_pos = Gauss_Legendre_quad(N)[0]

    BDRF_Fourier_modes_evaluated = [
        (
            None
            if np.isscalar(BDRF_Fourier_modes[m])
            else BDRF_Fourier_modes[m](mu_arr_pos, np.append(mu_arr_pos, mu0))
        )
        for m in range(NBDRF)
    ]

    cached_BDRF_Fourier_modes = [
        (
            lambda mu, neg_mup, m=m: BDRF_Fourier_modes[m]
            if np.isscalar(BDRF_Fourier_modes[m])
            else (
                BDRF_Fourier_modes_evaluated[m][:, [-1]]
                if len(neg_mup) == 1
                else BDRF_Fourier_modes_evaluated[m][:, :-1]
            )
        )
        for m in range(NBDRF)
    ]

    return cached_BDRF_Fourier_modes



def affine_transform_poly_coeffs(poly_coeffs, a_arr, b_arr):
    """Given a polynomial C_0 + C_1 x + ... C_n x^n and the affine transformation
    y -> ax + b, determine the coefficients of the polynomial D_0 + D_1 y + ... D_n y^n.

    Parameters
    ----------
    poly_coeffs : 2darray
        Array with columns [C_0, C_1, ..., C_n]. 
        The rows correspond to different affine transformations.
    a_arr : array
        Scale factors.
    b_arr : array
        Translations.

    Returns
    -------
    transformed_poly_coeffs : 2darray
        Array with columns [D_0, D_1, ..., D_n]. 
        The rows correspond to different affine transformations.
    """
    if np.any(a_arr) == 0:
        raise ValueError("The scale factors must be non-zero.") 
    
    Ntransformations, Ncoeffs = np.shape(poly_coeffs)
    Ncoeffs_arange = np.arange(Ncoeffs)
    pascal = sc.linalg.pascal(Ncoeffs, kind="upper")
    pow_for_shifts = np.where(
        pascal == 0, 0, Ncoeffs_arange[None, :] - Ncoeffs_arange[:, None]
    )

    T = (
        pascal[None, :, :]
        * (1 / a_arr)[:, None, None] ** (Ncoeffs_arange[None, None, :])
        * (-b_arr[:, None, None]) ** pow_for_shifts[None, :, :]
    )  # Transformation matrix for the polynomial coefficients
    
    return np.einsum("lij, lj -> li", T, poly_coeffs, optimize=True)



def interpolate(u):
    """Polynomial (Barycentric) interpolation with respect to ``mu``. The output 
    is a function that is continuous and variable in all three arguments: ``mu``, ``tau`` and ``phi``.
    Discussed in sections 3.7 and 6.3 of the Comprehensive Documentation.

    Parameters
    ----------
    u or u0 : function
        Non-interpolated intensity function as given by ``pydisort``.

    Returns
    -------
    u_interpol : function
        Intensity function with arguments ``(mu, tau, phi)`` (for ``u``) 
        or ``(mu, tau)`` (for ``u0``) each an array or scalar.
        Returns an ndarray with axes corresponding to variation with each argument in the same order.
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_Fourier_error = True`` (defaults to ``False``) to return the 
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).

    """
    
    N = len(u(0, 0)) // 2
    mu_arr_pos = Gauss_Legendre_quad(N)[0]
    
    u_pos_interpol = sc.interpolate.BarycentricInterpolator(mu_arr_pos)
    u_neg_interpol = sc.interpolate.BarycentricInterpolator(-mu_arr_pos)
    
    if u.__code__.co_argcount == 5: # Function is u instead of u0
        def u_interpol(mu, tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False, return_tau_arr=False):
            if not np.all(np.abs(mu) <= 1):
                raise ValueError("mu values must be between -1 and 1.")
            
            mu = np.atleast_1d(mu)

            if return_Fourier_error or return_tau_arr:
                u_outputs = u(tau, phi, is_antiderivative_wrt_tau, return_Fourier_error, return_tau_arr)
                u_cache = u_outputs[0]
            else:
                u_cache = u(tau, phi, is_antiderivative_wrt_tau)
            
            results = np.empty((len(mu),) + np.shape(u_cache)[1:])
            mask_pos = mu > 0
            mask_else = ~mask_pos
            
            if np.any(mask_pos):
                u_pos_interpol.set_yi(u_cache[:N])
                results[mask_pos] = u_pos_interpol(mu[mask_pos])
            if np.any(mask_else):
                u_neg_interpol.set_yi(u_cache[N:])
                results[mask_else] = u_neg_interpol(mu[mask_else])
            
            if return_Fourier_error or return_tau_arr:
                return (np.squeeze(results)[()],) + u_outputs[1:]
            else:
                return np.squeeze(results)[()]
    
    elif u.__code__.co_argcount == 4: # Function is u0 instead of u
        def u_interpol(mu, tau, is_antiderivative_wrt_tau=False, return_Fourier_error=False, return_tau_arr=False):
            if not np.all(np.abs(mu) <= 1):
                raise ValueError("mu values must be between -1 and 1.")
            
            mu = np.atleast_1d(mu)

            if return_Fourier_error or return_tau_arr:
                u_outputs = u(tau, is_antiderivative_wrt_tau, return_Fourier_error, return_tau_arr)
                u_cache = u_outputs[0]
            else:
                u_cache = u(tau, is_antiderivative_wrt_tau)
            
            results = np.empty((len(mu),) + np.shape(u_cache)[1:])
            mask_pos = mu > 0
            mask_else = ~mask_pos
            
            if np.any(mask_pos):
                u_pos_interpol.set_yi(u_cache[:N])
                results[mask_pos] = u_pos_interpol(mu[mask_pos])
            if np.any(mask_else):
                u_neg_interpol.set_yi(u_cache[N:])
                results[mask_else] = u_neg_interpol(mu[mask_else])
            
            if return_Fourier_error or return_tau_arr:
                return (np.squeeze(results)[()],) + u_outputs[1:]
            else:
                return np.squeeze(results)[()]
    else:
        raise ValueError("This subroutine can only interpolate u or u0")
    
    return u_interpol
         


def to_diag_ordered_form(A, Nsuperdiags, Nsubdiags):
    """
    Convert some matrix A to the diagonal ordered form required by ``scipy.linalg.solve_banded``.

    Parameters
    ----------
    A : 2darray
        The square matrix to be converted.
    Nsuperdiags : int
        The number of super-diagonals.
    Nsubdiags : int
        The number of sub-diagonals.

    Returns
    -------
    2darray
        The diagonal ordered form matrix as required by ``scipy.linalg.solve_banded``.
    """
    n = A.shape[0]
    indices = np.arange(n)

    return np.concatenate(
        [
            A[
                indices - np.arange(Nsuperdiags, -1, -1)[:, None],
                indices[None, :],
            ],
            A[
                indices - np.arange(n - 1, n - Nsubdiags - 1, -1)[:, None],
                indices[None, :],
            ],
        ],
        axis=0,
    )


    
def _mathscr_v(tau,                              # Input optical depths
                l,                               # Layer index of each input optical depth
                Nscoeffs,                        # Number of isotropic source polynomial coefficients
                s_poly_coeffs,                   # Polynomial coefficients of isotropic source
                G,                               # Eigenvector matrices
                K,                               # Eigenvalues
                G_inv,                           # Inverse of eigenvector matrix
                mu_arr,                          # Quadrature nodes for both hemispheres
                is_antiderivative_wrt_tau=False, # Switch to an antiderivative of the function?
                autograd_compatible=False,       # Should the output functions be compatible with autograd?
                ):
    """Particular solution for isotropic internal sources.
    Refer to section 3.6.1 of the Comprehensive Documentation.
    It has many seemingly redundant arguments to maximize 
    precomputation in the ``_assemble_intensity_and_fluxes`` 
    and ``_solve_for_coeffs`` functions which call it.
    
    Arguments of _mathscr_v
    |          Variable             |                   Shape                   |
    | ----------------------------- | ----------------------------------------- |
    | ``tau``                       | ``Ntau``                                  |
    | ``l``                         | ``Ntau``                                  |
    | ``Nscoeffs``                  | scalar                                    |
    | ``s_poly_coeffs``             | ``NLayers x Nscoeffs``                    |
    | ``G``                         | ``NLayers<= x NQuad x NQuad``             |
    | ``K``                         | ``NLayers<= x NQuad``                     |
    | ``G_inv``                     | ``NLayers<= x NQuad x NQuad`` or ``None`` |
    | ``mu_arr``                    | ``NQuad``                                 |
    | ``is_antiderivative_wrt_tau`` | boolean                                   |
    | ``autograd_compatible``       | boolean                                   |
    
    Notable internal variables of _mathscr_v
    |     Variable     |                Shape                  | 
    | ---------------- | ------------------------------------- |
    | i_arr            | ``Nscoeffs``                          |
    | i_arr_repeat     | ``Nscoeffs*(Nscoeffs+1)/2``           | Using triangular number formula
    | j_arr            | ``Nscoeffs*(Nscoeffs+1)/2``           | Using triangular number formula
    | s_poly_coeffs_nj | ``NLayers x Nscoeffs*(Nscoeffs+1)/2`` |
    | OUTPUT           | ``NQuad x Ntau``                      |
    """
    n = Nscoeffs - 1
    
    if autograd_compatible:
        import autograd.numpy as np
    
        def mathscr_b(i):
            """
            Notable internal variables of mathscr_b
            |     Variable     |                  Shape                  |
            | ---------------- | --------------------------------------- |
            | j_arr            | ``i + 1``                               |
            | s_poly_coeffs_nj | ``i + 1``                               |
            | OUTPUT           | ``Nscoeffs x (i + 1) x NQuad``          |
            """
        
            j_arr = np.arange(i + 1)
            s_poly_coeffs_nj = s_poly_coeffs[:, n - j_arr]
            return np.sum(
                (sc.special.factorial(n - j_arr) / sc.special.factorial(n - i))[None, None, :]
                * K[:, :, None] ** -(i - j_arr + 1)[None, None, :]
                * s_poly_coeffs_nj[:, None, :],
                axis=-1,
            )
        mathscr_v_coeffs = np.array(list(map(mathscr_b, range(Nscoeffs))))
    else:
        import numpy as np
        
        shape = np.shape(K)
        i_arr = np.arange(Nscoeffs)
        i_arr_repeat = np.repeat(i_arr, i_arr + 1)
        j_arr = np.concatenate([i_arr[:i] for i in range(1, Nscoeffs + 1)])
        s_poly_coeffs_nj = s_poly_coeffs[:, n - j_arr]

        mathscr_v_coeffs = np.zeros((Nscoeffs, shape[0], shape[1]))
        np.add.at(
            mathscr_v_coeffs,
            i_arr_repeat,
            np.moveaxis(
                (sc.special.factorial(n - j_arr) / sc.special.factorial(n - i_arr_repeat))[
                    None, None, :
                ]
                * K[:, :, None] ** -(i_arr_repeat - j_arr + 1)[None, None, :]
                * s_poly_coeffs_nj[:, None, :],
                -1,
                0,
            ),
        )
    
    powers = np.arange(Nscoeffs - 1, -1, -1)[None, :]
    if is_antiderivative_wrt_tau:
        tau_poly = tau[:, None] ** (powers + 1) / (powers + 1)
    else:
        tau_poly = tau[:, None] ** powers
    
    return np.einsum(
        "tik, tc, ctk -> it",
        G[l, :, :],
        tau_poly,
        (mathscr_v_coeffs * (G_inv @ (1 / mu_arr))[None, :, :])[:, l, :],
        optimize=True,
    )
    


def _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down, u=None):
    """Performs pointwise comparisons between results from Stamnes' DISORT,
    which are stored in ``.npz`` files, against results from PythonicDISORT. Used in our PyTests.
    See the ``*_test`` Jupyter Notebooks to see how this function is used and the arguments that go into it.
    
    """
    # Load saved results from Stamnes' DISORT
    flup = results["flup"]
    rfldn = results["rfldn"]
    rfldir = results["rfldir"]
    if u is not None:
        uu = results["uu"]

    # Load comparison points
    tau_test_arr = results["tau_test_arr"]
    if u is not None:
        phi_arr = results["phi_arr"]

    # Perform and print the comparisons
    # --------------------------------------------------------------------------------------------------
    print("Max pointwise differences")
    print()

    # Upward (diffuse) fluxes
    print("Upward (diffuse) fluxes")
    diff_flux_up = np.abs(flup - flux_up(tau_test_arr))
    ratio_flux_up = np.divide(
        diff_flux_up,
        flup,
        out=np.zeros_like(diff_flux_up),
        where=(flup != 0),
    )
    print("Difference =", np.max(diff_flux_up))
    print("Difference ratio =", np.max(ratio_flux_up))
    print()

    # Downward (diffuse) fluxes
    print("Downward (diffuse) fluxes")
    diff_flux_down_diffuse = np.abs(rfldn - flux_down(tau_test_arr)[0])
    ratio_flux_down_diffuse = np.divide(
        diff_flux_down_diffuse,
        rfldn,
        out=np.zeros_like(diff_flux_down_diffuse),
        where=(rfldn != 0),
    )
    print("Difference =", np.max(diff_flux_down_diffuse))
    print(
        "Difference ratio =",
        np.max(ratio_flux_down_diffuse),
    )
    print()

    # Direct (downward) fluxes
    print("Direct (downward) fluxes")
    diff_flux_down_direct = np.abs(rfldir - flux_down(tau_test_arr)[1])
    ratio_flux_down_direct = np.divide(
        diff_flux_down_direct,
        rfldir,
        out=np.zeros_like(diff_flux_down_direct),
        where=(rfldir != 0),
    )
    print("Difference =", np.max(diff_flux_down_direct))
    print(
        "Difference ratio =",
        np.max(ratio_flux_down_direct),
    )
    print()
    
    if u is not None:
        # Intensity
        u_cache = u(tau_test_arr, phi_arr)[reorder_mu].reshape(np.shape(uu))
        diff = np.abs(uu - u_cache)[mu_to_compare]
        diff_ratio = np.divide(
            diff,
            uu[mu_to_compare],
            out=np.zeros_like(diff),
            where=(uu[mu_to_compare] != 0),
        )
        max_diff_tau_index = np.argmax(np.max(np.max(diff, axis=0), axis=1))
        max_ratio_tau_index = np.argmax(np.max(np.max(diff_ratio, axis=0), axis=1))
        
        diff_tau_pt = tau_test_arr[max_diff_tau_index]
        ratio_tau_pt = tau_test_arr[max_ratio_tau_index]
        print("Intensities")
        print()
        print("At tau = " + str(diff_tau_pt))
        print("Max pointwise difference =", np.max(diff[:, max_diff_tau_index, :]))
        print("At tau = " + str(ratio_tau_pt))
        print("Max pointwise difference ratio =", np.max(diff_ratio[:, max_ratio_tau_index, :]))
        print()

        return (
            diff_flux_up,
            ratio_flux_up,
            diff_flux_down_diffuse,
            ratio_flux_down_diffuse,
            diff_flux_down_direct,
            ratio_flux_down_direct,
            diff,
            diff_ratio,
        )
    
    else:
        return (
            diff_flux_up,
            ratio_flux_up,
            diff_flux_down_diffuse,
            ratio_flux_down_diffuse,
            diff_flux_down_direct,
            ratio_flux_down_direct,
        )