import numpy as np
import scipy as sc
from math import pi
import warnings


def prepend(arr, arr_len, value):
    """Prepend value to array. 
    Faster than ``np.insert(arr, 0, value)``.

    Parameters
    ----------
    arr : array
        The 1D array to which to prepend.
    arr_len : int
        Length of original array (``arr``).
    value : float
        Value to prepend.

    Returns
    -------
    array
        Array of length ``arr_len + 1`` with ``value`` prepended.

    """
    new_arr = np.empty(arr_len + 1)
    new_arr[1:] = arr
    new_arr[0] = value
    return new_arr



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
    a : float
        Lower bound of interval [a, b].
    b : float
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
    a : float
        Lower bound of interval [a, b].
    b : float
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
    diff = prepend(2 / (1 - 4 * np.arange(1, Nphi_pos + 1) ** 2), Nphi_pos, 2)
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
    Ntau : int
        Number of grid nodes.
    a : float
        Start of diffentiation interval.
    b : float
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
        See the fourth "return" of the ``pydisort`` function
        (which may have been called with ``interpolate_mu=True``).

    Returns
    -------
    Fp_act(tau) : function
        Actinic flux function with argument ``tau`` (type: array) for positive (upward) ``mu`` values.
        Returns the diffuse flux magnitudes (type: array).
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        Carries ``tau_arr`` as an attribute.
    Fm_act(tau) : function
        Actinic flux function with argument ``tau`` (type: array) for negative (downward) ``mu`` values.
        Returns the diffuse flux magnitudes (type: array).
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        Carries ``tau_arr`` as an attribute.

    """
    u0 = getattr(u0, "_at_quad_nodes", u0)
    N = len(u0(0)) // 2
    GL_weights = Gauss_Legendre_quad(N)[1]

    # Note that the zeroth axis of the array u0(tau) captures variation with mu
    def flux_act_up(tau, tau_derivative_order=0):
        return np.squeeze(2 * pi * GL_weights @ u0(tau, tau_derivative_order)[:N])[()]

    def flux_act_down_diffuse(tau, tau_derivative_order=0):
        # `u0` is evaluated first and it validates `tau` and `tau_derivative_order` (no need to repeat the validation here)
        result_without_reclassification = 2 * pi * GL_weights @ u0(tau, tau_derivative_order)[N:]
        act_dscale_reclassification = u0._act_dscale_reclassification(tau, tau_derivative_order)
        return np.squeeze(result_without_reclassification + act_dscale_reclassification)[()]

    flux_act_up.tau_arr = flux_act_down_diffuse.tau_arr = u0.tau_arr
    return flux_act_up, flux_act_down_diffuse



def Planck(T, WVNM):
    """The Planck function for intensity leaving a blackbody surface,
    with units W / (m^2 sr cm^-1), i.e. per unit wavenumber. Its integral over wavenumber
    (see ``blackbody_contrib_to_BCs``) has units W / (m^2 sr), which Stamnes' DISORT labels W / m^2.

    Parameters
    ----------
    T : float or array
        Temperatures in kelvin.
    WVNM : float
        Wavenumber with units cm^-1 (matching Stamnes' DISORT convention).

    Returns
    -------
    float or array
        Blackbody intensity per unit wavenumber with units W / (m^2 sr cm^-1).

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
    i.e. the blackbody emission of that boundary, with units W / (m^2 sr).
    This convenience function is provided to help match the inputs for Stamnes' DISORT to those for PythonicDISORT.
    Users will have to manually adjust emissivities but ``PythonicDISORT.subroutines.generate_emissivity_from_BDRF``
    can help with that.

    Parameters
    ----------
    T : float or array
        Temperatures in kelvin.
    WVNMLO : float
        Lower bound of wavenumber interval with units cm^-1. This variable is identically named in Stamnes' DISORT.
    WVNMHI : float
        Upper bound of wavenumber interval with units cm^-1. This variable is identically named in Stamnes' DISORT.
    **kwargs
        Keyword arguments to pass to ``scipy.integrate.quad_vec``.

    Returns
    -------
    float or array
        The blackbody emission of each boundary with units W / (m^2 sr).
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
    `pydisort`, i.e. Kirchhoff's law of thermal radiation is enforced, just like in Stamnes' DISORT.

    Parameters
    ----------
    tau_arr : array or float
        Optical depth of the lower boundary of each atmospheric layer.
    TEMPER : array
        Temperature in kelvin at each boundary / interface from top to bottom.
        This variable is identically named in Stamnes' DISORT.
    WVNMLO : float
        Lower bound of wavenumber interval with units cm^-1. This variable is identically named in Stamnes' DISORT.
    WVNMHI : float
        Upper bound of wavenumber interval with units cm^-1. This variable is identically named in Stamnes' DISORT.
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

    tau_arr_with_0 = prepend(tau_arr, len(tau_arr), 0)
    blackbody_emission_at_each_boundary = sc.integrate.quad_vec(
        lambda WVNM: Planck(TEMPER, WVNM), WVNMLO, WVNMHI, **kwargs
    )[0]

    return linear_spline_coefficients(
        tau_arr_with_0, blackbody_emission_at_each_boundary, check_inputs=False
    )




def generate_emissivity_from_BDRF(N, zeroth_BDRF_Fourier_mode):
    """Use Kirchhoff's law of thermal radiation to determine the (directional) emissivity
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
        Emissivity for the blackbody contribution to the bottom-boundary source ``b_pos``.
    """
    if np.isscalar(zeroth_BDRF_Fourier_mode):
        return 1 - zeroth_BDRF_Fourier_mode
    else:
        mu_arr_pos, W = Gauss_Legendre_quad(N)
        return (
            1 - 2 * zeroth_BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos) * mu_arr_pos[None, :] @ W
        )



def cache_BDRF_Fourier_modes(N, BDRF_Fourier_modes, mu0=0):
    """If the same BDRF and number of streams will be used repeatedly,
    consider using this function to cache ``BDRF_Fourier_modes`` and pass
    ``cached_BDRF_Fourier_modes`` to ``pydisort``. It is also possible
    to cache ``BDRF_Fourier_modes`` with respect to a fixed ``mu0``.
    The cache is of the values at the quadrature nodes, which is all that ``pydisort`` requires
    unless ``interpolate_mu=True``; the cached modes evaluate ``BDRF_Fourier_modes`` at any other ``mu``.

    Parameters
    ----------
    N : int
        Number of upper hemisphere quadrature nodes. Equal to ``NQuad // 2``.
    BDRF_Fourier_modes : list of functions and scalars
        BDRF Fourier modes, each a scalar representing a constant function, 
        or a function with arguments ``mu, -mu_p`` of type array
        which output has the same dimensions as the outer product of the two arrays.
    mu0 : float, optional
        Cosine of polar angle of the incident beam.

    Returns
    -------
    cached_BDRF_Fourier_modes : list of functions and scalars
        Cached BDRF Fourier modes, each a scalar (if the Fourier mode was a scalar)
        or a function with arguments ``mu, -mu_p`` of type array
        which output has the same dimensions as the outer product of the two arrays.
    """
    mu_arr_pos = Gauss_Legendre_quad(N)[0]
    cache_mu0 = 0 < mu0 <= 1
    if not cache_mu0:
        warnings.warn("No caching with respect to `mu0`.")

    def cache(BDRF_Fourier_mode):
        if np.isscalar(BDRF_Fourier_mode):
            return BDRF_Fourier_mode
        at_nodes = BDRF_Fourier_mode(mu_arr_pos, mu_arr_pos)
        at_mu0 = BDRF_Fourier_mode(mu_arr_pos, np.array([mu0])) if cache_mu0 else None

        def cached_BDRF_Fourier_mode(mu, neg_mup):
            # The cached values if the arguments are those which were cached, else evaluate the mode
            if np.array_equal(mu, mu_arr_pos):
                if np.array_equal(neg_mup, mu_arr_pos):
                    return at_nodes
                if cache_mu0 and np.array_equal(neg_mup, [mu0]):
                    return at_mu0
            return BDRF_Fourier_mode(mu, neg_mup)

        return cached_BDRF_Fourier_mode

    return [cache(BDRF_Fourier_mode) for BDRF_Fourier_mode in BDRF_Fourier_modes]



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
    if np.any(a_arr == 0):
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

    return np.einsum("lij, lj -> li", T, poly_coeffs)



# The function is preserved below exactly as it was in version 1.9, commented out. 
# It relied on the attributes `_mu_arr_pos` and `_NT_data` of the intensity functions, which no longer exist.
#
# def interpolate(u, NT_cor=None):
#     """Polynomial (Barycentric) interpolation with respect to ``mu``. The output
#     is a function that is continuous and variable in all three arguments: ``mu``, ``tau`` and ``phi``.
#     Discussed in sections 3.7 and 6.3 of the Comprehensive Documentation.
#
#     Parameters
#     ----------
#     u or u0 : function
#         Non-interpolated intensity function as given by ``pydisort``.
#         Functions from ``pydisort`` carry their quadrature nodes as the attribute ``_mu_arr_pos``.
#         Any other function with the same arguments is instead evaluated once, at ``tau = 0``
#         (and ``phi = 0``), to determine ``NQuad``, from which the nodes are recomputed.
#     NT_cor : optional, None, bool or str
#         Nakajima-Tanaka (NT) corrections flag which overrides the ``NT_cor`` argument
#         that was passed to ``pydisort``. Only supported when interpolating the full
#         intensity function ``u`` as output by ``pydisort`` (not ``u0``). The options are:
#
#         - ``None`` (default): no override; interpolate ``u`` exactly as given.
#         - ``False`` or ``"off"``: interpolate the uncorrected, potentially delta-M scaled intensity,
#           even if ``pydisort`` was run with ``NT_cor=True``.
#         - ``True`` or ``"quad"``: apply the NT corrections at the quadrature points and
#           interpolate the corrected intensity; equivalent to interpolating ``u`` from
#           ``pydisort`` with ``NT_cor=True``.
#         - ``"eval"``: interpolate the uncorrected intensity and add the NT corrections
#           evaluated directly at each ``mu`` evaluation point (like Stamnes' DISORT).
#           This is generally more accurate than ``"quad"`` and is also cheaper when there
#           are fewer evaluation points than quadrature points, but it requires
#           ``|mu| >= 1e-8``.
#
#         If ``pydisort`` determined that NT corrections are not applicable (they require a
#         beam source, delta-M scaling, scattering and a truncated phase function), the
#         ``True / "quad"`` and ``"eval"`` options reduce to ``"off"``. The string options
#         ``"quad"`` and ``"eval"`` issue a warning when this happens, whereas the boolean
#         ``True`` reduces silently, mirroring how ``pydisort`` itself behaves when
#         ``NT_cor=True``.
#
#     Returns
#     -------
#     u_interpol : function
#         Intensity function with arguments ``(mu, tau, phi)`` (for ``u``)
#         or ``(mu, tau)`` (for ``u0``) each an array or scalar.
#         Returns an ndarray with axes corresponding to variation with each argument in the same order.
#         Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
#         to switch to an antiderivative of this function with respect to ``tau``.
#         Pass ``return_Fourier_error = True`` (defaults to ``False``; only for ``u``) to return the
#         Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
#         Carries ``tau_arr`` as an attribute if ``u`` does.
#
#     """
#     # --------------------------------------------------------------------------------------------------
#     add_corrections_at = None
#     if NT_cor is not None:
#         NT_mode = {False: "off", "off": "off", True: "quad", "quad": "quad", "eval": "eval"}.get(NT_cor)
#         if NT_mode is None:
#             raise ValueError(
#                 '`NT_cor` must be one of None, False/"off", True/"quad", or "eval".'
#             )
#
#         NT_data = getattr(u, "_NT_data", None)
#         if NT_data is None:
#             raise ValueError(
#                 "The `NT_cor` options of this subroutine require the full intensity function `u`"
#                 + " as output by `pydisort` (in particular, they are unsupported for `u0`)."
#             )
#         corrections_available = NT_data["corrections_available"]
#         NT_corrections_at_mu = NT_data["corrections_at_mu"]
#
#         # The string options explicitly request corrections, so warn if they are inapplicable;
#         # the boolean options instead mirror `pydisort`'s own silent behavior (see the docstring)
#         if NT_cor in ("quad", "eval") and not corrections_available:
#             warnings.warn(
#                 'NT corrections were requested through `interpolate` (`NT_cor="'
#                 + NT_cor
#                 + '"`) but are not applicable to this `pydisort` solution: they require'
#                 + " a beam source (`I0 > 0`), delta-M scaling (`f_arr > 0`), scattering"
#                 + " (`omega_arr > 0`) and a truncated phase function (`NLeg < NLeg_all`)."
#                 + " Proceeding without corrections."
#             )
#
#         # Interpolate the uncorrected `u_star` and add any corrections to it, unless `u` is already
#         # corrected at the quadrature points as requested (`pydisort` was run with `NT_cor=True`)
#         if not (NT_mode == "quad" and NT_data["NT_cor_applied"]):
#             if NT_data["NT_cor_applied"]:
#                 u = NT_data["u_star"]
#             if NT_mode != "off" and corrections_available:
#                 add_corrections_at = NT_mode
#     # --------------------------------------------------------------------------------------------------
#
#     # Functions from `pydisort` carry their upper hemisphere quadrature nodes as `_mu_arr_pos`;
#     # only for other functions are the nodes recomputed, after an evaluation to find `NQuad`
#     mu_arr_pos = getattr(u, "_mu_arr_pos", None)
#     if mu_arr_pos is None:
#         # A bit of spaghetti code here, but this works even if `u` is actually `u0` because
#         # in that case the second argument is `is_antiderivative_wrt_tau`.
#         mu_arr_pos = Gauss_Legendre_quad(len(u(0, 0)) // 2)[0]
#     N = len(mu_arr_pos)
#
#     if add_corrections_at == "quad":
#         mu_arr = np.concatenate([mu_arr_pos, -mu_arr_pos])
#
#     u_pos_interpol = sc.interpolate.BarycentricInterpolator(mu_arr_pos)
#     u_neg_interpol = sc.interpolate.BarycentricInterpolator(-mu_arr_pos)
#
#     # Value-keyed cache for the most recent (tau, phi, is_antideriv) -> u_cache.
#     # Hits when the user makes multiple `mu` calls with the same (tau, phi); otherwise re-evaluates `u`.
#     # Only used on the simple-return path (skipped when the Fourier error is requested).
#     _cache = [None, None, None, None]  # [tau_copy, phi_copy, is_antideriv, u_cache]
#
#     # Identity tokens for the data most recently loaded into each interpolator;
#     # they skip the redundant `set_yi` when the interpolator already holds `u_cache` (cache hits)
#     _yi_token = [None, None]
#
#     argcount = u.__code__.co_argcount
#     if argcount == 4: # Function is `u` instead of `u0`
#         def u_interpol(mu, tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False):
#             if not np.all(np.abs(mu) <= 1):
#                 raise ValueError("mu values must be between -1 and 1.")
#
#             mu = np.atleast_1d(mu)
#
#             if return_Fourier_error:
#                 u_cache, Fourier_error = u(tau, phi, is_antiderivative_wrt_tau, True)
#                 if add_corrections_at == "quad":
#                     u_cache = u_cache + NT_corrections_at_mu(
#                         mu_arr, tau, phi, is_antiderivative_wrt_tau
#                     ).reshape(np.shape(u_cache))
#             elif (_cache[3] is not None and is_antiderivative_wrt_tau == _cache[2]
#                   and np.array_equal(tau, _cache[0]) and np.array_equal(phi, _cache[1])):
#                 u_cache = _cache[3]
#             else:
#                 u_cache = u(tau, phi, is_antiderivative_wrt_tau)
#                 if add_corrections_at == "quad":
#                     u_cache = u_cache + NT_corrections_at_mu(
#                         mu_arr, tau, phi, is_antiderivative_wrt_tau
#                     ).reshape(np.shape(u_cache))
#                 _cache[:] = [np.array(tau, copy=True), np.array(phi, copy=True),
#                              is_antiderivative_wrt_tau, u_cache]
#
#             results = np.empty((len(mu),) + np.shape(u_cache)[1:])
#             mask_pos = mu > 0
#             mask_else = ~mask_pos
#
#             if np.any(mask_pos):
#                 if _yi_token[0] is not u_cache:
#                     u_pos_interpol.set_yi(u_cache[:N])
#                     _yi_token[0] = u_cache
#                 results[mask_pos] = u_pos_interpol(mu[mask_pos])
#             if np.any(mask_else):
#                 if _yi_token[1] is not u_cache:
#                     u_neg_interpol.set_yi(u_cache[N:])
#                     _yi_token[1] = u_cache
#                 results[mask_else] = u_neg_interpol(mu[mask_else])
#
#             if add_corrections_at == "eval":
#                 results += NT_corrections_at_mu(
#                     mu, tau, phi, is_antiderivative_wrt_tau
#                 ).reshape(np.shape(results))
#
#             if return_Fourier_error:
#                 return np.squeeze(results)[()], Fourier_error
#             else:
#                 return np.squeeze(results)[()]
#
#     elif argcount == 2: # Function is u0 instead of u
#         def u_interpol(mu, tau, is_antiderivative_wrt_tau=False):
#             if not np.all(np.abs(mu) <= 1):
#                 raise ValueError("mu values must be between -1 and 1.")
#
#             mu = np.atleast_1d(mu)
#
#             if (_cache[3] is not None and is_antiderivative_wrt_tau == _cache[2]
#                   and np.array_equal(tau, _cache[0])):
#                 u_cache = _cache[3]
#             else:
#                 u_cache = u(tau, is_antiderivative_wrt_tau)
#                 _cache[:] = [np.array(tau, copy=True), None, is_antiderivative_wrt_tau, u_cache]
#
#             results = np.empty((len(mu),) + np.shape(u_cache)[1:])
#             mask_pos = mu > 0
#             mask_else = ~mask_pos
#
#             if np.any(mask_pos):
#                 if _yi_token[0] is not u_cache:
#                     u_pos_interpol.set_yi(u_cache[:N])
#                     _yi_token[0] = u_cache
#                 results[mask_pos] = u_pos_interpol(mu[mask_pos])
#             if np.any(mask_else):
#                 if _yi_token[1] is not u_cache:
#                     u_neg_interpol.set_yi(u_cache[N:])
#                     _yi_token[1] = u_cache
#                 results[mask_else] = u_neg_interpol(mu[mask_else])
#
#             return np.squeeze(results)[()]
#     else:
#         raise ValueError("This subroutine can only interpolate u or u0.")
#
#     if hasattr(u, "tau_arr"):
#         u_interpol.tau_arr = u.tau_arr
#     return u_interpol



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


    
def _check_tau_derivative_order(tau_derivative_order):
    """Validates the ``tau_derivative_order`` argument of the output functions of ``pydisort``."""
    # `True == 1` in Python, so the `True` of the boolean flag which this argument replaced must not be read as 1
    if tau_derivative_order is True or tau_derivative_order not in (-1, 0, 1):
        raise ValueError("`tau_derivative_order` must be -1 (antiderivative), 0 or 1 (derivative).")



def _poly_tau_derivative(coeffs, tau, order):
    """The ``order``-th derivative (an antiderivative if ``order`` is -1) of polynomials in tau, each evaluated
    at its own value of tau. The exponentials in tau of the solution are instead differentiated by
    multiplying each by its rate to the power of ``order``.

    Arguments of _poly_tau_derivative
    |    Variable    |              Shape              |
    | -------------- | ------------------------------- |
    | ``coeffs``     | ``Ntau x ... x Ncoeffs``        | Arranged from lowest order term to highest
    | ``tau``        | ``Ntau``                        |
    | ``order``      | scalar                          |

    Returns an array of shape ``Ntau x ...``.
    """
    p = np.arange(np.shape(coeffs)[-1])
    # The derivative of tau^p is p! / (p - order)! * tau^(p - order), which is zero if order > p
    monomials = (
        sc.special.gamma(p + 1) * sc.special.rgamma(p + 1 - order) * tau[:, None] ** np.maximum(p - order, 0)
    )
    return np.einsum("t...p, tp -> t...", coeffs, monomials)



def _mathscr_v_coeffs(s_poly_coeffs, G, K, G_inv_mu_inv):
    """Polynomial coefficients of the particular solution for isotropic internal sources,
    arranged from lowest order term to highest. Refer to section 3.6.1 of the Comprehensive Documentation.
    The polynomials are in the same (delta-scaled) tau as ``s_poly_coeffs``
    and are evaluated by ``_poly_tau_derivative``.

    Arguments of _mathscr_v_coeffs
    |      Variable       |           Shape           |
    | ------------------- | ------------------------- |
    | ``s_poly_coeffs``   | ``NLayers x Nscoeffs``    |
    | ``G``               | ``NLayers x NQuad x NQuad`` |
    | ``K``               | ``NLayers x NQuad``       |
    | ``G_inv_mu_inv``    | ``NLayers x NQuad``       | Inverse of eigenvector matrix @ (1 / ``mu_arr``)

    Returns an array of shape ``NLayers x NQuad x Nscoeffs``.
    """
    Nscoeffs = np.shape(s_poly_coeffs)[1]
    if Nscoeffs > 10:
        warnings.warn("`Nscoeffs` is large and may cause instability.")

    # Coefficient of tau^q in the p-th derivative of the source polynomial; (NLayers, p, q)
    p = np.arange(Nscoeffs)[:, None]
    q = np.arange(Nscoeffs)[None, :]
    factorials = sc.special.factorial(np.arange(2 * Nscoeffs))
    source_derivs = np.pad(s_poly_coeffs, ((0, 0), (0, Nscoeffs)))[:, p + q] * (factorials[p + q] / factorials[q])

    K_inv_powers = (1 / K)[:, :, None] ** (np.arange(Nscoeffs) + 1)                 # (NLayers, NQuad, p)
    return G @ (G_inv_mu_inv[:, :, None] * (K_inv_powers @ source_derivs))



def _Chapman_factors(mu0, radii_arr):
    """Chapman factors of the direct beam in a spherical atmosphere, for the pseudo-spherical correction.
    Entry [l, n] is the ratio of the slant path to the vertical path across layer n for the beam which
    reaches the bottom of layer l at the polar angle arccos(``mu0``); it is zero if layer n is below layer l.
    Their product with the optical thicknesses of the layers is therefore the slant optical depth of the
    direct beam at the bottom of each layer. Refer to section 4.2 of the Comprehensive Documentation.

    Arguments of _Chapman_factors
    |    Variable    |      Shape      |
    | -------------- | --------------- |
    | ``mu0``        | scalar          |
    | ``radii_arr``  | ``NLayers + 1`` | Distance of each layer boundary from the centre of the planet, top to bottom

    Returns an array of shape ``NLayers x NLayers``.
    """
    NLayers = len(radii_arr) - 1
    # The beam which reaches the bottom of layer l is closest to the centre of the planet at the distance `closest[l]`,
    # and it crosses the boundary of radius r at the distance sqrt(r^2 - closest[l]^2) along the beam from that point
    closest = np.sqrt(1 - mu0**2) * radii_arr[1:]                                              # (NLayers,)
    along_beam = np.sqrt(np.maximum(radii_arr[None, :] ** 2 - closest[:, None] ** 2, 0))       # (NLayers, NLayers + 1)
    # The slant path across a layer is the difference of `along_beam` between its boundaries and the vertical path
    # is the difference of their radii; the ratio is written as below to avoid subtracting nearly equal numbers
    return np.divide(
        (radii_arr[:-1] + radii_arr[1:])[None, :],
        along_beam[:, :-1] + along_beam[:, 1:],
        out=np.zeros((NLayers, NLayers)),
        where=np.tri(NLayers, dtype=bool),
    )



def _Fourier_error(um_last, cos_last, u):
    """Cauchy / Fourier convergence evaluation for the last Fourier term of the intensity ``u``:
    the largest ratio of that term to the intensity, over the points where the intensity is not negligible.

    Arguments of _Fourier_error
    |    Variable    |         Shape          |
    | -------------- | ---------------------- |
    | ``um_last``    | ``Nmu x Ntau``         | Last Fourier mode of the intensity
    | ``cos_last``   | ``Nphi``               | Cosine of that mode at each azimuthal angle
    | ``u``          | ``Nmu x Ntau x Nphi``  |
    """
    u_abs = np.abs(u)
    return np.max(
        np.divide(
            np.abs(um_last[:, :, None] * cos_last[None, None, :]),
            u_abs,
            out=np.zeros_like(u_abs),
            where=u_abs > 1e-8,
        )
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