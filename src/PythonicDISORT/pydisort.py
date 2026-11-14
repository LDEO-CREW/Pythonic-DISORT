from PythonicDISORT import subroutines
from PythonicDISORT.subroutines import prepend
from PythonicDISORT._assemble_intensity_and_fluxes import _assemble_intensity_and_fluxes
from PythonicDISORT._NT_corrections import _NT_corrections
import warnings

import numpy as np
from math import pi


def pydisort(
    tau_arr, omega_arr,
    NQuad,
    Leg_coeffs_all,
    mu0, I0, phi0,
    NLeg=None, 
    NFourier=None,
    b_pos=0, 
    b_neg=0,
    only_flux=False,
    f_arr=0, 
    NT_cor=False,
    BDRF_Fourier_modes=[],
    s_poly_coeffs=np.array([[]]),
    use_banded_solver_NLayers=10,
    cache_asso_leg=None,
    interpolate_mu=False,
    radii_arr=None,
):
    """Solves the 1D RTE for the fluxes, and optionally intensity,
    of a multi-layer atmosphere with the specified optical properties, boundary conditions
    and sources. Optionally performs delta-M scaling, NT corrections and the pseudo-spherical correction,
    and optionally returns the intensity at arbitrary polar angles instead of at the quadrature nodes.
    Refer to the ``*_test.ipynb`` Jupyter Notebooks in the ``pydisotest`` directory for examples of use.
    
        See https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html#1.-USER-INPUT-REQUIRED:-Choose-parameters
        for a more detailed explanation of each parameter.
        See https://pythonic-disort.readthedocs.io/en/latest/Pythonic-DISORT.html#2.-PythonicDISORT-modules-and-outputs
        for a more detailed explanation of each output.
        The notebook also has numerous examples of this function being called.

    Parameters
    ----------
    tau_arr : array or float
        Optical depth of the lower boundary of each atmospheric layer.
    omega_arr : array or float
        Single-scattering albedo of each atmospheric layer.
    NQuad : int
        Number of ``mu`` quadrature nodes, i.e. number of streams.
    Leg_coeffs_all : 2darray
        All available unweighted phase function Legendre coefficients.
        Each row pertains to an atmospheric layer (from top to bottom).
        Each coefficient must be strictly between -1 and 1, except the zeroth which must be 1.
    mu0 : float
        Cosine of polar angle of the incident beam.
    I0 : float
        Intensity of the incident beam.
    phi0 : float
        Azimuthal angle of the incident beam.
    NLeg : optional, int
        Number of phase function Legendre coefficients
        to use in the pre-correction solver.
    NFourier : optional, int
        Number of Fourier modes to use to construct the intensity function.
    b_pos : optional, 2darray or float
        Dirichlet condition at the bottom boundary for the upward direction.
        Each column pertains to a Fourier mode (ascending order).
    b_neg : optional, 2darray or float
        Dirichlet condition at the top boundary for the downward direction.
        Each column pertains to a Fourier mode (ascending order).
    only_flux : optional, bool
        Do NOT compute the intensity function?
    f_arr : optional, array or float
        Fractional scattering into peak for each atmospheric layer.
        Each row pertains to an atmospheric layer (from top to bottom).
        We recommend setting ``f_arr`` to ``Leg_coeffs_all[NLeg]``,
        or ``Leg_coeffs_all[:, NLeg]`` for a multi-layer atmosphere.
    NT_cor : optional, bool
        Perform Nakajima-Tanaka intensity corrections?
        The corrections are applied at the quadrature points or, if ``interpolate_mu=True``,
        directly at the ``mu`` values that are passed to ``u``, as Stamnes' DISORT does,
        which requires ``|mu| >= 1e-8``. As in Stamnes' DISORT, the corrections are not performed
        if the pseudo-spherical correction is applied (see ``radii_arr``).
    BDRF_Fourier_modes : optional, list of functions
        BDRF Fourier modes, each a float, or a function with arguments ``mu, -mu_p`` of type array
        which output has the same dimensions as the outer product of the two arrays.
    s_poly_coeffs : optional, array
        Polynomial coefficients of isotropic internal sources.
        Each row pertains to an atmospheric layer (from top to bottom).
        Arrange coefficients from lowest order term to highest.
    use_banded_solver_NLayers : optional, int
        At or above how many atmospheric layers should ``scipy.linalg.solve_banded`` be used?
    cache_asso_leg : optional, {None, "no_mu0", "mu0"}
        In-process caching of the seminormalized associated Legendre table, keyed on
        (``NQuad``, ``NLeg``) and, for the direct beam, on ``mu0``.
        This is a pure speed/memory option: the computed fluxes and intensities are identical for
        all three settings.

        - ``None`` (default): no caching; the table is rebuilt on every call.
        - ``"no_mu0"``: cache the ``mu0``-independent part (the quadrature-node table) and reuse it
          across all calls with the same discretization; the single direct-beam ``-mu0`` column is
          still rebuilt each call. Recommended for large batch runs that share ``NQuad`` but vary
          ``mu0`` (or have no beam, where the whole table is cached).
        - ``"mu0"``: additionally cache the full table keyed on ``mu0``. Use this only when ``mu0``
          is constant (or takes few distinct values) across many calls, e.g. a large batch of columns
          at a fixed solar angle, in which case the associated Legendre table is built just once.

        The cache lives for the lifetime of the process and is bounded (least-recently-used); a run
        with a single discretization keeps a single entry.
    interpolate_mu : optional, bool
        If ``True``, the intensity functions ``u0`` and ``u`` take ``mu`` (cosine of polar angle) as their
        first argument and return the intensity at those ``mu`` values rather than at the quadrature nodes.
        The intensity at each ``mu`` is computed by the integration of the source function, like Stamnes'
        DISORT does: the transfer equation is solved exactly along ``mu`` with the source function of the
        discrete ordinates solution. The result equals that solution at the quadrature nodes and satisfies
        the boundary conditions at every ``mu``. Vector boundary conditions ``b_pos`` and ``b_neg`` are only
        known at the quadrature nodes and are interpolated polynomially.
        It is fastest to pass all the ``mu`` values of interest in one call of ``u``.
    radii_arr : optional, array
        Distance of each layer boundary from the centre of the planet, from the top boundary to the
        bottom boundary (``NLayers + 1`` values) and in any unit of length, e.g. the radius of the planet
        plus the altitude of each boundary. If provided, the pseudo-spherical correction is applied:
        the direct beam is attenuated along its path through the spherical atmosphere rather than
        through a plane-parallel one, which matters when the sun is low. ``mu0`` is then the cosine of
        the angle between the beam and the local vertical.
        The diffuse radiation is still treated as plane-parallel.

    Returns
    -------
    mu_arr : array
        All ``mu`` (cosine of polar angle) quadrature nodes.
    Fp(tau) : function
        (Energetic) Flux function with argument ``tau`` (type: array or float) for positive (upward) ``mu`` values.
        Returns the diffuse flux magnitudes (same type and size as ``tau``).
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        For example, ``Fp(tau_arr, -1) - Fp(np.insert(np.nextafter(tau_arr[:-1], np.inf), 0, 0), -1)``
        will produce an array of the tau-integral over each layer.
        Carries ``tau_arr`` as an attribute.
    Fm(tau) : function
        (Energetic) Flux function with argument ``tau`` (type: array or float) for negative (downward) ``mu`` values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively where each entry is of the
        same type and size as ``tau``.
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        Carries ``tau_arr`` as an attribute.
    u0(tau) or u0(mu, tau) : function
        Zeroth Fourier mode of the intensity with argument ``tau`` (type: array or float)
        or, if ``interpolate_mu=True``, with arguments ``(mu, tau)`` each of type array or float.
        Returns an ndarray with axes corresponding to variation with ``mu`` and ``tau`` respectively,
        where ``mu`` are the quadrature nodes ``mu_arr`` unless ``interpolate_mu=True``.
        This function is useful for calculating actinic fluxes and other quantities of interest,
        but reclassification of delta-scaled flux and other corrections must be done manually
        (for actinic flux ``subroutines.generate_diff_act_flux_funcs`` will automatically perform the reclassification).
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        Carries ``tau_arr`` as an attribute.
    u(tau, phi) or u(mu, tau, phi) : function, optional
        Intensity function with arguments ``(tau, phi)`` each of type array or float
        or, if ``interpolate_mu=True``, with arguments ``(mu, tau, phi)`` each of type array or float.
        Returns an ndarray with axes corresponding to variation with ``mu, tau, phi`` respectively,
        where ``mu`` are the quadrature nodes ``mu_arr`` unless ``interpolate_mu=True``.
        Pass ``tau_derivative_order = 1`` or ``-1`` (defaults to ``0``)
        to switch to the derivative or an antiderivative of this function with respect to ``tau``.
        Pass ``return_Fourier_error = True`` (defaults to ``False``) to return the
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term,
        over the points at which the intensity is evaluated.
        Carries ``tau_arr`` as an attribute.
    """

    """
    Arguments of pydisort
    |          Variable           |                    Type / Shape                 |
    | --------------------------- | ----------------------------------------------- |
    | `tau_arr`                   | `NLayers`                                       |
    | `omega_arr`                 | `NLayers`                                       |
    | `NQuad`                     | scalar                                          |
    | `Leg_coeffs_all`            | `NLayers x NLeg_all`                            |
    | `mu0`                       | scalar                                          |
    | `I0`                        | scalar                                          |
    | `phi0`                      | scalar                                          |
    | `NLeg`                      | scalar                                          |
    | `NFourier`                  | scalar                                          |
    | `b_pos`                     | `NQuad/2 x NFourier` or `NQuad/2` or scalar     |
    | `b_neg`                     | `NQuad/2 x NFourier` or `NQuad/2` or scalar     |              
    | `only_flux`                 | boolean                                         |
    | `f_arr`                     | `NLayers`                                       |
    | `NT_cor`                    | boolean                                         |
    | `BDRF_Fourier_modes`        | `NBDRF`                                         |
    | `s_poly_coeffs`             | `NLayers x Nscoeffs`                            |
    | `use_banded_solver_NLayers` | scalar                                          |
    | `cache_asso_leg`            | None, "no_mu0" or "mu0"                         |
    | `interpolate_mu`            | boolean                                         |
    | `radii_arr`                 | `NLayers + 1`                                   |

    Notable internal variables of pydisort
    |          Variable            |     Type / Shape     |
    | ---------------------------- | -------------------- |
    | `rescale_factor`             | scalar               |
    | `thickness_arr`              | `NLayers`            |
    | `weighted_Leg_coeffs_all`    | `NLayers x NLeg_all` |
    | `Leg_coeffs`                 | `NLayers x NLeg`     |
    | `weighted_scaled_Leg_coeffs` | `NLayers x NLeg`     |
    | `mu_arr_pos`                 | `NQuad/2`            |
    | `W`                          | `NQuad/2`            |
    | `mu_arr`                     | `NQuad`              |
    | `M_inv`                      | `NQuad/2`            |
    | `scale_tau`                  | `NLayers`            |
    | `scaled_tau_arr_with_0`      | `NLayers + 1`        |
    | `scaled_thickness_arr`       | `NLayers`            |
    | `scaled_omega_arr`           | `NLayers`            |
    | `scaled_s_poly_coeffs`       | `NLayers x Nscoeffs` |
    | `mu0_arr`                    | `NLayers`            |
    | `slant_tau_arr_with_0`       | `NLayers + 1`        |
    | `scaled_mu0_arr`             | `NLayers`            |
    | `scaled_slant_tau_arr_with_0`| `NLayers + 1`        |
    """
    
    # Turn floats into arrays
    # --------------------------------------------------------------------------------------------------------------------------
    tau_arr = np.atleast_1d(tau_arr).copy()
    omega_arr = np.atleast_1d(omega_arr).copy()
    Leg_coeffs_all = np.atleast_2d(Leg_coeffs_all).copy()
    s_poly_coeffs = np.atleast_2d(s_poly_coeffs)
    f_arr = np.atleast_1d(f_arr).copy()
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Setup
    # --------------------------------------------------------------------------------------------------------------------------
    if NLeg is None:
        NLeg = NQuad
    if only_flux:
        NFourier = 1 # We only need to solve for the 0th Fourier mode to compute the flux
    elif NFourier is None:
        NFourier = NLeg
    if np.all(b_pos == 0):
        b_pos = 0
    if np.all(b_neg == 0):
        b_neg = 0
    if np.all(s_poly_coeffs == 0):
        Nscoeffs = 0
    else:
        Nscoeffs = np.shape(s_poly_coeffs)[1]
    NLayers = len(tau_arr)
    if len(f_arr) == 1 and NLayers > 1:
        f_arr = np.full(NLayers, f_arr[0]) # Broadcast scalar `f_arr` to all layers
    b_pos_is_scalar = False
    b_neg_is_scalar = False
    b_pos_is_vector = False
    b_neg_is_vector = False
    thickness_arr = prepend(np.diff(tau_arr), NLayers - 1, tau_arr[0])
    NLeg_all = np.shape(Leg_coeffs_all)[1]
    N = NQuad // 2
    there_is_beam_source = I0 > 0
    there_is_iso_source = Nscoeffs > 0
    is_atmos_multilayered = NLayers > 1
    # --------------------------------------------------------------------------------------------------------------------------

    # Input checks (refer to section 1 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    # Optical depths and thickness must be positive
    if not np.all(tau_arr > 0):
        raise ValueError("tau values cannot be non-positive.")
    if not np.all(thickness_arr > 0):
        raise ValueError("Layer thicknesses cannot be non-positive.")
    # Single-scattering albedo must be between 0 and 1, excluding 1
    if not (np.all(omega_arr >= 0) and np.all(omega_arr < 1)):
        raise ValueError("Single-scattering albedo must be between 0 and 1, excluding 1.") 
    # There must be a positive number of Legendre coefficients each with magnitude <= 1
    # The user must supply at least as many phase function Legendre coefficients as intended for use
    if not NLeg > 0:
        raise ValueError("The number of phase function Legendre coefficients must be positive.") 
    if not NLeg <= NLeg_all:
        raise ValueError("`NLeg` cannot be larger than the number of phase function Legendre coefficients provided.")
    # Ensure that the first dimension of the following inputs corresponds to the number of layers
    if not np.shape(Leg_coeffs_all)[0] == NLayers:
        raise ValueError("The zeroth dimension of the shape of `Leg_coeffs_all` does not match the number of layers which is deduced from the length of `tau_arr`.")
    if not len(omega_arr) == NLayers:
        raise ValueError("The zeroth dimension of the shape of `omega_arr` does not match the number of layers which is deduced from the length of `tau_arr`.")
    if np.any(f_arr != 0) and not len(f_arr) == NLayers:
        raise ValueError("The length of `f_arr` does not match the number of layers which is deduced from the length of `tau_arr`.")
    if there_is_iso_source and not np.shape(s_poly_coeffs)[0] == NLayers:
        raise ValueError("The zeroth dimension of the shape of `s_poly_coeffs` does not match the number of layers which is deduced from the length of `tau_arr`.")
    # Value checks on the phase function Legendre coefficients
    if not np.all(omega_arr * Leg_coeffs_all[:, 0] == omega_arr):
        warnings.warn("The zeroth index phase function Legendre coefficient must be, and has been corrected to, 1.")
        Leg_coeffs_all[:, 0] = 1
    if not (np.all(-1 < Leg_coeffs_all[:, 1:]) and np.all(Leg_coeffs_all[:, 1:] < 1)):
        raise ValueError("The phase function Legendre coefficients must all be between -1 and 1 exclusive (only the zeroth coefficient can equal 1).")
    # Conditions on the number of quadrature angles (NQuad), Legendre coefficients (NLeg) and loops (NFourier)
    if not NQuad >= 2:
        raise ValueError("There must be at least two streams.")
    if not NQuad % 2 == 0:
        raise ValueError("The number of streams must be even.")
    if not NFourier > 0:
        raise ValueError("The number of Fourier modes to use in the solution must be positive.")
    if not NFourier <= NLeg:
        raise ValueError("The number of Fourier modes to use in the solution must be less than or equal to the number of phase function Legendre coefficients used.")
    if NFourier > 64 and not only_flux:
        warnings.warn("`NFourier` is large and may cause errors, consider decreasing `NFourier` to 64 and it probably should be even less. By default `NFourier` equals `NLeg`.")
    # Not strictly necessary but there will be tremendous inaccuracies if this is violated
    if not NLeg <= NQuad:
        raise ValueError("There should be more streams than the number of phase function Legendre coefficients used.")
    # We require principal angles and a downward incident beam
    if I0 < 0:
        raise ValueError("The intensity of the incident beam cannot be negative.")
    if there_is_beam_source:
        if not (0 < mu0 and mu0 <= 1):
            raise ValueError("The cosine of the polar angle of the incident beam must be between 0 and 1, excluding 0.")
        if not (0 <= phi0 and phi0 < 2 * pi):
            raise ValueError("Provide the principal azimuthal angle for the incident beam (must be between 0 and 2pi, excluding 2pi).")
    # Ensure that the BC inputs are of the correct shape
    if len(np.atleast_1d(b_pos)) == 1:
        b_pos_is_scalar = True
    elif np.ndim(b_pos) == 1 and len(b_pos) == N:
        b_pos_is_vector = True
    elif not np.shape(b_pos) == (N, NFourier):
        raise ValueError("The shape of the bottom boundary condition is incorrect.")
    if len(np.atleast_1d(b_neg)) == 1:
        b_neg_is_scalar = True
    elif np.ndim(b_neg) == 1 and len(b_neg) == N:
        b_neg_is_vector = True
    elif not np.shape(b_neg) == (N, NFourier):
        raise ValueError("The shape of the top boundary condition is incorrect.")
    # The fractional scattering must be between 0 and 1, excluding 1
    if not (np.all(0 <= f_arr) and np.all(f_arr < 1)):
        raise ValueError("The fractional scattering must be between 0 and 1, excluding 1.")
    # The minimum threshold is 3 else the matrix will not be banded
    if not use_banded_solver_NLayers >= 3:
        raise ValueError("The minimum threshold `use_banded_solver_NLayers` is 3, else the matrix will not be banded.")
    # Associated Legendre table caching mode
    if cache_asso_leg not in (None, "no_mu0", "mu0"):
        raise ValueError('`cache_asso_leg` must be one of None, "no_mu0" or "mu0".')
    # The radii of the layer boundaries must be positive and decrease from the top boundary to the bottom boundary
    if radii_arr is not None:
        radii_arr = np.array(radii_arr, dtype=float)
        if not np.shape(radii_arr) == (NLayers + 1,):
            raise ValueError("`radii_arr` must have one radius for each layer boundary, i.e. one more than the number of layers.")
        if not (np.all(np.diff(radii_arr) < 0) and radii_arr[-1] > 0):
            raise ValueError("The radii of the layer boundaries must be positive and strictly decreasing (top to bottom).")
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Post input checks setup
    # --------------------------------------------------------------------------------------------------------------------------
    NBDRF = len(BDRF_Fourier_modes)
    weighted_Leg_coeffs_all = (2 * np.arange(NLeg_all) + 1) * Leg_coeffs_all
    Leg_coeffs = Leg_coeffs_all[:, :NLeg]
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Generation of "double-Gauss" quadrature weights and points (refer to section 3.4 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    # For positive mu values (the weights are identical for both domains)
    mu_arr_pos, W = subroutines.Gauss_Legendre_quad(N)  # mu_arr_neg = -mu_arr_pos
    mu_arr = np.concatenate([mu_arr_pos, -mu_arr_pos])
    M_inv = 1 / mu_arr_pos

    # Delta-M scaling; there is no scaling if f = 0
    # Refer to sections 1.3.1 and 3.3 of the Comprehensive Documentation
    # --------------------------------------------------------------------------------------------------------------------------
    if np.any(f_arr > 0):

        scale_tau = 1 - omega_arr * f_arr
        scaled_thickness_arr = scale_tau * thickness_arr
        scaled_tau_arr_with_0 = prepend(np.cumsum(scaled_thickness_arr), NLayers, 0)
        scaled_Leg_coeffs = (Leg_coeffs - f_arr[:, None]) / (1 - f_arr)[:, None]
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = (1 - f_arr) / scale_tau * omega_arr

        translations = scaled_tau_arr_with_0[:-1] - scale_tau * prepend(tau_arr[:-1], NLayers - 1, 0)
        scaled_s_poly_coeffs = (
            subroutines.affine_transform_poly_coeffs(s_poly_coeffs, scale_tau, translations)
            / scale_tau[:, None]  # Divide by d\tau^* / d\tau
        ) * (1 - omega_arr)[:, None]  # Enforce Kirchhoff's Law of Thermal Radiation: absorptivity equals emissivity

    else:
        # This is a shortcut to the same results
        scale_tau = np.ones(NLayers)
        scaled_thickness_arr = thickness_arr
        scaled_tau_arr_with_0 = prepend(tau_arr, NLayers, 0)
        scaled_Leg_coeffs = Leg_coeffs
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = omega_arr
        scaled_s_poly_coeffs = s_poly_coeffs * (1 - omega_arr)[:, None]  # Enforce Kirchhoff's Law of Thermal Radiation: absorptivity equals emissivity
        
    if np.any(scaled_omega_arr > 1 - 1e-6):
        warnings.warn("Some delta-scaled single-scattering albedos are very close to 1 which may cause numerical instability.")
    if (np.any(-0.95 > scaled_Leg_coeffs[:, 1:]) or np.any(scaled_Leg_coeffs[:, 1:] > 0.95)):
        warnings.warn("Some delta-scaled phase function Legendre coefficients have a magnitude that is very close to 1" +
        " (this excludes the zeroth index coefficient which must be 1) which may cause numerical instability.")
    
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Rescale of sources 
    # Refer to section 1.4 of the Comprehensive Documentation
    # --------------------------------------------------------------------------------------------------------------------------
    if there_is_iso_source:
        rescale_factor = np.max(
            (
                I0,
                np.max(b_pos),
                np.max(b_neg),
                scaled_s_poly_coeffs[0, 0],
                scaled_s_poly_coeffs[-1, :] @ (scaled_tau_arr_with_0[-1] ** np.arange(Nscoeffs))
            )
        )
        if rescale_factor > 0:
            I0 = (I0 / rescale_factor).copy()
            b_pos = (b_pos / rescale_factor).copy()
            b_neg = (b_neg / rescale_factor).copy()
            scaled_s_poly_coeffs = (scaled_s_poly_coeffs / rescale_factor).copy()
        else:
            rescale_factor = 1
    else:
        rescale_factor = np.max((I0, np.max(b_pos), np.max(b_neg)))
        if rescale_factor != 0:
            I0 = (I0 / rescale_factor).copy()
            b_pos = (b_pos / rescale_factor).copy()
            b_neg = (b_neg / rescale_factor).copy()
            
    I0_div_4pi = I0 / (4 * pi)
    # --------------------------------------------------------------------------------------------------------------------------

    # Direct beam: its slant optical depth at each layer boundary and its cosine in each layer,
    # for the atmosphere and for the delta-scaled atmosphere.
    # Refer to section 4.2 of the Comprehensive Documentation for the pseudo-spherical correction.
    # --------------------------------------------------------------------------------------------------------------------------
    if not there_is_beam_source:
        mu0_arr = scaled_mu0_arr = slant_tau_arr_with_0 = scaled_slant_tau_arr_with_0 = None

    elif radii_arr is None:
        # Plane-parallel atmosphere
        mu0_arr = scaled_mu0_arr = np.full(NLayers, mu0)
        slant_tau_arr_with_0 = prepend(tau_arr, NLayers, 0) / mu0
        scaled_slant_tau_arr_with_0 = scaled_tau_arr_with_0 / mu0

    else:
        # Pseudo-spherical correction: the slant optical depth follows from the Chapman factors at the layer
        # boundaries and is linear in tau within each layer, where the beam therefore has an effective cosine
        Chapman_factors = subroutines._Chapman_factors(mu0, radii_arr)
        slant_tau_arr_with_0 = prepend(Chapman_factors @ thickness_arr, NLayers, 0)
        scaled_slant_tau_arr_with_0 = prepend(Chapman_factors @ scaled_thickness_arr, NLayers, 0)
        mu0_arr = thickness_arr / np.diff(slant_tau_arr_with_0)
        scaled_mu0_arr = scaled_thickness_arr / np.diff(scaled_slant_tau_arr_with_0)
    # --------------------------------------------------------------------------------------------------------------------------

    # Delta-M scaled solution; the flux is never NT-corrected
    outputs = _assemble_intensity_and_fluxes(
        scaled_omega_arr,
        tau_arr,
        scaled_tau_arr_with_0,
        mu_arr_pos,
        M_inv, W,
        N, NQuad, NLeg,
        NFourier, NLayers, NBDRF,
        is_atmos_multilayered,
        weighted_scaled_Leg_coeffs,
        BDRF_Fourier_modes,
        mu0, I0, I0_div_4pi,
        rescale_factor, phi0,
        there_is_beam_source,
        mu0_arr, slant_tau_arr_with_0,
        scaled_mu0_arr, scaled_slant_tau_arr_with_0,
        b_pos, b_neg,
        b_pos_is_scalar, b_neg_is_scalar,
        b_pos_is_vector, b_neg_is_vector,
        scaled_s_poly_coeffs,
        there_is_iso_source,
        scale_tau,
        only_flux,
        use_banded_solver_NLayers,
        cache_asso_leg,
        interpolate_mu,
    )
    if only_flux:
        return (mu_arr,) + outputs
    flux_up, flux_down, u0, u_star = outputs

    # NT corrections to the intensity but not the flux (refer to section 3.7.2 of the Comprehensive Documentation).
    # As in Stamnes' DISORT, they are not performed if the pseudo-spherical correction is applied.
    u = _NT_corrections(
        u_star,
        NT_cor and radii_arr is None,
        interpolate_mu,
        tau_arr,
        thickness_arr,
        omega_arr,
        f_arr,
        Leg_coeffs_all,
        weighted_Leg_coeffs_all,
        NLeg, NLeg_all,
        NLayers,
        is_atmos_multilayered,
        mu_arr_pos, M_inv,
        scale_tau,
        scaled_tau_arr_with_0,
        scaled_thickness_arr,
        scaled_omega_arr,
        mu0, phi0, I0_div_4pi,
        rescale_factor,
        there_is_beam_source,
    )

    return mu_arr, flux_up, flux_down, u0, u
