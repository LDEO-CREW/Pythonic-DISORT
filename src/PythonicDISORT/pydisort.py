from PythonicDISORT import subroutines
from PythonicDISORT.subroutines import prepend
from PythonicDISORT._assemble_intensity_and_fluxes import _assemble_intensity_and_fluxes
import warnings

import numpy as np
import scipy as sc
from math import pi
from numpy.polynomial.legendre import legval
from scipy import integrate
    

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
    autograd_compatible=False,
    cache_asso_leg=None,
):
    """Solves the 1D RTE for the fluxes, and optionally intensity,
    of a multi-layer atmosphere with the specified optical properties, boundary conditions
    and sources. Optionally performs delta-M scaling and NT corrections. 
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
        Each coefficient should be between 0 and 1 inclusive.
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
        The corrections are applied at the quadrature points; see the ``NT_cor``
        argument of ``subroutines.interpolate`` for overriding this option after
        the fact, in particular to instead apply the corrections directly at
        interpolated ``mu`` evaluation points as Stamnes' DISORT does.
    BDRF_Fourier_modes : optional, list of functions
        BDRF Fourier modes, each a float, or a function with arguments ``mu, -mu_p`` of type array
        which output has the same dimensions as the outer product of the two arrays.
    s_poly_coeffs : optional, array
        Polynomial coefficients of isotropic internal sources.
        Each row pertains to an atmospheric layer (from top to bottom).
        Arrange coefficients from lowest order term to highest.
    use_banded_solver_NLayers : optional, int
        At or above how many atmospheric layers should ``scipy.linalg.solve_banded`` be used?
    autograd_compatible : optional, bool
        If ``True``, the autograd package: https://github.com/HIPS/autograd can be used to compute
        the ``tau``-derivatives of the output functions but ``pydisort`` will be less efficient.
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

    Returns
    -------
    mu_arr : array
        All ``mu`` (cosine of polar angle) quadrature nodes.
    Fp(tau) : function
        (Energetic) Flux function with argument ``tau`` (type: array or float) for positive (upward) ``mu`` values.
        Returns the diffuse flux magnitudes (same type and size as ``tau``).
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        For example, ``Fp(tau_arr, True) - Fp(np.insert(tau_arr[:-1] + 1e-15, 0, 0), True)``
        will produce an array of the tau-integral over each layer.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).
    Fm(tau) : function
        (Energetic) Flux function with argument ``tau`` (type: array or float) for negative (downward) ``mu`` values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively where each entry is of the
        same type and size as ``tau``.
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).
    u0(tau) : function
        Zeroth Fourier mode of the intensity with argument ``tau`` (type: array or float).
        Returns an ndarray with axes corresponding to variation with ``mu`` and ``tau`` respectively.
        This function is useful for calculating actinic fluxes and other quantities of interest,
        but reclassification of delta-scaled flux and other corrections must be done manually
        (for actinic flux ``subroutines.generate_diff_act_flux_funcs`` will automatically perform the reclassification).
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).
    u(tau, phi) : function, optional
        Intensity function with arguments ``(tau, phi)`` each of type array or float.
        Returns an ndarray with axes corresponding to variation with ``mu, tau, phi`` respectively.
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        Pass ``return_Fourier_error = True`` (defaults to ``False``) to return the 
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
        Pass ``return_tau_arr`` to return ``tau_arr`` (defaults to ``False``).
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
    | `scaled_omega_arr`           | `NLayers`            |
    | `scaled_s_poly_coeffs`       | `NLayers x Nscoeffs` |
    | `sum1`                       | scalar               |
    | `omega_avg`                  | scalar               |
    | `sum2`                       | scalar               |
    | `f_avg`                      | scalar               |
    | `Leg_coeffs_residue`         | `NLayers x NLeg_all` |
    | `Leg_coeffs_residue_avg`     | `NLeg_all`           |
    | `scaled_mu0`                 | scalar               |
    """
    
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np
    
    # Turn floats into arrays
    # --------------------------------------------------------------------------------------------------------------------------
    tau_arr = np.atleast_1d(tau_arr)
    omega_arr = np.atleast_1d(omega_arr)
    Leg_coeffs_all = np.atleast_2d(Leg_coeffs_all)
    s_poly_coeffs = np.atleast_2d(s_poly_coeffs)
    f_arr = np.atleast_1d(f_arr)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Setup
    # --------------------------------------------------------------------------------------------------------------------------
    if NLeg is None:
        NLeg = NQuad
    if only_flux:
        NFourier = 1 # We only need to solve for the 0th Fourier mode to compute the flux
    elif NFourier is None:
        NFourier = NQuad
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
        # Copy before correcting: `np.atleast_2d` returns a view of 2D inputs,
        # so writing in place would mutate the caller's array
        Leg_coeffs_all = Leg_coeffs_all.copy()
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
        warnings.warn("`NFourier` is large and may cause errors, consider decreasing `NFourier` to 64 and it probably should be even less. By default `NFourier` equals `NQuad`.")
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
    elif len(b_pos) == N:
        b_pos_is_vector = True
    elif not np.shape(b_pos) == (N, NFourier):
        raise ValueError("The shape of the bottom boundary condition is incorrect.")
    if len(np.atleast_1d(b_neg)) == 1:
        b_neg_is_scalar = True
    elif len(b_neg) == N:
        b_neg_is_vector = True
    elif not np.shape(b_neg) == (N, NFourier):
        raise ValueError("The shape of the top boundary condition is incorrect.")
    # The fractional scattering must be between 0 and 1
    if not (np.all(0 <= f_arr) and np.all(f_arr <= 1)):
        raise ValueError("The fractional scattering must be between 0 and 1.")
    # The minimum threshold is 3 else the matrix will not be banded
    if not use_banded_solver_NLayers >= 3:
        raise ValueError("The minimum threshold `use_banded_solver_NLayers` is 3, else the matrix will not be banded.")
    # Associated Legendre table caching mode
    if cache_asso_leg not in (None, "no_mu0", "mu0"):
        raise ValueError('`cache_asso_leg` must be one of None, "no_mu0" or "mu0".')
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

    # NOTE: `mu0` MAY equal a quadrature / computational angle; the removable
    # singularities this creates in the NT corrections are handled exactly by the
    # phi-kernel formulations in `_TMS_correction` and `_IMS_correction`
    # --------------------------------------------------------------------------------------------------------------------------

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
        I0 = (I0 / rescale_factor).copy()
        b_pos = (b_pos / rescale_factor).copy()
        b_neg = (b_neg / rescale_factor).copy()
        scaled_s_poly_coeffs = (scaled_s_poly_coeffs / rescale_factor).copy()
    else:
        rescale_factor = np.max((I0, np.max(b_pos), np.max(b_neg)))
        if rescale_factor != 0:
            I0 = (I0 / rescale_factor).copy()
            b_pos = (b_pos / rescale_factor).copy()
            b_neg = (b_neg / rescale_factor).copy()
            
    I0_div_4pi = I0 / (4 * pi)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # NT corrections require a beam source, delta-M scaling (`f_arr > 0`), scattering
    # (`omega_arr > 0`) and a truncated phase function (`NLeg < NLeg_all`)
    # Refer to section 3.7.2 of the Comprehensive Documentation
    NT_corrections_possible = bool(
        there_is_beam_source and np.any(f_arr > 0) and NLeg < NLeg_all and np.any(omega_arr > 0)
    )

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
        b_pos, b_neg,
        b_pos_is_scalar, b_neg_is_scalar,
        b_pos_is_vector, b_neg_is_vector,
        Nscoeffs,
        scaled_s_poly_coeffs,
        there_is_iso_source,
        scale_tau,
        only_flux,
        use_banded_solver_NLayers,
        autograd_compatible,
        cache_asso_leg,
    )
    if only_flux:
        return (mu_arr,) + outputs
    flux_up, flux_down, u0, u_star = outputs

    if NT_corrections_possible:

        # Statics: quantities used by the NT corrections which are independent of the evaluation points.
        # Their computation is deferred so that `NT_cor=False` runs only pay for them if corrections
        # are later requested through `subroutines.interpolate`.
        # --------------------------------------------------------------------------------------------------------------------------
        def _compute_NT_statics():
            statics = {}

            # `legval` is linear in its coefficients, so the TMS phase function difference
            # (p_true / (1 - f) - p_trun) can be evaluated in a single call on the difference
            # of the coefficient sets. For l < NLeg the difference is exactly (2l + 1) f / (1 - f).
            Leg_coeffs_diff = weighted_Leg_coeffs_all / (1 - f_arr)[:, None]
            Leg_coeffs_diff[:, :NLeg] = (
                (2 * np.arange(NLeg) + 1)[None, :] * (f_arr / (1 - f_arr))[:, None]
            )
            statics["Leg_coeffs_diff_T"] = np.ascontiguousarray(Leg_coeffs_diff.T)
            statics["exp_scaled_tau_arr_with_0"] = np.exp(-scaled_tau_arr_with_0 / mu0)

            # IMS statics (refer to section 3.7.2 of the Comprehensive Documentation and
            # eq. A.15 of STWL2000). The averages are weighted by layer optical THICKNESS
            # (DISORT's `DTAUC`), matching DISORT's `SECSCA` subroutine. DISORT averages
            # from the top of the atmosphere only DOWN TO each output depth, so its averages
            # are tau-dependent; those depth-dependent averages do not admit the closed-form
            # tau-antiderivatives used by `is_antiderivative_wrt_tau`. PythonicDISORT instead
            # evaluates the averages down to each LAYER BOUNDARY (index k = 0, ..., NLayers,
            # where the k = 0 top-boundary limit equals the k = 1 set as only the first layer
            # is then sampled) and `_IMS_correction` blends the resulting correction linearly
            # in tau within each layer. The blend matches DISORT's averaging exactly at every
            # layer boundary, throughout the first layer, and everywhere when all layers are
            # identical (in particular for single-layer atmospheres, which also skip the
            # blend entirely: see `_IMS_correction`).
            weights = omega_arr * thickness_arr
            cumsum1 = np.cumsum(weights)                                     # (NLayers,)
            cumsum2 = np.cumsum(f_arr * weights)                             # (NLayers,)
            # Guarded divisions: a zero cumulative sum means no (peaked) scattering above
            # the boundary, whence `IMS_scalar` and `IMS_Leg_coeffs` correctly vanish
            omega_avg = cumsum1 / tau_arr
            f_avg = np.where(cumsum1 > 0, cumsum2 / np.where(cumsum1 == 0, 1, cumsum1), 0)
            omega_f_avg = omega_avg * f_avg                                  # (NLayers,)
            Leg_coeffs_residue = Leg_coeffs_all.copy()
            Leg_coeffs_residue[:, :NLeg] = f_arr[:, None]
            Leg_coeffs_residue_avg = np.where(
                (cumsum2 > 0)[:, None],
                np.cumsum(Leg_coeffs_residue * weights[:, None], axis=0)
                / np.where(cumsum2 == 0, 1, cumsum2)[:, None],
                0,
            )                                                                # (NLayers, NLeg_all)
            omega_f_avg = np.concatenate([omega_f_avg[:1], omega_f_avg])
            Leg_coeffs_residue_avg = np.concatenate(
                [Leg_coeffs_residue_avg[:1], Leg_coeffs_residue_avg], axis=0
            )
            statics["scaled_mu0"] = mu0 / (1 - omega_f_avg)                  # (NLayers + 1,)
            statics["IMS_scalar"] = (
                I0 / (4 * pi) * omega_f_avg**2 / (1 - omega_f_avg)
            )                                                                # (NLayers + 1,)
            statics["IMS_Leg_coeffs"] = (2 * np.arange(NLeg_all) + 1) * (
                2 * Leg_coeffs_residue_avg - Leg_coeffs_residue_avg**2
            )                                                                # (NLayers + 1, NLeg_all)
            statics["tau_arr_with_0"] = np.concatenate([np.zeros(1), tau_arr])

            # Decay and coefficient arrays for the multi-layer scans at the quadrature nodes
            if is_atmos_multilayered:
                statics["TMS_statics_quad_pos"] = _TMS_mu_statics(
                    M_inv, True, scaled_thickness_arr, mu0, statics["exp_scaled_tau_arr_with_0"]
                )
                statics["TMS_statics_quad_neg"] = _TMS_mu_statics(
                    M_inv, False, scaled_thickness_arr, mu0, statics["exp_scaled_tau_arr_with_0"]
                )
            else:
                statics["TMS_statics_quad_pos"] = None
                statics["TMS_statics_quad_neg"] = None
            return statics

        _NT_statics_cache = [None]

        def _get_NT_statics():
            if _NT_statics_cache[0] is None:
                _NT_statics_cache[0] = _compute_NT_statics()
            return _NT_statics_cache[0]
        # --------------------------------------------------------------------------------------------------------------------------

        # NT corrections at arbitrary `mu`; used by `subroutines.interpolate` (see the now closed GitHub issue #47)
        # --------------------------------------------------------------------------------------------------------------------------
        def _NT_corrections_at_mu(mu, tau, phi, is_antiderivative_wrt_tau=False):
            """NT (TMS + IMS) corrections to the delta-M scaled, uncorrected intensity `u_star`,
            evaluated at arbitrary cosine polar angles `mu`. Returns an (Nmu, Ntau, Nphi) ndarray
            which is NOT squeezed but IS already multiplied by `rescale_factor`.
            This function is private but is exposed to `subroutines.interpolate`
            through the `_NT_data` attribute of the intensity function `u`.
            """
            statics = _get_NT_statics()
            mu = np.atleast_1d(mu)
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)

            if not np.all(np.abs(mu) <= 1):
                raise ValueError("mu values must be between -1 and 1.")
            if np.any(np.abs(mu) < 1e-8):
                raise ValueError(
                    "NT corrections at evaluation points require `mu` values bounded away from 0 (|mu| >= 1e-8)."
                )
            if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
                raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")

            mask_pos = mu > 0
            mask_neg = ~mask_pos
            mu_pos = mu[mask_pos]
            abs_mu_neg = -mu[mask_neg]
            Nmu_pos = len(mu_pos)

            # NOTE: downward `mu` MAY equal `-mu0` or the IMS-scaled `-mu0`; the removable
            # singularities there are handled exactly by the phi-kernel formulations in
            # `_TMS_correction` and `_IMS_correction`

            # Atmospheric layer indices: smallest l with tau <= tau_arr[l]
            l = np.searchsorted(tau_arr, tau, side="left")

            # Delta-M scaling
            if np.any(scale_tau != 1):
                scaled_tau = scaled_tau_arr_with_0[l + 1] - (tau_arr[l] - tau) * scale_tau[l]
            else:
                scaled_tau = tau

            # The scattering angle cosines are shared by the TMS and IMS corrections
            nu = np.asarray(
                subroutines.calculate_nu(np.concatenate([mu_pos, -abs_mu_neg]), phi, -mu0, phi0)
            ).reshape(len(mu), len(phi))

            NT_corrections = _TMS_correction(
                mu_pos, abs_mu_neg, nu,
                l, scaled_tau,
                is_antiderivative_wrt_tau,
                statics["Leg_coeffs_diff_T"],
                scaled_omega_arr, I0_div_4pi, mu0,
                scale_tau, scaled_thickness_arr,
                scaled_tau_arr_with_0, statics["exp_scaled_tau_arr_with_0"],
                NLayers, is_atmos_multilayered,
                autograd_compatible,
            )
            if Nmu_pos < len(mu):
                NT_corrections[Nmu_pos:, :, :] += _IMS_correction(
                    abs_mu_neg, nu[Nmu_pos:], tau, l,
                    is_antiderivative_wrt_tau,
                    statics["scaled_mu0"], statics["IMS_scalar"], statics["IMS_Leg_coeffs"],
                    statics["tau_arr_with_0"],
                    autograd_compatible,
                )

            # Return to the original `mu` ordering
            NT_corrections_ordered = np.empty(np.shape(NT_corrections))
            NT_corrections_ordered[mask_pos] = NT_corrections[:Nmu_pos]
            NT_corrections_ordered[mask_neg] = NT_corrections[Nmu_pos:]
            return rescale_factor * NT_corrections_ordered
        # --------------------------------------------------------------------------------------------------------------------------

    else:
        _NT_corrections_at_mu = None

    if NT_cor and NT_corrections_possible:

        ############################### Perform NT corrections on the intensity but not the flux ###############################
        ############################### Refer to section 3.7.2 of the Comprehensive Documentation ##############################

        NT_statics = _get_NT_statics()

        # The corrected intensity: `u_star` plus the TMS and IMS corrections evaluated at the quadrature nodes
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False, return_tau_arr=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
                raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")

            # Atmospheric layer indices: smallest l with tau <= tau_arr[l]
            l = np.searchsorted(tau_arr, tau, side="left")

            # Delta-M scaling
            if np.any(scale_tau != 1):
                scaled_tau = scaled_tau_arr_with_0[l + 1] - (tau_arr[l] - tau) * scale_tau[l]
            else:
                scaled_tau = tau

            # The scattering angle cosines are shared by the TMS and IMS corrections
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )

            # TMS correction for the intensity (see section 3.7.2)
            NT_corrections = _TMS_correction(
                mu_arr_pos, mu_arr_pos, nu,
                l, scaled_tau,
                is_antiderivative_wrt_tau,
                NT_statics["Leg_coeffs_diff_T"],
                scaled_omega_arr, I0_div_4pi, mu0,
                scale_tau, scaled_thickness_arr,
                scaled_tau_arr_with_0, NT_statics["exp_scaled_tau_arr_with_0"],
                NLayers, is_atmos_multilayered,
                autograd_compatible,
                NT_statics["TMS_statics_quad_pos"],
                NT_statics["TMS_statics_quad_neg"],
            )

            # IMS correction for the intensity (see section 3.7.2); it applies only to downward directions
            IMS_corrections = _IMS_correction(
                mu_arr_pos, nu[N:], tau, l,
                is_antiderivative_wrt_tau,
                NT_statics["scaled_mu0"], NT_statics["IMS_scalar"], NT_statics["IMS_Leg_coeffs"],
                NT_statics["tau_arr_with_0"],
                autograd_compatible,
            )
            if autograd_compatible:
                NT_corrections = NT_corrections + np.concatenate(
                    [np.zeros((N, len(tau), len(phi))), IMS_corrections], axis=0
                )
            else:
                NT_corrections[N:, :, :] += IMS_corrections

            # The corrected intensity
            if return_Fourier_error or return_tau_arr:
                u_star_outputs = u_star(tau, phi, is_antiderivative_wrt_tau, return_Fourier_error, return_tau_arr)
                return (
                    u_star_outputs[0] + rescale_factor * np.squeeze(NT_corrections),
                ) + u_star_outputs[1:]
            else:
                return u_star(tau, phi, is_antiderivative_wrt_tau) + rescale_factor * np.squeeze(NT_corrections)
        # --------------------------------------------------------------------------------------------------------------------------

        u_out = u_corrected
    else:
        u_out = u_star

    # Attach the information required by `subroutines.interpolate` to apply or remove NT corrections
    # (see its `NT_cor` argument and GitHub issue #47). This changes neither the API nor the behavior
    # of `pydisort` itself.
    u_out._NT_data = {
        "NT_cor_applied": u_out is not u_star,
        "u_star": u_star,
        "corrections_available": NT_corrections_possible,
        "corrections_at_mu": _NT_corrections_at_mu,
    }

    return mu_arr, flux_up, flux_down, u0, u_out


def _TMS_mu_statics(M_inv_h, positive_hemisphere, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0):
    """Precomputes the decay and coefficient arrays used by the multi-layer accumulation scans
    in `_TMS_correction` for one hemisphere of `mu` values. These arrays are independent of the
    evaluation points `tau` and `phi` and so can be reused between evaluations.
    `M_inv_h` are the reciprocals of the absolute `mu` values of the hemisphere;
    `positive_hemisphere` selects between the upward (True) and downward (False) hemispheres.

    Arguments of _TMS_mu_statics
    |          Variable            |     Type / Shape     |
    | ---------------------------- | -------------------- |
    | `M_inv_h`                    | `H`                  |
    | `positive_hemisphere`        | boolean              |
    | `scaled_thickness_arr`       | `NLayers`            |
    | `mu0`                        | scalar               |
    | `exp_scaled_tau_arr_with_0`  | `NLayers + 1`        |
    """
    # decay[n, r] = exp(-scaled_thickness_arr[r] / |mu[n]|)
    delta = scaled_thickness_arr[None, :] * M_inv_h[:, None]                        # (H, NLayers)
    decay = np.exp(-delta)                                                          # (H, NLayers)

    # exp(-tau_front[r] / mu0) = exp_scaled_tau_arr_with_0[:-1]
    # exp(-tau_back[r] / mu0)  = exp_scaled_tau_arr_with_0[1:]
    if positive_hemisphere:
        # fac_pos = 1 - exp(-scaled_thickness_arr*(1/mu + 1/mu0)) = -expm1(-scaled_thickness_arr*(...))
        delta_pos = delta + (scaled_thickness_arr / mu0)[None, :]                   # (H, NLayers) >= 0
        fac_pos = -np.expm1(-delta_pos)                                             # (H, NLayers) in [0,1)
        # coef[n,r] multiplies exp((tau - tau_front[r])/mu - tau_front[r]/mu0) after re-referencing;
        # includes exp(-tau_front[r]/mu0)
        coef = fac_pos * exp_scaled_tau_arr_with_0[:-1][None, :]                    # (H, NLayers)
    else:
        # We need coef[n,r] = [exp(-tau_back[r]/mu0) - exp(-scaled_thickness_arr[r]/mu)*exp(-tau_front[r]/mu0)]
        # MULTIPLIED by the beam factor mu0/(mu0 - mu) = 1/(mu*y) of `mathscr_B` (see `_TMS_correction`),
        # whose singularity at mu = mu0 (y = 0) is removable: the bracket vanishes there too.
        # With w = scaled_thickness_arr * y, near the singularity (|w| < 1) the combination is
        # evaluated exactly through the kernel phi1(-w) = -expm1(-w)/w; away from it (|w| >= 1)
        # the bracket is computed stably via expm1 using only non-positive inputs and divided by y.
        w = delta - (scaled_thickness_arr / mu0)[None, :]                           # (H, NLayers) = thickness*y, can be +/-.
        y = M_inv_h - 1 / mu0                                                       # (H,)
        near_singularity = np.abs(w) < 1.0
        y_safe = np.where(y == 0.0, 1.0, y)[:, None]
        w_safe = np.where(w == 0.0, 1.0, w)
        phi1_neg_w = np.where(w == 0.0, 1.0, -np.expm1(-w_safe) / w_safe)           # phi1(-w)
        em1 = np.expm1(-np.abs(w))                                                  # (H, NLayers) in [-1,0], argument <= 0
        exp_x1 = exp_scaled_tau_arr_with_0[1:][None, :]                             # (1, NLayers)
        exp_x0 = decay * exp_scaled_tau_arr_with_0[:-1][None, :]                    # (H, NLayers)
        # if w>=0: bracket = exp_x1*(1-exp(-w)) = exp_x1*(-expm1(-w)) = exp_x1*(-em1)
        # if w<0 : bracket = exp_x0*(exp(w)-1)  = exp_x0*expm1(w)     = exp_x0*( em1)  (since w<0 -> em1=expm1(w))
        coef = M_inv_h[:, None] * np.where(
            near_singularity,
            exp_x1 * scaled_thickness_arr[None, :] * phi1_neg_w,
            np.where(w >= 0.0, -em1 * exp_x1, em1 * exp_x0) / y_safe,
        )                                                                           # (H, NLayers)

    return decay, coef


def _TMS_correction(
    mu_pos, abs_mu_neg,             # Positive-hemisphere `mu` values; absolute values of negative-hemisphere `mu`
    nu,                             # Scattering angle cosines (positive-hemisphere rows first)
    l, scaled_tau,                  # Layer indices and delta-scaled optical depths of the evaluation points
    is_antiderivative_wrt_tau,      # Switch to an antiderivative of the function?
    Leg_coeffs_diff_T,              # Transposed weighted difference of true and delta-scaled phase function coefficients
    scaled_omega_arr,               # Delta-scaled single-scattering albedos
    I0_div_4pi,                     # (Rescaled) intensity of the direct beam divided by 4 pi
    mu0,                            # Cosine of polar angle of the incident beam
    scale_tau,                      # Delta-scale factor for tau
    scaled_thickness_arr,           # Delta-scaled thickness of each layer
    scaled_tau_arr_with_0,          # Delta-scaled lower boundary of layers with 0 inserted in front
    exp_scaled_tau_arr_with_0,      # exp(-scaled_tau_arr_with_0 / mu0)
    NLayers,                        # Number of layers
    is_atmos_multilayered,          # Is the atmosphere multilayered?
    autograd_compatible,            # Should the output functions be compatible with autograd?
    statics_pos=None,               # Optional precomputed (decay, coef) for the positive hemisphere (multi-layer only)
    statics_neg=None,               # Optional precomputed (decay, coef) for the negative hemisphere (multi-layer only)
):
    """TMS correction to the intensity at arbitrary `mu` evaluation points.
    Refer to section 3.7.2 of the Comprehensive Documentation.
    Returns an ndarray of shape (Npos + Nneg, Ntau, Nphi) with the positive-hemisphere rows first.
    This function is private and should be called only through `pydisort`
    (directly, or through the closures `pydisort` creates).

    Arguments of _TMS_correction
    |          Variable            |         Type / Shape         |
    | ---------------------------- | ---------------------------- |
    | `mu_pos`                     | `Npos`                       |
    | `abs_mu_neg`                 | `Nneg`                       |
    | `nu`                         | `(Npos + Nneg) x Nphi`       |
    | `l`                          | `Ntau`                       |
    | `scaled_tau`                 | `Ntau`                       |
    | `Leg_coeffs_diff_T`          | `NLeg_all x NLayers`         |
    | `scaled_omega_arr`           | `NLayers`                    |
    | `scale_tau`                  | `NLayers`                    |
    | `scaled_thickness_arr`       | `NLayers`                    |
    | `scaled_tau_arr_with_0`      | `NLayers + 1`                |
    | `exp_scaled_tau_arr_with_0`  | `NLayers + 1`                |
    | `statics_pos`, `statics_neg` | 2-tuples of `H x NLayers`    |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np

    Nmu_pos = len(mu_pos)
    Nmu_neg = len(abs_mu_neg)
    Nphi = np.shape(nu)[-1]

    scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
    scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

    # mathscr_B
    # --------------------------------------------------------------------------------------------------------------------------
    # Vectorized Legendre evaluation: legval broadcasts the trailing-axis
    # coefficient set, returning (NLayers, Nmu, Nphi). Move the layer axis
    # to the middle to obtain the (Nmu, NLayers, Nphi) layout.
    # `legval` is linear in its coefficients, so evaluating the precomputed difference
    # coefficients gives p_true / (1 - f) - p_trun in a single call.
    p_diff = np.moveaxis(legval(nu, Leg_coeffs_diff_T, tensor=True), 0, 1)
    # The positive (upward) hemisphere keeps the (never singular) mu0 / (mu0 + mu) beam factor.
    # For the negative (downward) hemisphere the mu0 / (mu0 - mu) factor has a REMOVABLE
    # singularity at mu = mu0 which cancels against the bracketed differences of exponentials;
    # it is therefore folded into those brackets (below) and evaluated through the stable
    # kernel phi1(z) = (e^z - 1)/z, so `mathscr_B` itself carries no beam factor for this hemisphere.
    mathscr_B_fac = np.concatenate([mu0 / (mu0 + mu_pos), np.ones(Nmu_neg)])
    mathscr_B = (
        (scaled_omega_arr * I0_div_4pi)[None, :, None]
        * mathscr_B_fac[:, None, None]
        * p_diff
    )
    Bpos = mathscr_B[:Nmu_pos, :, :]                                    # (Npos, NLayers, Nphi)
    Bneg = mathscr_B[Nmu_pos:, :, :]                                    # (Nneg, NLayers, Nphi)
    # --------------------------------------------------------------------------------------------------------------------------

    exp_neg_scaled_tau_div_mu0 = np.exp(-scaled_tau / mu0)              # (Ntau,)

    # Negative-hemisphere phi1 kernel quantities. With y = 1/mu - 1/mu0 (which vanishes at
    # the mu = mu0 singularity), take the optical distance difference (Delta) from the layer top 
    # and w = Delta * y
    # For |w| < 1 the phi1 form (via expm1) is exact including AT w = 0;
    # for |w| >= 1 there is no cancellation and the regular, direct form is used instead. 
    # This replaces the former requirement that mu0 be bounded away from the quadrature / evaluation angles.
    y_neg = 1 / abs_mu_neg - 1 / mu0                                    # (Nneg,)
    Delta = scaled_tau - scaled_tau_arr_lm1                             # (Ntau,)
    w = Delta[None, :] * y_neg[:, None]                                 # (Nneg, Ntau)
    near_singularity = np.abs(w) < 1.0
    y_safe = np.where(y_neg == 0.0, 1.0, y_neg)[:, None]
    w_safe = np.where(w == 0.0, 1.0, w)
    phi1_neg_w = np.where(w == 0.0, 1.0, -np.expm1(-w_safe) / w_safe)   # phi1(-w)
    E1 = exp_neg_scaled_tau_div_mu0[None, :]
    E2 = np.exp(
        (scaled_tau_arr_lm1 - scaled_tau)[None, :] / abs_mu_neg[:, None]
        - scaled_tau_arr_lm1[None, :] / mu0
    )

    if is_antiderivative_wrt_tau:
        scale_tau_l = scale_tau[l]                                      # (Ntau,)
        common_term = (exp_neg_scaled_tau_div_mu0 / (-scale_tau_l / mu0))[None, :]

        TMS_correction_pos = common_term - np.exp(
            (scaled_tau - scaled_tau_arr_l)[None, :] / mu_pos[:, None]
            - scaled_tau_arr_l[None, :] / mu0
        ) * (mu_pos[:, None] / scale_tau_l[None, :])
        # Antiderivative of the phi1-kernel form above
        TMS_correction_neg = np.where(
            near_singularity,
            -(E1 / scale_tau_l[None, :]) * (Delta[None, :] * phi1_neg_w + mu0),
            (abs_mu_neg[:, None] * E2 - mu0 * E1)
            / (scale_tau_l[None, :] * abs_mu_neg[:, None] * y_safe),
        )
    else:
        TMS_correction_pos = exp_neg_scaled_tau_div_mu0[None, :] - np.exp(
            (scaled_tau - scaled_tau_arr_l)[None, :] / mu_pos[:, None]
            - scaled_tau_arr_l[None, :] / mu0
        )
        TMS_correction_neg = np.where(
            near_singularity,
            (Delta[None, :] / abs_mu_neg[:, None]) * phi1_neg_w * E1,
            (E1 - E2) / (abs_mu_neg[:, None] * y_safe),
        )

    sol_pos = Bpos[:, l, :] * TMS_correction_pos[:, :, None]            # (Npos, Ntau, Nphi)
    sol_neg = Bneg[:, l, :] * TMS_correction_neg[:, :, None]            # (Nneg, Ntau, Nphi)

    if not is_atmos_multilayered:
        return np.concatenate((sol_pos, sol_neg), axis=0)

    # ---------------- other-layer contributions ----------------
    any_pos = (l.min() < NLayers - 1)  # exists tau with layers below
    any_neg = (l.max() > 0)            # exists tau with layers above

    # ===== POS: sum over r > l =====
    if any_pos and Nmu_pos > 0:
        M_inv_pos = 1 / mu_pos
        if statics_pos is None:
            decay_pos, coef_pos = _TMS_mu_statics(
                M_inv_pos, True, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0
            )
        else:
            decay_pos, coef_pos = statics_pos

        l_min = int(l.min())  # only Acc_pos at layers >= l_min is ever indexed
        Nacc = NLayers - l_min

        # NOTE: antiderivative target-layer factor `mu/scale_tau[l]` is applied
        # OUTSIDE the accumulator, not per-source-layer inside it.

        # Reverse scan to build Acc_l (referenced at tau_back[l]) INCLUDING Bpos:
        # Acc_{r-1} = coef_pos[:,r]*Bpos[:,r,:] + decay_pos[:,r]*Acc_r
        # Stored array covers layers [l_min..NLayers-1]; layer r maps to index r-l_min.
        if autograd_compatible:
            acc0 = np.zeros((Nmu_pos, Nphi))
            acc = acc0
            cols_rev = [acc0]  # Acc_{NLayers-1} = 0
            for r in range(NLayers - 1, l_min, -1):
                term = coef_pos[:, r][:, None] * Bpos[:, r, :]          # (Npos, Nphi)
                acc = term + decay_pos[:, r][:, None] * acc
                cols_rev.append(acc)
            Acc_pos = np.stack(cols_rev[::-1], axis=1)                  # (Npos, Nacc, Nphi)
        else:
            Acc_pos = np.empty((Nmu_pos, Nacc, Nphi))
            Acc_pos[:, Nacc - 1, :] = 0.0                               # Acc_{NLayers-1}
            for r in range(NLayers - 1, l_min, -1):
                i_dest = r - 1 - l_min
                i_source = r - l_min
                np.multiply(coef_pos[:, r:r+1], Bpos[:, r, :],
                            out=Acc_pos[:, i_dest, :])
                Acc_pos[:, i_dest, :] += decay_pos[:, r:r+1] * Acc_pos[:, i_source, :]

        # Apply to tau: multiply by exp((scaled_tau - tau_back[l]) / mu) (argument <= 0 in exact arithmetic)
        expfac_pos = np.exp(M_inv_pos[:, None] * (scaled_tau - scaled_tau_arr_l)[None, :])  # (Npos, Ntau)
        contrib_pos = Acc_pos[:, l - l_min, :] * expfac_pos[:, :, None]                     # (Npos, Ntau, Nphi)

        if is_antiderivative_wrt_tau:
            contrib_pos = contrib_pos * (mu_pos[:, None] / scale_tau_l[None, :])[:, :, None]

        sol_pos = sol_pos + contrib_pos

    # ===== NEG: sum over r < l =====
    if any_neg and Nmu_neg > 0:
        M_inv_neg = 1 / abs_mu_neg
        if statics_neg is None:
            decay_neg, coef_neg = _TMS_mu_statics(
                M_inv_neg, False, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0
            )
        else:
            decay_neg, coef_neg = statics_neg

        l_max = int(l.max())  # only Acc_neg at layers <= l_max is ever indexed
        Nacc = l_max + 1

        # NOTE: antiderivative target-layer factor `-mu/scale_tau[l]` is applied
        # OUTSIDE the accumulator, not per-source-layer inside it.

        # Forward scan to build Acc_l (referenced at tau_front[l]) INCLUDING Bneg:
        # Acc_{r+1} = coef_neg[:,r]*Bneg[:,r,:] + decay_neg[:,r]*Acc_r
        # Stored array covers layers [0..l_max]; layer index unchanged.
        if autograd_compatible:
            acc0 = np.zeros((Nmu_neg, Nphi))
            acc = acc0
            cols = [acc0]  # Acc_0 = 0
            for r in range(0, l_max):
                term = coef_neg[:, r][:, None] * Bneg[:, r, :]          # (Nneg, Nphi)
                acc = term + decay_neg[:, r][:, None] * acc
                cols.append(acc)
            Acc_neg = np.stack(cols, axis=1)                            # (Nneg, Nacc, Nphi)
        else:
            Acc_neg = np.empty((Nmu_neg, Nacc, Nphi))
            Acc_neg[:, 0, :] = 0.0                                      # Acc_0
            for r in range(0, l_max):
                np.multiply(coef_neg[:, r:r+1], Bneg[:, r, :],
                            out=Acc_neg[:, r + 1, :])
                Acc_neg[:, r + 1, :] += decay_neg[:, r:r+1] * Acc_neg[:, r, :]

        # Apply to tau: multiply by exp((tau_front[l] - scaled_tau) / mu) (argument <= 0 in exact arithmetic)
        expfac_neg = np.exp(M_inv_neg[:, None] * (scaled_tau_arr_lm1 - scaled_tau)[None, :])  # (Nneg, Ntau)
        contrib_neg = Acc_neg[:, l, :] * expfac_neg[:, :, None]                               # (Nneg, Ntau, Nphi)

        if is_antiderivative_wrt_tau:
            contrib_neg = -contrib_neg * (abs_mu_neg[:, None] / scale_tau_l[None, :])[:, :, None]

        sol_neg = sol_neg + contrib_neg

    return np.concatenate((sol_pos, sol_neg), axis=0)


# Coefficients of the Taylor series of phi2(z) = (e^z - 1 - z)/z^2 = sum_k z^k / (k + 2)!
# for k = 0, ..., 16; the truncation error at |z| = 1 is ~1/19! ~ 8e-18 (below machine precision) -- tested by Fable 5
_PHI2_TAYLOR_COEFFS = 1.0 / np.cumprod(np.arange(2, 19))


def _IMS_correction(
    abs_mu_neg,                     # Absolute values of negative-hemisphere `mu`
    nu_neg,                         # Scattering angle cosines for the negative hemisphere
    tau,                            # Optical depths of the evaluation points (NOT delta-scaled)
    l,                              # Atmospheric layer indices of the evaluation points
    is_antiderivative_wrt_tau,      # Switch to an antiderivative of the function?
    scaled_mu0,                     # mu0 / (1 - omega_f_avg) at each layer boundary
    IMS_scalar,                     # I0 / (4 pi) * omega_f_avg^2 / (1 - omega_f_avg) at each layer boundary
    IMS_Leg_coeffs,                 # (2l + 1) * (2 * residue_avg - residue_avg^2) at each layer boundary
    tau_arr_with_0,                 # Unscaled layer boundaries with 0 prepended
    autograd_compatible,            # Should the output functions be compatible with autograd?
):
    """IMS correction to the intensity at arbitrary downward `mu` evaluation points.
    Refer to section 3.7.2 of the Comprehensive Documentation.
    Returns an ndarray of shape (Nneg, Ntau, Nphi).
    This function is private and should be called only through `pydisort`
    (directly, or through the closures `pydisort` creates).

    The optical-property averages follow DISORT's `SECSCA` (top of the atmosphere down
    to the output depth) evaluated at the layer boundaries k = 0, ..., NLayers, giving
    per-boundary correction functions
        y_k(tau) = IMS_scalar_k * PSPIKE_k(nu) * chi(tau; scaled_mu0_k).
    Within layer l (0-indexed; tau in (tau_arr_with_0[l], tau_arr_with_0[l + 1]]) the
    correction blends the enclosing boundary sets linearly in tau,
        u_IMS(tau) = y_l(tau) + w(tau) * (y_{l+1}(tau) - y_l(tau)),
        w(tau) = (tau - tau_arr_with_0[l]) / (tau_arr_with_0[l + 1] - tau_arr_with_0[l]),
    which matches DISORT's tau-dependent averaging exactly at every layer boundary and
    throughout the first layer, is continuous in tau, and preserves the closed-form
    per-layer antiderivatives (unlike DISORT's averaging, under which the tau-dependence
    of the averages makes the antiderivative non-elementary). When all layers are
    identical -- in particular for single-layer atmospheres, where the k = 0 set is a
    copy of the k = 1 set -- the boundary sets coincide, the blend differences vanish
    exactly and u_IMS reduces to the unblended y.

    Arguments of _IMS_correction
    |          Variable            |          Type / Shape        |
    | ---------------------------- | ---------------------------- |
    | `abs_mu_neg`                 | `Nneg`                       |
    | `nu_neg`                     | `Nneg x Nphi`                |
    | `tau`                        | `Ntau`                       |
    | `l`                          | `Ntau`                       |
    | `scaled_mu0`                 | `NLayers + 1`                |
    | `IMS_scalar`                 | `NLayers + 1`                |
    | `IMS_Leg_coeffs`             | `(NLayers + 1) x NLeg_all`   |
    | `tau_arr_with_0`             | `NLayers + 1`                |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np

    # chi has a REMOVABLE singularity at x = 0 (mu = scaled_mu0). Substituting
    # exp(-tau/mu) = exp(-tau/scaled_mu0) * exp(z) with z = -tau * x collapses chi to
    #     chi      = tau^2 * exp(-tau/scaled_mu0) * phi2(z) / (mu * scaled_mu0)
    #     chi_anti = -exp(-tau/scaled_mu0) * (tau^2 * phi2(z) + scaled_mu0*(scaled_mu0 + tau)) / scaled_mu0
    # and (exactly, for the antiderivative of chi_anti needed by the blended antiderivative below)
    #     chichi_anti = exp(-tau/scaled_mu0) * (tau^2 * phi2(z) + 2*tau*scaled_mu0 + 3*scaled_mu0^2
    #                                           + x*scaled_mu0^2*(tau + 2*scaled_mu0)) / (1 + x*scaled_mu0)
    # where phi2(z) = (e^z - 1 - z)/z^2 with phi2(0) = 1/2, and 1 + x*scaled_mu0 = scaled_mu0/mu > 0.
    # The direct formulas below suffer catastrophic cancellation for small |z| (losing ~half the
    # digits at |z| ~ 1e-8), so for |z| < 1 phi2 is evaluated by its truncated Taylor series
    # sum_k z^k / (k+2)! which is exact to machine precision there; both branches are accurate
    # to ~1 ulp at the |z| ~ 1 crossover. This replaces the former requirement that no
    # quadrature / evaluation angle coincide with `scaled_mu0`.
    exp_tau_mu = np.exp(-tau[None, :] / abs_mu_neg[:, None])            # (Nneg, Ntau)

    def chi_kernels(smu0, need_chichi_anti=True):
        """chi (or, on the antiderivative branch, chi_anti and optionally chichi_anti) of
        shape (Nneg, Ntau) for one gathered set of per-evaluation-point statics `smu0` (Ntau,)."""
        x = 1 / abs_mu_neg[:, None] - 1 / smu0[None, :]                 # (Nneg, Ntau)
        z = -tau[None, :] * x
        near_singularity = np.abs(z) < 1.0
        x_safe = np.where(x == 0.0, 1.0, x)
        exp_tau_smu0 = np.exp(-tau / smu0)[None, :]
        phi2 = _PHI2_TAYLOR_COEFFS[-1]
        for c in _PHI2_TAYLOR_COEFFS[-2::-1]:                           # Horner evaluation; written to be autograd-compatible
            phi2 = phi2 * z + c

        if is_antiderivative_wrt_tau:
            chi_anti = np.where(
                near_singularity,
                -exp_tau_smu0
                * (tau[None, :] ** 2 * phi2 + (smu0 * (smu0 + tau))[None, :])
                / smu0[None, :],
                (
                    (smu0[None, :] - x_safe * (smu0 * (smu0 + tau))[None, :]) * exp_tau_smu0
                    - abs_mu_neg[:, None] * exp_tau_mu
                ) / (abs_mu_neg[:, None] * smu0[None, :] * x_safe**2),
            )
            if not need_chichi_anti:
                return chi_anti, None
            chichi_anti = np.where(
                near_singularity,
                exp_tau_smu0
                * (
                    tau[None, :] ** 2 * phi2
                    + (2 * smu0 * tau + 3 * smu0**2)[None, :]
                    + x * (smu0**2 * (tau + 2 * smu0))[None, :]
                )
                / (1 + x * smu0[None, :]),
                (
                    abs_mu_neg[:, None] ** 2 * exp_tau_mu
                    - (smu0**2)[None, :] * (1 - x * (tau + 2 * smu0)[None, :]) * exp_tau_smu0
                ) / (abs_mu_neg[:, None] * smu0[None, :] * x_safe**2),
            )
            return chi_anti, chichi_anti
        else:
            chi = np.where(
                near_singularity,
                tau[None, :] ** 2 * exp_tau_smu0 * phi2 / (abs_mu_neg[:, None] * smu0[None, :]),
                (
                    (tau[None, :] - 1 / x_safe) * exp_tau_smu0 + exp_tau_mu / x_safe
                ) / (abs_mu_neg[:, None] * smu0[None, :] * x_safe),
            )
            return chi

    if len(IMS_scalar) == 2:
        # Single-layer atmosphere: the two boundary sets are identical, so skip the blend
        # (this path costs the same as the pre-blending whole-atmosphere implementation
        # and produces identical results)
        smu0 = scaled_mu0[np.full(len(tau), 1)]
        scalP = (IMS_scalar[1] * legval(nu_neg, IMS_Leg_coeffs[1]))[:, None, :]
        if is_antiderivative_wrt_tau:
            return scalP * chi_kernels(smu0, need_chichi_anti=False)[0][:, :, None]
        else:
            return scalP * chi_kernels(smu0)[:, :, None]

    # Blend the enclosing boundary sets k = l ("lo", averages down to the top of the
    # layer of each evaluation point) and k = l + 1 ("hi", averages down to its bottom)
    t0 = tau_arr_with_0[l]                                              # (Ntau,)
    thickness_l = tau_arr_with_0[l + 1] - t0                            # (Ntau,)

    # PSPIKE (the Legendre series in the scattering angle) for each needed boundary set,
    # evaluated in a single vectorized `legval` call
    sets = np.unique(np.concatenate([l, l + 1]))
    P_sets = legval(nu_neg, np.ascontiguousarray(IMS_Leg_coeffs[sets].T), tensor=True)  # (Nsets, Nneg, Nphi)
    scalP_lo = IMS_scalar[l][None, :, None] * np.moveaxis(
        P_sets[np.searchsorted(sets, l)], 0, 1
    )                                                                   # (Nneg, Ntau, Nphi)
    scalP_hi = IMS_scalar[l + 1][None, :, None] * np.moveaxis(
        P_sets[np.searchsorted(sets, l + 1)], 0, 1
    )                                                                   # (Nneg, Ntau, Nphi)

    if is_antiderivative_wrt_tau:
        # With w(tau) linear in tau, the per-layer antiderivative of the blend is
        #     F = Y1_lo + ((tau - t0) * (Y1_hi - Y1_lo) - (Y2_hi - Y2_lo)) / thickness_l
        # where Y1_k = scalP_k * chi_anti_k and Y2_k = scalP_k * chichi_anti_k satisfy
        # Y1_k' = y_k and Y2_k' = Y1_k; indeed F' = y_lo + w * (y_hi - y_lo) as the Y1_hi - Y1_lo
        # terms cancel. Like the rest of the NT corrections, F is a LOCAL (per-layer)
        # antiderivative to be consumed through the documented telescoping idiom.
        chi_anti_lo, chichi_anti_lo = chi_kernels(scaled_mu0[l])
        chi_anti_hi, chichi_anti_hi = chi_kernels(scaled_mu0[l + 1])
        Y1_lo = scalP_lo * chi_anti_lo[:, :, None]
        Y1_hi = scalP_hi * chi_anti_hi[:, :, None]
        Y2_diff = scalP_hi * chichi_anti_hi[:, :, None] - scalP_lo * chichi_anti_lo[:, :, None]
        return Y1_lo + (
            (tau - t0)[None, :, None] * (Y1_hi - Y1_lo) - Y2_diff
        ) / thickness_l[None, :, None]
    else:
        w = (tau - t0) / thickness_l                                    # (Ntau,)
        y_lo = scalP_lo * chi_kernels(scaled_mu0[l])[:, :, None]
        y_hi = scalP_hi * chi_kernels(scaled_mu0[l + 1])[:, :, None]
        return y_lo + w[None, :, None] * (y_hi - y_lo)
