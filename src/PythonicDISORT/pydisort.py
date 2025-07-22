from PythonicDISORT import subroutines
from PythonicDISORT._assemble_intensity_and_fluxes import _assemble_intensity_and_fluxes
import warnings

import numpy as np
import scipy as sc
from math import pi
from numpy.polynomial.legendre import Legendre
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
        We recommend setting ``f_arr`` to ``Leg_coeffs_all[NQuad]``, 
        or ``Leg_coeffs_all[:, NQuad]`` for a multi-layer atmosphere.
    NT_cor : optional, bool
        Perform Nakajima-Tanaka intensity corrections?
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

    Returns
    -------
    mu_arr : array
        All ``mu`` (cosine of polar angle) quadrature nodes.
    Fp(tau) : function
        (Energetic) Flux function with argument ``tau`` (type: array or float) for positive (upward) ``mu`` values.
        Returns the diffuse flux magnitudes (same type and size as ``tau``).
        Pass ``is_antiderivative_wrt_tau = True`` (defaults to ``False``)
        to switch to an antiderivative of this function with respect to ``tau``.
        For example, ``Fp(tau_arr, is_antiderivative_wrt_tau = True) - Fp(np.insert(tau_arr[:-1] + 1e-15, 0, 0), is_antiderivative_wrt_tau = True)``
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
    b_pos_is_scalar = False
    b_neg_is_scalar = False
    b_pos_is_vector = False
    b_neg_is_vector = False
    thickness_arr = np.insert(np.diff(tau_arr), 0, tau_arr[0])
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
    if not (np.all(-1 <= Leg_coeffs_all) and np.all(Leg_coeffs_all <= 1)):
        raise ValueError("The phase function Legendre coefficients must all be between -1 and 1.")
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
    
    # We do not allow mu0 to equal a quadrature / computational angle if `NT_cor = True`
    if NT_cor and np.any(np.abs(mu_arr_pos - mu0) < 1e-8):
        raise ValueError("Some quadrature angles come too close to `mu0`. Perturb `NQuad` or `mu0` to rectify this error.")
    # --------------------------------------------------------------------------------------------------------------------------

    # Delta-M scaling; there is no scaling if f = 0
    # Refer to sections 1.3.1 and 3.3 of the Comprehensive Documentation
    # --------------------------------------------------------------------------------------------------------------------------
    if np.any(f_arr > 0):

        scale_tau = 1 - omega_arr * f_arr
        scaled_thickness_arr = scale_tau * thickness_arr
        scaled_tau_arr_with_0 = np.insert(np.cumsum(scaled_thickness_arr), 0, 0)
        scaled_Leg_coeffs = (Leg_coeffs - f_arr[:, None]) / (1 - f_arr[:, None])
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = (1 - f_arr) / scale_tau * omega_arr

        translations = scaled_tau_arr_with_0[:-1] - scale_tau * np.insert(tau_arr[:-1], 0, 0)
        scaled_s_poly_coeffs = (
            subroutines.affine_transform_poly_coeffs(s_poly_coeffs, scale_tau, translations)
            / scale_tau[:, None]  # Divide by d\tau^* / d\tau
        ) * (1 - omega_arr[:, None])  # Enforce Kirchoff's Law of Thermal Radiation: absorptivity equals emissivity

    else:
        # This is a shortcut to the same results
        scale_tau = np.ones(NLayers)
        scaled_tau_arr_with_0 = np.insert(tau_arr, 0, 0)
        scaled_Leg_coeffs = Leg_coeffs
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = omega_arr
        scaled_s_poly_coeffs = s_poly_coeffs * (1 - omega_arr[:, None])  # Enforce Kirchoff's Law of Thermal Radiation: absorptivity equals emissivity
        
    if np.any(scaled_omega_arr > 1 - 1e-6):
        warnings.warn("Some delta-scaled single-scattering albedos are very close to 1 which may cause numerical instability.")
    if (np.any(-0.95 > scaled_Leg_coeffs[:, 1:]) and np.any(scaled_Leg_coeffs[:, 1:] > 0.95)):
        warnings.warn("Some delta-scaled phase function Legendre coefficients have a magnitude that is very close to 1" +
        " (this excludes the zeroth index coefficient which must be 1) and this may cause numerical instability.")
    
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
    # --------------------------------------------------------------------------------------------------------------------------
    
    if NT_cor and not only_flux and there_is_beam_source and np.any(f_arr > 0) and NLeg < NLeg_all and np.any(omega_arr > 0):
        
        ############################### Perform NT corrections on the intensity but not the flux ###############################
        ############################### Refer to section 3.7.2 of the Comprehensive Documentation ##############################
        
        # Delta-M scaled solution; no further corrections to the flux
        flux_up, flux_down, u0, u_star = _assemble_intensity_and_fluxes(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr,
            M_inv, W,
            N, NQuad, NLeg,
            NFourier, NLayers, NBDRF,
            is_atmos_multilayered,
            weighted_scaled_Leg_coeffs,
            BDRF_Fourier_modes,
            mu0, I0, rescale_factor, phi0,
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
        )
        
        # TMS correction for the intensity (see section 3.7.2)
        # --------------------------------------------------------------------------------------------------------------------------
        def TMS_correction(tau, phi, is_antiderivative_wrt_tau):
            Ntau = len(tau)
            Nphi = len(phi)
            # Atmospheric layer indices
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
            scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

            # Delta-M scaling
            if np.any(scale_tau != np.ones(NLayers)):
                tau_dist_from_top = tau_arr[l] - tau
                scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
                scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top
            else:
                scaled_tau = tau
            
            # mathscr_B
            # --------------------------------------------------------------------------------------------------------------------------
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )
            p_true = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_Leg_coeffs_all)
                ],
                axis=1,
            )
            p_trun = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_scaled_Leg_coeffs)
                ],
                axis=1,
            )
            mathscr_B = (
                (scaled_omega_arr * I0)[None, :, None]
                / (4 * np.pi)
                * (mu0 / (mu0 + mu_arr))[:, None, None]
                * (p_true / (1 - f_arr[None, :, None]) - p_trun)
            )
            # --------------------------------------------------------------------------------------------------------------------------
            
            if is_antiderivative_wrt_tau:
                TMS_correction_pos = (
                    np.exp(-scaled_tau / mu0)[None, :] / (-scale_tau / mu0)[None, l]
                    - np.exp(
                        (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                        - scaled_tau_arr_l[None, :] / mu0
                    ) / (scale_tau[None, l] / mu_arr_pos[:, None])
                )
                TMS_correction_neg = (
                    np.exp(-scaled_tau / mu0)[None, :] / (-scale_tau / mu0)[None, l]
                    - np.exp(
                        (scaled_tau_arr_lm1 - scaled_tau)[None, :] / mu_arr_pos[:, None]
                        - scaled_tau_arr_lm1[None, :] / mu0
                    ) / (-scale_tau[None, l] / mu_arr_pos[:, None])
                )
            else:
                TMS_correction_pos = (
                    np.exp(-scaled_tau / mu0)[None, :]
                    - np.exp(
                        (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                        - scaled_tau_arr_l[None, :] / mu0
                    )
                )
                TMS_correction_neg = (
                    np.exp(-scaled_tau / mu0)[None, :]
                    - np.exp(
                        (scaled_tau_arr_lm1 - scaled_tau)[None, :] / mu_arr_pos[:, None]
                        - scaled_tau_arr_lm1[None, :] / mu0
                    )
                )
            
            ## Contribution from other layers
            # TODO: Despite being vectorized and despite many attempts at further optimization,
            # this block of code is slower than desired and is the bottleneck for the TMS correction.
            # --------------------------------------------------------------------------------------------------------------------------
            solution = (
                mathscr_B[:, l, :] * np.vstack((TMS_correction_pos, TMS_correction_neg))[:, :, None]
            )
            
            if is_atmos_multilayered:
                layers_arr = np.arange(NLayers)[:, None]
                pos_contribution_mask = (l[None, :] < layers_arr).ravel()
                neg_contribution_mask = (l[None, :] > layers_arr).ravel()
                any_pos_contribution = np.any(pos_contribution_mask)
                any_neg_contribution = np.any(neg_contribution_mask)
            
                if not (any_pos_contribution or any_neg_contribution):
                    return solution

                layers_arr_repeat = np.repeat(layers_arr, Ntau)
                scaled_tau_tile = np.tile(scaled_tau, NLayers)

                if any_pos_contribution:
                    contribution_from_each_other_layer_pos = np.zeros((N, NLayers * Ntau, Nphi))
                    scaled_tau_tile_pos = scaled_tau_tile[pos_contribution_mask]
                    
                    layers_arr_repeat_pos = layers_arr_repeat[pos_contribution_mask]
                    mathscr_B_repeat_pos = mathscr_B[:N, layers_arr_repeat_pos, :]
                    scaled_tau_arr_with_0_l_repeat_pos = scaled_tau_arr_with_0[:-1][layers_arr_repeat_pos]
                    scaled_tau_arr_with_0_lp1_repeat_pos = scaled_tau_arr_with_0[1:][layers_arr_repeat_pos]
                    
                    if is_antiderivative_wrt_tau:
                        scale_tau_repeat_pos = scale_tau[layers_arr_repeat_pos]
                        contribution_from_each_other_layer_pos[:, pos_contribution_mask, :] = (
                            mathscr_B_repeat_pos
                            * (
                                (
                                    np.exp(
                                        (scaled_tau_tile_pos - scaled_tau_arr_with_0_l_repeat_pos)
                                        / mu_arr_pos[:, None]
                                        - scaled_tau_arr_with_0_l_repeat_pos / mu0
                                    )
                                    - np.exp(
                                        (scaled_tau_tile_pos - scaled_tau_arr_with_0_lp1_repeat_pos)
                                        / mu_arr_pos[:, None]
                                        - scaled_tau_arr_with_0_lp1_repeat_pos / mu0
                                    )
                                )
                                / (scale_tau_repeat_pos / mu_arr_pos[:, None])
                            )[:, :, None]
                        )
                        contribution_from_other_layers_pos = np.sum(
                            contribution_from_each_other_layer_pos.reshape((N, NLayers, Ntau, Nphi)),
                            axis=1,
                        )
                    else:
                        contribution_from_each_other_layer_pos[:, pos_contribution_mask, :] = (
                            mathscr_B_repeat_pos
                            * (
                                np.exp(
                                    (scaled_tau_tile_pos - scaled_tau_arr_with_0_l_repeat_pos)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0_l_repeat_pos / mu0
                                )
                                - np.exp(
                                    (scaled_tau_tile_pos - scaled_tau_arr_with_0_lp1_repeat_pos)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0_lp1_repeat_pos / mu0
                                )
                            )[:, :, None]
                        )
                        contribution_from_other_layers_pos = np.sum(
                            contribution_from_each_other_layer_pos.reshape((N, NLayers, Ntau, Nphi)),
                            axis=1,
                        )
                        
                if any_neg_contribution:
                    contribution_from_each_other_layer_neg = np.zeros((N, NLayers * Ntau, Nphi))
                    scaled_tau_tile_neg = scaled_tau_tile[neg_contribution_mask]
                    
                    layers_arr_repeat_neg = layers_arr_repeat[neg_contribution_mask]
                    mathscr_B_repeat_neg = mathscr_B[N:, layers_arr_repeat_neg, :]
                    scaled_tau_arr_with_0_l_repeat_neg = scaled_tau_arr_with_0[:-1][layers_arr_repeat_neg]
                    scaled_tau_arr_with_0_lp1_repeat_neg = scaled_tau_arr_with_0[1:][layers_arr_repeat_neg]

                    if is_antiderivative_wrt_tau:
                        scale_tau_repeat_neg = scale_tau[layers_arr_repeat_neg]
                        contribution_from_each_other_layer_neg[:, neg_contribution_mask, :] = (
                            mathscr_B_repeat_neg
                            * (
                                (
                                    np.exp(
                                        (scaled_tau_arr_with_0_lp1_repeat_neg - scaled_tau_tile_neg)
                                        / mu_arr_pos[:, None]
                                        - scaled_tau_arr_with_0_lp1_repeat_neg / mu0
                                    )
                                    - np.exp(
                                        (scaled_tau_arr_with_0_l_repeat_neg - scaled_tau_tile_neg)
                                        / mu_arr_pos[:, None]
                                        - scaled_tau_arr_with_0_l_repeat_neg / mu0
                                    )
                                )
                                / (-scale_tau_repeat_neg / mu_arr_pos[:, None])
                            )[:, :, None]
                        )
                        contribution_from_other_layers_neg = np.sum(
                            contribution_from_each_other_layer_neg.reshape((N, NLayers, Ntau, Nphi)),
                            axis=1,
                        )
                    else:
                        contribution_from_each_other_layer_neg[:, neg_contribution_mask, :] = (
                            mathscr_B_repeat_neg
                            * (
                                np.exp(
                                    (scaled_tau_arr_with_0_lp1_repeat_neg - scaled_tau_tile_neg)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0_lp1_repeat_neg / mu0
                                )
                                - np.exp(
                                    (scaled_tau_arr_with_0_l_repeat_neg - scaled_tau_tile_neg)
                                    / mu_arr_pos[:, None]
                                    - scaled_tau_arr_with_0_l_repeat_neg / mu0
                                )
                            )[:, :, None]
                        )
                        contribution_from_other_layers_neg = np.sum(
                            contribution_from_each_other_layer_neg.reshape((N, NLayers, Ntau, Nphi)),
                            axis=1,
                        )

                if autograd_compatible:
                    if not any_pos_contribution:
                        TMS_correction_pos = np.zeros((N, Ntau, Nphi))
                    if not any_neg_contribution:
                        TMS_correction_neg = np.zeros((N, Ntau, Nphi))
                    return solution + np.concatenate(
                        (contribution_from_other_layers_pos, contribution_from_other_layers_neg),
                        axis=0,
                    )
                else:
                    if any_pos_contribution:
                        solution[:N, :, :] += contribution_from_other_layers_pos
                    if any_neg_contribution:
                        solution[N:, :, :] += contribution_from_other_layers_neg
                    return solution
                    
            # --------------------------------------------------------------------------------------------------------------------------
            
            else:
                return solution
        # --------------------------------------------------------------------------------------------------------------------------

        # IMS correction for the intensity (see section 3.7.2)
        # --------------------------------------------------------------------------------------------------------------------------
        sum1 = np.sum(omega_arr * tau_arr)
        omega_avg = sum1 / np.sum(tau_arr)
        sum2 = np.sum(f_arr * omega_arr * tau_arr)
        f_avg = sum2 / sum1
        Leg_coeffs_residue = Leg_coeffs_all.copy()
        Leg_coeffs_residue[:, :NLeg] = np.tile(f_arr, (NLeg, 1)).T
        Leg_coeffs_residue_avg = (
            np.sum(Leg_coeffs_residue * omega_arr[:, None] * tau_arr[:, None], axis=0)
            / sum2
        )
        scaled_mu0 = mu0 / (1 - omega_avg * f_avg)

        def IMS_correction(tau, phi, is_antiderivative_wrt_tau):
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(-mu_arr_pos, phi, -mu0, phi0)
            )
            x = 1 / mu_arr_pos - 1 / scaled_mu0
            if is_antiderivative_wrt_tau:
                chi = (
                    (scaled_mu0 - x[:, None] * scaled_mu0 * (scaled_mu0 + tau)[None, :])
                    * np.exp(-tau / scaled_mu0)[None, :]
                    - mu_arr_pos[:, None] * np.exp(-tau[None, :] / mu_arr_pos[:, None])
                ) / (mu_arr_pos * scaled_mu0 * x**2)[:, None]
            else:
                chi = (
                    (tau[None, :] - 1 / x[:, None]) * np.exp(-tau / scaled_mu0)[None, :]
                    + np.exp(-tau[None, :] / mu_arr_pos[:, None]) / x[:, None]
                ) / (mu_arr_pos * scaled_mu0 * x)[:, None]
                
            return (
                I0
                / (4 * pi)
                * (omega_avg * f_avg) ** 2
                / (1 - omega_avg * f_avg)
                * Legendre(
                    (2 * np.arange(NLeg_all) + 1)
                    * (2 * Leg_coeffs_residue_avg - Leg_coeffs_residue_avg**2)
                )(nu)
            )[:, None, :] * chi[:, :, None]
        # --------------------------------------------------------------------------------------------------------------------------

        # The corrected intensity
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False, return_tau_arr=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            NT_corrections = TMS_correction(tau, phi, is_antiderivative_wrt_tau)
            
            if autograd_compatible:
                NT_corrections = NT_corrections + np.concatenate(
                    [np.zeros((N, len(tau), len(phi))), IMS_correction(tau, phi, is_antiderivative_wrt_tau)], axis=0
                )
            else:
                NT_corrections[N:, :, :] += IMS_correction(tau, phi, is_antiderivative_wrt_tau)
            
            if return_Fourier_error or return_tau_arr:
                u_star_outputs = u_star(tau, phi, is_antiderivative_wrt_tau, return_Fourier_error, return_tau_arr)
                return (
                    u_star_outputs[0] + rescale_factor * np.squeeze(NT_corrections),
                ) + u_star_outputs[1:]
            else:
                return u_star(tau, phi, is_antiderivative_wrt_tau) + rescale_factor * np.squeeze(NT_corrections)
        # --------------------------------------------------------------------------------------------------------------------------

        return mu_arr, flux_up, flux_down, u0, u_corrected
        
    else:
        return (mu_arr,) + _assemble_intensity_and_fluxes(
            scaled_omega_arr,
            tau_arr,
            scaled_tau_arr_with_0,
            mu_arr_pos, mu_arr,
            M_inv, W,
            N, NQuad, NLeg,
            NFourier, NLayers, NBDRF,
            is_atmos_multilayered,
            weighted_scaled_Leg_coeffs,
            BDRF_Fourier_modes,
            mu0, I0, rescale_factor, phi0,
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
        )