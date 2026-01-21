from PythonicDISORT import subroutines
from PythonicDISORT.subroutines import prepend
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
        scaled_tau_arr_with_0 = prepend(np.cumsum(scaled_thickness_arr), NLayers, 0)
        scaled_Leg_coeffs = (Leg_coeffs - f_arr[:, None]) / (1 - f_arr)[:, None]
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = (1 - f_arr) / scale_tau * omega_arr

        translations = scaled_tau_arr_with_0[:-1] - scale_tau * prepend(tau_arr[:-1], NLayers - 1, 0)
        scaled_s_poly_coeffs = (
            subroutines.affine_transform_poly_coeffs(s_poly_coeffs, scale_tau, translations)
            / scale_tau[:, None]  # Divide by d\tau^* / d\tau
        ) * (1 - omega_arr)[:, None]  # Enforce Kirchoff's Law of Thermal Radiation: absorptivity equals emissivity

    else:
        # This is a shortcut to the same results
        scale_tau = np.ones(NLayers)
        scaled_tau_arr_with_0 = prepend(tau_arr, NLayers, 0)
        scaled_Leg_coeffs = Leg_coeffs
        weighted_scaled_Leg_coeffs = scaled_Leg_coeffs * (2 * np.arange(NLeg) + 1)[None, :]
        scaled_omega_arr = omega_arr
        scaled_s_poly_coeffs = s_poly_coeffs * (1 - omega_arr)[:, None]  # Enforce Kirchoff's Law of Thermal Radiation: absorptivity equals emissivity
        
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
    
    if NT_cor and not only_flux and there_is_beam_source and np.any(f_arr > 0) and NLeg < NLeg_all and np.any(omega_arr > 0):
        
        ############################### Perform NT corrections on the intensity but not the flux ###############################
        ############################### Refer to section 3.7.2 of the Comprehensive Documentation ##############################
        
        # Delta-M scaled solution; no further corrections to the flux
        flux_up, flux_down, u0, u_star = _assemble_intensity_and_fluxes(
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
        )
        
        # TMS correction for the intensity (see section 3.7.2)
        # --------------------------------------------------------------------------------------------------------------------------
        def TMS_correction(tau, phi, is_antiderivative_wrt_tau, l, l_uniq, l_inv):
            Ntau = len(tau)
            Nphi = len(phi)
            # Atmospheric layer indices
            if not there_is_iso_source:
                l_uniq, l_inv = np.unique(l, return_inverse=True)
            scaled_tau_arr_l = scaled_tau_arr_with_0[1:][l]
            scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

            # Delta-M scaling
            tau_dist_from_top = tau_arr[l] - tau
            scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
            scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top
            
            # mathscr_B
            # --------------------------------------------------------------------------------------------------------------------------
            nu = subroutines.atleast_2d_append(
                subroutines.calculate_nu(mu_arr, phi, -mu0, phi0)
            )
            p_true = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_Leg_coeffs_all[l_uniq])
                ],
                axis=1,
            ) # (NQuad, NLayers_uniq, Nphi)
            p_trun = np.concatenate(
                [
                    f(nu)[:, None, :]
                    # Iterates over the 0th axis
                    for f in map(Legendre, weighted_scaled_Leg_coeffs[l_uniq])
                ],
                axis=1,
            ) # (NQuad, NLayers_uniq, Nphi)
            
            mathscr_B = (
                (scaled_omega_arr[l_uniq] * I0_div_4pi)[None, :, None]
                * (mu0 / (mu0 + mu_arr))[:, None, None]
                * (p_true / (1 - f_arr)[None, l_uniq, None] - p_trun)
            ) # (NQuad, NLayers_uniq, Nphi)
            # --------------------------------------------------------------------------------------------------------------------------
            
            neg_scaled_tau_div_mu0 = -scaled_tau / mu0
            if is_antiderivative_wrt_tau:

                scale_tau_l = scale_tau[l_uniq]
                neg_scale_tau_l_div_mu0 = -scale_tau_l / mu0
                scale_tau_l_div_mu_arr_pos = scale_tau_l[None, :] / mu_arr_pos[:, None]

                TMS_correction_pos = (
                    np.exp(neg_scaled_tau_div_mu0) / neg_scale_tau_l_div_mu0[l_inv]
                )[None, :] - np.exp(
                    (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_l[None, :] / mu0
                ) / scale_tau_l_div_mu_arr_pos[:, l_inv]
                TMS_correction_neg = (
                    np.exp(neg_scaled_tau_div_mu0) / neg_scale_tau_l_div_mu0[l_inv]
                )[None, :] + np.exp(
                    (scaled_tau_arr_lm1 - scaled_tau)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_lm1[None, :] / mu0
                ) / scale_tau_l_div_mu_arr_pos[:, l_inv]
            else:
                TMS_correction_pos = np.exp(neg_scaled_tau_div_mu0)[None, :] - np.exp(
                    (scaled_tau - scaled_tau_arr_l)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_l[None, :] / mu0
                )
                TMS_correction_neg = np.exp(neg_scaled_tau_div_mu0)[None, :] - np.exp(
                    (scaled_tau_arr_lm1 - scaled_tau)[None, :] / mu_arr_pos[:, None]
                    - scaled_tau_arr_lm1[None, :] / mu0
                )

            
            ## Contribution from other layers
            # --------------------------------------------------------------------------------------------------------------------------
            solution = (
                mathscr_B[:, l_inv, :]
                * np.concatenate((TMS_correction_pos, TMS_correction_neg), axis=0)[:, :, None]
            )
            
            if is_atmos_multilayered:
                any_pos = (l.min() < NLayers - 1)
                any_neg = (l.max() > 0)

                mu0_inv = 1 / mu0

                scaled_tauwith0_front = scaled_tau_arr_with_0[:-1]          # (NLayers,)
                scaled_tauwith0_back  = scaled_tau_arr_with_0[1:]           # (NLayers,); i.e. `scaled_tau_arr`

                exp_taufront_mu0_inv = np.concatenate(
                    ([1], np.exp(-scaled_tauwith0_front[1:] * mu0_inv))
                )  # (Nlayers, )


                Bpos = mathscr_B[:N, :, :]                       # (N, NLayers_uniq, Nphi)
                Bneg = mathscr_B[N:, :, :]                       # (N, NLayers_uniq, Nphi)

                # decay[n,r] = exp(-scaled_thickness_arr[r]/mu[n]); always <= 1
                log_decay = -scaled_thickness_arr[None, :] * M_inv[:, None]                    # (N, NLayers)

                # decay_prod[n,k] = prod_{m=0}^{k-1} decay[n,m], with decay_prod[:,0]=1
                log_decay_prod = np.cumsum(log_decay, axis=1) # The cumsum requires ALL layers; `l_uniq` is inapplicable
                decay_prod = np.concatenate((np.ones((N, 1)), np.exp(log_decay_prod)), axis=1) # (N, NLayers + 1)

                if is_antiderivative_wrt_tau:
                    # (N, NLayers)
                    integration_factor = mu_arr_pos[:, None] / scale_tau

                # ====================================================================
                # POS: pre-accumulate "contribution from below" sums Rpos[N, l, Nphi]
                # ====================================================================
                if any_pos:
                    thickness_pos = scaled_thickness_arr[None, :] * (M_inv + mu0_inv)[:, None]  # (N, NLayers); always > 0
                    # em1_pos = 1 - exp(-scaled_thickness_arr*(1/mu + 1/mu0)) = -expm1(-thickness_pos)
                    em1_pos = -np.expm1(-thickness_pos)                                         # (N, NLayers)
                    if is_antiderivative_wrt_tau:
                        em1_pos = integration_factor * em1_pos

                    layer_term_pos = em1_pos * exp_taufront_mu0_inv[None, :]                    # (N, NLayers)

                    # Reverse-scan via cumsum:
                    # Rpos[l] = (1/decay_prod[l+1]) * sum_{r=l+1..} (layer_term_pos[r] * decay_prod[r])
                    weighted_layer_term_pos = layer_term_pos * decay_prod[:, :NLayers]          # (N, NLayers)
                    cumsum_pos = np.cumsum(weighted_layer_term_pos[:, ::-1], axis=1)[:, ::-1]   # (N, NLayers)

                    Rpos_rest = cumsum_pos[:, 1:] / decay_prod[:, 1 : NLayers]                  # (N, NLayers - 1)
                    Rpos = np.concatenate([Rpos_rest, np.zeros((N, 1))], axis=1)                # (N, NLayers)

                    # Apply to all `tau`
                    # expfac_pos = exp(M_inv*tau) * exp(-M_inv*scaled_tauwith0_back[l])
                    expfac_pos = np.exp(M_inv[:, None] * (scaled_tau - scaled_tauwith0_back[l])[None, :])  # (N, Ntau)
                    tmp = np.take(Rpos, l, axis=1)                                                         # (N, Ntau)
                    tmp = (tmp * expfac_pos)[:, :, None] * Bpos[:, l_inv, :]
                    
                    if autograd_compatible:
                        solution = np.concatenate((solution[:N, :, :] + tmp, solution[N:, :, :]), axis=0)
                    else:
                        solution[:N, :, :] += tmp

                # ====================================================================
                # NEG: pre-accumulate "contribution from above" sums Rneg[N, l, Nphi]
                # ====================================================================
                if any_neg:
                    decay = np.exp(log_decay)                                                    # (N, NLayers)
                    
                    # Ensure no positive exponents
                    thickness_neg = scaled_thickness_arr[None, :] * (M_inv - mu0_inv)[:, None]   # (N, NLayers)
                    mask = (thickness_neg >= 0)

                    exp_x1 = np.exp(-scaled_tauwith0_back * mu0_inv)[None, :]                    # (1, NLayers)
                    exp_x0 = decay * exp_taufront_mu0_inv[None, :]                               # (N, NLayers)

                    # Let a = |thickness_neg| >= 0. Then expm1(-a) uses only non-positive inputs.
                    em1_neg = np.expm1(-np.abs(thickness_neg))                                       # (N, NLayers)

                    # thickness_neg>=0: exp_x1 * (1 - exp(-a)) = exp_x1 * (-expm1(-a)) = exp_x1 * (-em1_neg)
                    # thickness_neg<0 : exp_x0 * (exp(-a) - 1) = exp_x0 * expm1(-a)    = exp_x0 * ( em1_neg)
                    layer_term_negthick = em1_neg * exp_x0                                       # (N, NLayers)
                    layer_term_posthick = -em1_neg * exp_x1                                      # (N, NLayers)
                    layer_term_neg = layer_term_posthick * mask + layer_term_negthick * (~mask)  # (N, NLayers)

                    if is_antiderivative_wrt_tau:
                        layer_term_neg = -integration_factor * layer_term_neg

                    # Forward-scan via cumsum:
                    # Rneg[l] = decay_prod[l] * sum_{r<l} layer_term_neg[r] / decay_prod[r+1]
                    weighted_layer_term_neg = layer_term_neg / decay_prod[:, 1:]                 # (N, NLayers)
                    cumsum_neg = np.cumsum(weighted_layer_term_neg, axis=1)                      # (N, NLayers)
                    
                    Rneg_rest = decay_prod[:, 1 : NLayers] * cumsum_neg[:, :-1]                  # (N, NLayers - 1)
                    Rneg = np.concatenate([np.zeros((N, 1)), Rneg_rest], axis=1)                 # (N, NLayers)

                    # Apply to all `tau`
                    expfac_neg = np.exp(M_inv[:, None] * (scaled_tauwith0_front[l] - scaled_tau)[None, :])  # (N, Ntau)
                    tmp = np.take(Rneg, l, axis=1)                                                          # (N, Ntau)
                    tmp = (tmp * expfac_neg)[:, :, None] * Bneg[:, l_inv, :]
                    
                    if autograd_compatible:
                        solution = np.concatenate((solution[:N, :, :], solution[N:, :, :] + tmp), axis=0)
                    else:
                        solution[N:, :, :] += tmp

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
                I0_div_4pi
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
        def u_corrected(
            tau,
            phi,
            is_antiderivative_wrt_tau=False,
            return_Fourier_error=False,
            return_tau_arr=False,
        ):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)

            u_star_outputs = u_star(
                tau,
                phi,
                is_antiderivative_wrt_tau,
                return_Fourier_error,
                return_tau_arr,
                _return_l=True,
            )
            if there_is_iso_source:
                NT_corrections = TMS_correction(
                    tau,
                    phi,
                    is_antiderivative_wrt_tau,
                    u_star_outputs[-3],
                    u_star_outputs[-2],
                    u_star_outputs[-1],
                )
                u_star_outputs = u_star_outputs[:-3]
            else:
                NT_corrections = TMS_correction(
                    tau, phi, is_antiderivative_wrt_tau, u_star_outputs[-1], None, None
                )
                u_star_outputs = u_star_outputs[:-1]

            if autograd_compatible:
                NT_corrections = np.concatenate(
                    (
                        NT_corrections[:N, :, :],
                        NT_corrections[N:, :, :]
                        + IMS_correction(tau, phi, is_antiderivative_wrt_tau),
                    ),
                    axis=0,
                )
            else:
                NT_corrections[N:, :, :] += IMS_correction(tau, phi, is_antiderivative_wrt_tau)

            if return_Fourier_error or return_tau_arr:
                return (
                    u_star_outputs[0] + rescale_factor * np.squeeze(NT_corrections),
                ) + u_star_outputs[1:]
            else:
                return u_star_outputs[0] + rescale_factor * np.squeeze(NT_corrections)

        # --------------------------------------------------------------------------------------------------------------------------

        return mu_arr, flux_up, flux_down, u0, u_corrected
        
    else:
        return (mu_arr,) + _assemble_intensity_and_fluxes(
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
        )