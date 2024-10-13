from PythonicDISORT.subroutines import _mathscr_v
from PythonicDISORT._solve_for_gen_and_part_sols import _solve_for_gen_and_part_sols
from PythonicDISORT._solve_for_coeffs import _solve_for_coeffs

import numpy as np
from math import pi

def _assemble_intensity_and_fluxes(
    scaled_omega_arr,                   # Delta-scaled single-scattering albedos
    tau_arr,                            # Lower boundary of layers
    scaled_tau_arr_with_0,              # Delta-scaled lower boundary of layers with 0 inserted in front
    mu_arr_pos, mu_arr,                 # Quadrature nodes for 1) upper 2) both hemispheres
    M_inv, W,                           # 1) 1 / mu; 2) quadrature weights for each hemisphere
    N, NQuad, NLeg,                     # Number of 1) upper 2) both hemispheres quadrature nodes; 3) phase function Legendre coefficients 
    NFourier, NLayers, NBDRF,           # Number of 1) intensity Fourier modes; 2) layers; 3) BDRF Fourier modes
    is_atmos_multilayered,              # Is the atmosphere multilayered?
    weighted_scaled_Leg_coeffs,         # Weighted and delta-scaled Legendre coefficients
    BDRF_Fourier_modes,                 # BDRF Fourier modes
    mu0, I0, I0_orig, phi0,             # Properties of direct beam. If I0 was rescaled to 1, I0_orig is the original I0
    there_is_beam_source,               # Is there a beam source?
    b_pos, b_neg,                       # Dirichlet BCs
    b_pos_is_scalar, b_neg_is_scalar,   # Is each Dirichlet BCs scalar (isotropic)?
    Nscoeffs,                           # Number of isotropic source polynomial coefficients
    s_poly_coeffs,                      # Polynomial coefficients of isotropic source
    there_is_iso_source,                # Is there an isotropic source?
    scale_tau,                          # Delta-scale factor for tau
    only_flux,                          # Only compute fluxes?
    use_banded_solver_NLayers,          # Number of layers above or equal which to use `scipy.linalg.solve_banded`
    autograd_compatible,                # Should the output functions be compatible with autograd?
):  
    """Assembles the solution functions: intensity `u`, upward flux `flux_up`, downward flux `flux_down` 
    from the previously solved eigenpairs and coefficients from the boundary conditions. 
    Returns these solution functions.
    This function is wrapped by the `pydisort` function.
    It should be called through `pydisort` and never directly.
    It has many seemingly redundant arguments, which are described below,
    to maximize precomputation in `pydisort`
    These arguments are passed on to the `_solve_for_gen_and_part_sols` and `_solve_for_coeffs` functions 
    which this function wraps. See the Jupyter Notebook, especially section 3, for 
    documentation, explanation and derivation.
    
    Arguments of _assemble_intensity_and_fluxes
    |            Variable            |            Type / Shape            |
    | ------------------------------ | ---------------------------------- |
    | `scaled_omega_arr`             | `NLayers`                          |
    | `tau_arr`                      | `NLayers`                          |
    | `scaled_tau_arr_with_0`        | `NLayers + 1`                      |
    | `mu_arr_pos`                   | `NQuad/2`                          |
    | `mu_arr`                       | `NQuad`                            |
    | `M_inv`                        | `NQuad/2`                          |
    | `W`                            | `NQuad/2`                          |
    | `N`                            | scalar                             |
    | `NQuad`                        | scalar                             |
    | `NLeg`                         | scalar                             |
    | `NFourier`                     | scalar                             |
    | `NLayers`                      | scalar                             |
    | `NBDRF`                        | scalar                             |
    | `is_atmos_multilayered`        | boolean                            |
    | `weighted_scaled_Leg_coeffs`   | `NLayers x NLeg`                   |
    | `BDRF_Fourier_modes`           | `NBDRF`                            |
    | `mu0`                          | scalar                             |
    | `I0`                           | scalar                             |
    | `I0_orig`                      | scalar                             |
    | `phi0`                         | scalar                             |
    | `there_is_beam_source`         | scalar                             |
    | `b_pos`                        | `NQuad/2 x NFourier` or scalar     |
    | `b_neg`                        | `NQuad/2 x NFourier` or scalar     |
    | `b_pos_is_scalar`              | boolean                            |
    | `b_neg_is_scalar`              | boolean                            |        
    | `Nscoeffs`                     | scalar                             |    
    | `s_poly_coeffs`                | `NLayers x Nscoeffs` or `Nscoeffs` |
    | `there_is_iso_source`          | boolean                            |
    | `scale_tau`                    | `NLayers`                          |
    | `only_flux`                    | boolean                            |
    | `use_banded_solver_NLayers`    | scalar                             |
    | `autograd_compatible`          | boolean                            |
    
    Notable internal variables of _assemble_intensity_and_fluxes
    |     Variable      |             Type / Shape               |
    | ----------------- | -------------------------------------- |
    | `G_collect`       | `NFourier x NLayers x NQuad x NQuad`   |
    | `G_collect_0`     | `NLayers x NQuad x NQuad`              |
    | `G_inv_collect_0` | `NLayers x NQuad x NQuad` or `None`    |
    | `K_collect`       | `NFourier x NLayers x NQuad`           |
    | `K_collect_0`     | `NLayers x NQuad`                      |
    | `B_collect`       | `NFourier x NLayers x NQuad` or `None` |
    | `B_collect_0`     | `NLayers x NQuad` or `None`            |
    | `B_pos`           | `NLayers x NQuad/2` or `None`          |
    | `B_neg`           | `NLayers x NQuad/2` or `None`          |
    | `GC_collect`      | `NFourier x NLayers x NQuad x NQuad`   |
    | `GC_collect_0`    | `NLayers x NQuad x NQuad`              |
    | `GC_pos`          | `NLayers x NQuad/2 x NQuad`            |
    | `GC_neg`          | `NLayers x NQuad/2 x NQuad`            |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np
    
    ################################## Assemble uncorrected intensity and flux functions #######################################
    
    # Compute all the necessary quantities
    # --------------------------------------------------------------------------------------------------------------------------
    outputs = _solve_for_gen_and_part_sols(
        NFourier,
        scaled_omega_arr,
        mu_arr_pos, mu_arr,
        M_inv, W,
        N, NQuad, NLeg,
        NLayers,
        weighted_scaled_Leg_coeffs,
        mu0, I0,
        there_is_beam_source,
        Nscoeffs,
        there_is_iso_source,
    )
    if there_is_beam_source and there_is_iso_source:
        G_collect, K_collect, B_collect, G_inv_collect_0 = outputs
        B_collect_0 = B_collect[0, :, :]
    elif there_is_beam_source and not there_is_iso_source:
        G_collect, K_collect, B_collect = outputs
        B_collect_0 = B_collect[0, :, :]
        G_inv_collect_0 = None
    elif not there_is_beam_source and there_is_iso_source:
        G_collect, K_collect, G_inv_collect_0 = outputs
        B_collect = None
    else:
        G_collect, K_collect = outputs
        B_collect = None
        G_inv_collect_0 = None
        
            
    GC_collect = _solve_for_coeffs(
        NFourier,
        G_collect,
        K_collect,
        B_collect,
        G_inv_collect_0,
        tau_arr,
        scaled_tau_arr_with_0,
        mu_arr, mu_arr_pos, mu_arr_pos * W,
        N, NQuad,
        NLayers, NBDRF,
        is_atmos_multilayered,
        BDRF_Fourier_modes,
        mu0, I0,
        there_is_beam_source,
        b_pos, b_neg,
        b_pos_is_scalar, b_neg_is_scalar,
        Nscoeffs,
        s_poly_coeffs,
        there_is_iso_source,
        use_banded_solver_NLayers,
    )
    
    G_collect_0 = G_collect[0, :, :, :]
    K_collect_0 = K_collect[0, :, :]
    GC_collect_0 = GC_collect[0, :, :, :]
    # --------------------------------------------------------------------------------------------------------------------------   
        
    if not only_flux:
 
        # Construct the intensity function (refer to Section 3.7 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        def u(tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False, return_tau_arr=False):
            """
            Intensity function with arguments `(tau, phi)` of types `(array or float, array or float)`.
            Returns an ndarray with axes corresponding to variation with `mu, tau, phi` respectively.
            Pass `return_Fourier_error = True` (defaults to `False`) to return the 
            Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
            Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
            to switch to an antiderivative of the function with respect to `tau`.
            """
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            if np.all(tau < 0) or np.all(tau > tau_arr[-1]):
                raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")
            
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
            
            exponent = np.concatenate(
                [
                    K_collect[:, l, :N] * (scaled_tau - scaled_tau_arr_lm1)[None, :, None],
                    K_collect[:, l, N:] * (scaled_tau - scaled_tau_arr_l)[None, :, None],
                ],
                axis=-1,
            )
            
            if is_antiderivative_wrt_tau and there_is_beam_source:
                um = (
                    np.einsum(
                        "mtij, mtj -> mti",
                        GC_collect[:, l, :, :],
                        np.exp(exponent) / (scale_tau[None, :, None] * K_collect)[:, l, :],
                        optimize=True,
                    )
                    + (B_collect / (-scale_tau / mu0)[None, :, None])[:, l, :]
                    * np.exp(-scaled_tau / mu0)[None, :, None]
                )
            elif is_antiderivative_wrt_tau and not there_is_beam_source:
                um = np.einsum(
                    "mtij, mtj -> mti",
                    GC_collect[:, l, :, :],
                    np.exp(exponent) / (scale_tau[None, l, None] * K_collect)[:, l, :],
                    optimize=True,
                )
            elif not is_antiderivative_wrt_tau and there_is_beam_source:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                ) + B_collect[:, l, :] * np.exp(-scaled_tau / mu0)[None, :, None]
            else:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                )
            
            # Contribution from particular solution for isotropic internal sources
            if there_is_iso_source:
                l_uniq, l_inv = np.unique(l, return_inverse=True)
                _mathscr_v_contribution = _mathscr_v(
                    tau, l_inv, Nscoeffs,
                    s_poly_coeffs[l_uniq],
                    G_collect_0[l_uniq],
                    K_collect_0[l_uniq],
                    G_inv_collect_0[l_uniq],
                    mu_arr,
                    is_antiderivative_wrt_tau,
                    autograd_compatible
                )
                
                if NFourier == 1:
                    um = um + _mathscr_v_contribution.T
                elif autograd_compatible:
                    um = um + np.concatenate((_mathscr_v_contribution.T, np.zeros((NFourier - 1, len(tau), NQuad))))
                else:
                    um[0, :, :] += _mathscr_v_contribution.T
                
            u = np.einsum(
                "mti, mp -> itp",
                um,
                np.cos(np.arange(NFourier)[:, None] * (phi0 - phi)[None, :]),
                optimize=True,
            )

            if return_Fourier_error:
                exponent = np.concatenate(
                    [
                        K_collect[-1, l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                        K_collect[-1, l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
                    ],
                    axis=-1,
                )
                if is_antiderivative_wrt_tau and there_is_beam_source:
                    ulast = (
                        np.einsum(
                            "tij, tj -> it",
                            GC_collect[-1, l, :, :],
                            np.exp(exponent) / (scale_tau[:, None] * K_collect[-1, :, :])[l, :],
                            optimize=True,
                        )
                        + (B_collect.T[-1, :, :] / (-scale_tau / mu0)[None, :])[:, l]
                        * np.exp(-scaled_tau / mu0)[None, :]
                    )
                elif is_antiderivative_wrt_tau and not there_is_beam_source:
                    ulast = np.einsum(
                        "tij, tj -> it",
                        GC_collect[-1, l, :, :],
                        np.exp(exponent) / (scale_tau[:, None] * K_collect[-1, :, :])[l, :],
                        optimize=True,
                        )
                elif not is_antiderivative_wrt_tau and not there_is_beam_source:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    ) + B_collect.T[-1, :, l] * np.exp(-scaled_tau / mu0)[None, :]
                else:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    )
                        
                u_abs = np.abs(u)
                Fourier_error = np.max(
                    np.divide(
                        np.abs(
                            ulast[:, :, None] * np.cos((NFourier - 1) * (phi0 - phi))[None, None, :]
                        ),
                        u_abs,
                        out=np.zeros_like(u_abs),
                        where=u_abs > 1e-15,
                    )
                )
                
                if return_tau_arr:
                    return I0_orig * np.squeeze(u), Fourier_error, tau_arr
                else:
                    return I0_orig * np.squeeze(u), Fourier_error
            else:
                if return_tau_arr:
                    return I0_orig * np.squeeze(u), tau_arr
                else:
                    return I0_orig * np.squeeze(u)
        # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct u0
    # --------------------------------------------------------------------------------------------------------------------------
    def u0(tau, is_antiderivative_wrt_tau=False, return_tau_arr=False, _return_act_dscale_for_reclass=False):
        """        
        Zeroth Fourier mode of the intensity with argument `tau` (type: array or float).
        Returns an ndarray with axes corresponding to variation with `mu` and `tau` respectively.
        This function is useful for calculating actinic fluxes and other quantities of interest,
        but reclassification of delta-scaled flux and other corrections must be done manually
        (for actinic flux `generate_diff_act_flux_funcs` will automatically perform the reclassification).
        Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
        to switch to an antiderivative of the function with respect to `tau`.
        """
        tau = np.atleast_1d(tau)
        if np.all(tau < 0) or np.all(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range given for the atmosphere (check `tau_arr`).")
        
        # Atmospheric layer indices
        l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
        scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
        scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

        # Delta-M scaling
        if np.any(scale_tau != np.ones(NLayers)):
            tau_dist_from_top = tau_arr[l] - tau
            scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
            scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top
            # The following complements the function `subroutines.generate_diff_act_flux_funcs`
            # in performing the reclassification of delta-scaled actinic flux
            if _return_act_dscale_for_reclass: 
                if is_antiderivative_wrt_tau:
                    act_dscale_reclassification = (
                        I0 * np.exp(-scaled_tau / mu0) / (-scale_tau[l] / mu0)
                        - I0 * np.exp(-tau / mu0) * -mu0
                    )
                else:
                    act_dscale_reclassification = I0 * np.exp(-scaled_tau / mu0) - I0 * np.exp(-tau / mu0)
        else:
            scaled_tau = tau
            if _return_act_dscale_for_reclass:
                act_dscale_reclassification = 0

        exponent = np.concatenate(
            [
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ],
            axis=-1,
        )
        
        if is_antiderivative_wrt_tau and there_is_beam_source:
            u0 = (
                np.einsum(
                    "tij, tj -> it",
                    GC_collect_0[l, :, :],
                    np.exp(exponent) / (scale_tau[:, None] * K_collect_0)[l, :],
                    optimize=True,
                )
                + (B_collect_0.T / (-scale_tau / mu0)[None, :])[:, l]
                * np.exp(-scaled_tau / mu0)[None, :]
            )
        if is_antiderivative_wrt_tau and not there_is_beam_source:
            u0 = np.einsum(
                "tij, tj -> it",
                GC_collect_0[l, :, :],
                np.exp(exponent) / (scale_tau[:, None] * K_collect_0)[l, :],
                optimize=True,
            )
        if not is_antiderivative_wrt_tau and there_is_beam_source:
            u0 = np.einsum(
                "tij, tj -> it", GC_collect_0[l, :, :], np.exp(exponent), optimize=True
            ) + B_collect_0.T[:, l] * np.exp(-scaled_tau / mu0)[None, :]
        else:
            u0 = np.einsum(
                "tij, tj -> it", GC_collect_0[l, :, :], np.exp(exponent), optimize=True
            )
        
        # Contribution from particular solution for isotropic internal sources
        if there_is_iso_source:
            l_uniq, l_inv = np.unique(l, return_inverse=True)
            _mathscr_v_contribution = _mathscr_v(
                tau, l_inv, Nscoeffs,
                s_poly_coeffs[l_uniq],
                G_collect_0[l_uniq],
                K_collect_0[l_uniq],
                G_inv_collect_0[l_uniq],
                mu_arr,
                is_antiderivative_wrt_tau,
                autograd_compatible
            )
            u0 = u0 + _mathscr_v_contribution
            
        if return_tau_arr and _return_act_dscale_for_reclass:
            return I0_orig * np.squeeze(u0), tau_arr, act_dscale_reclassification
        elif return_tau_arr and not _return_act_dscale_for_reclass:
            return I0_orig * np.squeeze(u0), tau_arr
        elif not return_tau_arr and _return_act_dscale_for_reclass:
            return I0_orig * np.squeeze(u0), act_dscale_reclassification
        else:
            return I0_orig * np.squeeze(u0)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct the flux functions (refer to Section 3.8 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    GC_pos = GC_collect_0[:, :N, :]
    GC_neg = GC_collect_0[:, N:, :]
    if there_is_beam_source:
        B_pos = B_collect_0[:, :N].T
        B_neg = B_collect_0[:, N:].T


    def flux_up(tau, is_antiderivative_wrt_tau=False, return_tau_arr=False):
        """    
        (Energetic) Flux function with argument `tau` (type: array or float) for positive (upward) `mu` values.
        Returns the diffuse flux magnitudes (same type and size as `tau`).
        Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
        to switch to an antiderivative of the function with respect to `tau`.
        """
        tau = np.atleast_1d(tau)
        if np.all(tau < 0) or np.all(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range given for the atmosphere (check `tau_arr`).")
        
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

        if there_is_iso_source:
            l_uniq, l_inv = np.unique(l, return_inverse=True)
            _mathscr_v_contribution = _mathscr_v(
                tau, l_inv, Nscoeffs,
                s_poly_coeffs[l_uniq],
                G_collect_0[l_uniq, :N, :],
                K_collect_0[l_uniq],
                G_inv_collect_0[l_uniq],
                mu_arr,
                is_antiderivative_wrt_tau,
                autograd_compatible
            )
        else:
            _mathscr_v_contribution = 0
        
        if there_is_beam_source and is_antiderivative_wrt_tau:
            direct_beam_contribution = (
                (B_pos / (-scale_tau / mu0)[None, :])[:, l] * np.exp(-scaled_tau / mu0)[None, :] 
            )
        elif there_is_beam_source and not is_antiderivative_wrt_tau:
            direct_beam_contribution = B_pos[:, l] * np.exp(-scaled_tau / mu0)[None, :]
        else:
            direct_beam_contribution = 0

        exponent = np.concatenate(
            [
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ],
            axis=-1,
        )
        
        if is_antiderivative_wrt_tau:
            u0_pos = (
                np.einsum(
                    "tij, tj -> it",
                    GC_pos[l, :, :],
                    np.exp(exponent) / (scale_tau[:, None] * K_collect_0)[l, :],
                )
                + direct_beam_contribution
                + _mathscr_v_contribution
            )
        else:
            u0_pos = (
                np.einsum("tij, tj -> it", GC_pos[l, :, :], np.exp(exponent))
                + direct_beam_contribution
                + _mathscr_v_contribution
            )
        flux = 2 * pi * (mu_arr_pos * W) @ u0_pos
        
        if return_tau_arr:
            return I0_orig * np.squeeze(flux)[()], tau_arr
        else:
            return I0_orig * np.squeeze(flux)[()]


    def flux_down(tau, is_antiderivative_wrt_tau=False, return_tau_arr=False):
        """
        (Energetic) Flux function with argument `tau` (type: array or float) for negative (downward) `mu` values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively where each entry is of the
        same type and size as `tau`.
        Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
        to switch to an antiderivative of the function with respect to `tau`.
        """
        tau = np.atleast_1d(tau)
        if np.all(tau < 0) or np.all(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range given for the atmosphere (check `tau_arr`).")

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

        if there_is_iso_source:
            l_uniq, l_inv = np.unique(l, return_inverse=True)
            _mathscr_v_contribution = _mathscr_v(
                tau, l_inv, Nscoeffs,
                s_poly_coeffs[l_uniq],
                G_collect_0[l_uniq, N:, :],
                K_collect_0[l_uniq],
                G_inv_collect_0[l_uniq],
                mu_arr,
                is_antiderivative_wrt_tau,
                autograd_compatible
            )
        else:
            _mathscr_v_contribution = 0
        
        if there_is_beam_source and is_antiderivative_wrt_tau:
            direct_beam_contribution = (B_neg / (-scale_tau / mu0)[None, l])[:, l] * np.exp(-scaled_tau / mu0)[None, :]
            direct_beam = I0 * mu0 * np.exp(-tau / mu0) * -mu0
            direct_beam_scaled = I0 * mu0 * np.exp(-scaled_tau / mu0) / (-scale_tau / mu0)[None, l]
        elif there_is_beam_source and not is_antiderivative_wrt_tau:
            direct_beam_contribution = B_neg[:, l] * np.exp(-scaled_tau / mu0)[None, :]
            direct_beam = I0 * mu0 * np.exp(-tau / mu0)
            direct_beam_scaled = I0 * mu0 * np.exp(-scaled_tau / mu0)
        else:
            direct_beam_contribution = 0
            direct_beam = 0
            direct_beam_scaled = 0
        
        exponent = np.concatenate(
            [
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ],
            axis=-1,
        )
        
        if is_antiderivative_wrt_tau:
            u0_neg = (
                np.einsum("tij, tj -> it", GC_neg[l, :, :], np.exp(exponent) / (scale_tau[:, None] * K_collect_0)[l, :])
                + direct_beam_contribution
                + _mathscr_v_contribution
            )
        else:
            u0_neg = (
                np.einsum("tij, tj -> it", GC_neg[l, :, :], np.exp(exponent))
                + direct_beam_contribution
                + _mathscr_v_contribution
            )
        diffuse_flux = 2 * pi * (mu_arr_pos * W) @ u0_neg + direct_beam_scaled - direct_beam
        
        if return_tau_arr:
            return (
                I0_orig * np.squeeze(diffuse_flux)[()],
                I0_orig * I0 * np.squeeze(direct_beam)[()],
                tau_arr
            )
        else:
            return (
                I0_orig * np.squeeze(diffuse_flux)[()],
                I0_orig * I0 * np.squeeze(direct_beam)[()],
            )
        # --------------------------------------------------------------------------------------------------------------------------

    if only_flux:
        return flux_up, flux_down, u0, G_collect_0, K_collect_0 # TODO: Remove the temporary G_collect_0, K_collect_0
    else:
        return flux_up, flux_down, u0, u