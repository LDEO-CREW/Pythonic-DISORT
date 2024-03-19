from PythonicDISORT.subroutines import _mathscr_v
from PythonicDISORT._diagonalize import _diagonalize
from PythonicDISORT._solve_for_coefs import _solve_for_coefs

import numpy as np
from math import pi

def _assemble_solution_functions(
    scaled_omega_arr,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr,
    M_inv, W,
    N, NQuad, NLeg, NFourier,
    NLayers, NBDRF,
    atmos_is_multilayered,
    weighted_scaled_Leg_coeffs,
    BDRF_Fourier_modes,
    mu0, I0, I0_orig, phi0,
    there_is_beam_source,
    b_pos, b_neg,
    b_pos_is_scalar, b_neg_is_scalar,
    s_poly_coeffs,
    Nscoeffs,
    there_is_iso_source,
    scale_tau,
    only_flux,
    use_sparse_NLayers,
    _is_compatible_with_autograd,
):  
    """This function is wrapped by the `pydisort` function.
    It should be called through `pydisort` and never directly.
    It has many seemingly redundant arguments to maximize precomputation in `pydisort`.
    These arguments are passed on to the `_diagonalize` and `_solve_for_coefs` functions 
    which this function wraps. See the Jupyter Notebook, especially section 3, for 
    documentation, explanation and derivation.
    
    """
    if _is_compatible_with_autograd:
        import autograd.numpy as np
    else:
        import numpy as np
    
    ###################################### Assemble uncorrected solution functions #############################################
    
    # Compute all the necessary quantities
    # --------------------------------------------------------------------------------------------------------------------------
    outputs = _diagonalize(
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
    if there_is_beam_source:
        if there_is_iso_source:
            G_collect, K_collect, B_collect, G_inv_collect_0 = outputs
        else:
            G_collect, K_collect, B_collect = outputs
            G_inv_collect_0 = None
        B_collect_0 = B_collect[0, :, :]
    else:
        if there_is_iso_source:
            G_collect, K_collect, G_inv_collect_0 = outputs
        else:
            G_collect, K_collect = outputs
            G_inv_collect_0 = None
        B_collect = None
            
    GC_collect = _solve_for_coefs(
        NFourier,
        G_collect,
        K_collect,
        B_collect,
        G_inv_collect_0,
        tau_arr,
        scaled_tau_arr_with_0,
        mu_arr_pos, mu_arr_pos * W, mu_arr,
        N, NQuad,
        NLayers, NBDRF,
        atmos_is_multilayered,
        BDRF_Fourier_modes,
        mu0, I0,
        there_is_beam_source,
        b_pos, b_neg,
        b_pos_is_scalar, b_neg_is_scalar,
        s_poly_coeffs,
        Nscoeffs,
        there_is_iso_source,
        use_sparse_NLayers,
    )
    
    G_collect_0 = G_collect[0, :, :, :]
    K_collect_0 = K_collect[0, :, :]
    GC_collect_0 = GC_collect[0, :, :, :]
    # --------------------------------------------------------------------------------------------------------------------------   
        
    if not only_flux:
 
        # Construct the intensity function
        # --------------------------------------------------------------------------------------------------------------------------
        def u(tau, phi, return_Fourier_error=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            assert np.all(tau >= 0)
            assert np.all(tau <= tau_arr[-1])
            
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
            if there_is_beam_source:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                ) + B_collect[:, l, :] * np.exp(-scaled_tau[None, :, None] / mu0)
            else:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                )
            
            # Contribution from particular solution for isotropic internal sources
            if there_is_iso_source:
                _mathscr_v_contribution = _mathscr_v(
                    tau, l,
                    s_poly_coeffs,
                    Nscoeffs,
                    G_collect_0,
                    K_collect_0,
                    G_inv_collect_0,
                    mu_arr,
                    _is_compatible_with_autograd,
                )
                
                if _is_compatible_with_autograd:
                    um = um + np.concatenate((_mathscr_v_contribution.T, np.zeros((NFourier - 1, len(tau), NQuad))))
                else:
                    um[0, :, :] += _mathscr_v_contribution.T
                
            intensities = np.einsum(
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
                if there_is_beam_source:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    ) + B_collect[-1, l, :].T * np.exp(-scaled_tau[None, :] / mu0)
                else:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    )
                intensities_abs = np.abs(intensities)
                Fourier_error = np.max(
                    np.abs(ulast[:, :, None] * np.cos((NFourier - 1) * (phi0 - phi))[None, None, :]),
                    intensities_abs,
                    out=np.zeros_like(intensities_abs),
                    where=intensities_abs > 1e-8,
                )
                return I0_orig * np.squeeze(intensities), Fourier_error
            else:
                return I0_orig * np.squeeze(intensities)
        # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct u0
    # --------------------------------------------------------------------------------------------------------------------------
    def u0(tau, return_act_dscale_reclassification=False):
        tau = np.atleast_1d(tau)
        assert np.all(tau >= 0)
        assert np.all(tau <= tau_arr[-1])
        
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
            if return_act_dscale_reclassification: 
                act_dscale_reclassification = I0 * np.exp(-scaled_tau / mu0) - I0 * np.exp(-tau / mu0)
        else:
            scaled_tau = tau
            if return_act_dscale_reclassification:
                act_dscale_reclassification = 0

        exponent = np.concatenate(
            [
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ],
            axis=-1,
        )
        if there_is_beam_source:
            u0 = np.einsum(
                "tij, tj -> it", GC_collect_0[l, :, :], np.exp(exponent), optimize=True
            ) + B_collect_0[l, :].T * np.exp(-scaled_tau[None, :] / mu0)
        else:
            u0 = np.einsum(
                "tij, tj -> it", GC_collect_0[l, :, :], np.exp(exponent), optimize=True
            )
        
        # Contribution from particular solution for isotropic internal sources
        if there_is_iso_source:
            _mathscr_v_contribution = _mathscr_v(
                tau, l,
                s_poly_coeffs,
                Nscoeffs,
                G_collect_0,
                K_collect_0,
                G_inv_collect_0,
                mu_arr,
                _is_compatible_with_autograd,
            )
            u0 += _mathscr_v_contribution
        
        if return_act_dscale_reclassification:
            return I0_orig * np.squeeze(u0), act_dscale_reclassification
        else:
            return I0_orig * np.squeeze(u0)
    # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct the flux functions
    # --------------------------------------------------------------------------------------------------------------------------
    GC_pos = GC_collect_0[:, :N, :]
    GC_neg = GC_collect_0[:, N:, :]
    if there_is_beam_source:
        B_pos = B_collect_0[:, :N].T
        B_neg = B_collect_0[:, N:].T


    def flux_up(tau):
        tau = np.atleast_1d(tau)
        assert np.all(tau >= 0)
        assert np.all(tau <= tau_arr[-1])
        
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
            _mathscr_v_contribution = _mathscr_v(
                tau, l,
                s_poly_coeffs,
                Nscoeffs,
                G_collect_0,
                K_collect_0,
                G_inv_collect_0,
                mu_arr,
                _is_compatible_with_autograd,
            )[:N, :]
        else:
            _mathscr_v_contribution = 0

        if there_is_beam_source:
            direct_beam_contribution = B_pos[:, l] * np.exp(-scaled_tau[None, :] / mu0)
        else:
            direct_beam_contribution = 0

        exponent = np.concatenate(
            [
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ],
            axis=-1,
        )
        u0_pos = (
            np.einsum("tij, tj -> it", GC_pos[l, :, :], np.exp(exponent))
            + direct_beam_contribution
            + _mathscr_v_contribution
        )
        
        flux = 2 * pi * (mu_arr_pos * W) @ u0_pos
        return I0_orig * np.squeeze(flux)[()]


    def flux_down(tau):
        tau = np.atleast_1d(tau)
        assert np.all(tau >= 0)
        assert np.all(tau <= tau_arr[-1])

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
            _mathscr_v_contribution = _mathscr_v(
                tau, l,
                s_poly_coeffs,
                Nscoeffs,
                G_collect_0,
                K_collect_0,
                G_inv_collect_0,
                mu_arr,
                _is_compatible_with_autograd,
            )[N:, :]
        else:
            _mathscr_v_contribution = 0
        
        if there_is_beam_source:
            direct_beam_contribution = B_neg[:, l] * np.exp(-scaled_tau[None, :] / mu0)
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
        u0_neg = (
            np.einsum("tij, tj -> it", GC_neg[l, :, :], np.exp(exponent))
            + direct_beam_contribution
            + _mathscr_v_contribution
        )
        
        diffuse_flux = 2 * pi * (mu_arr_pos * W) @ u0_neg + direct_beam_scaled - direct_beam
        return (
            I0_orig * np.squeeze(diffuse_flux)[()],
            I0_orig * I0 * np.squeeze(direct_beam)[()],
        )
        # --------------------------------------------------------------------------------------------------------------------------

    if only_flux:
        return flux_up, flux_down, u0
    else:
        return flux_up, flux_down, u0, u