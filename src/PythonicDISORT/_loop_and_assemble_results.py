from PythonicDISORT.subroutines import _mathscr_v
from PythonicDISORT._one_Fourier_mode import _one_Fourier_mode 
from math import pi
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np

def _loop_and_assemble_results(
    scaled_omega_arr,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr,
    M_inv, W,
    N, NQuad, NLeg, NLoops,
    NLayers, NBDRF,
    weighted_scaled_Leg_coeffs,
    weighted_Leg_coeffs_BDRF,
    mu0, I0, phi0,
    b_pos, b_neg,
    scalar_b_pos, scalar_b_neg,
    s_poly_coeffs,
    Nscoeffs,
    scale_tau,
    only_flux,
    use_sparse_NLayers
):  
    """This function is wrapped by the `pydisort` function.
    It should be called through `pydisort` and never directly.
    It has many seemingly redundant arguments to maximize precomputation in `pydisort`.
    Most of the arguments are passed to the `_one_Fourier_mode` function which this function wraps and loops.
    These loops are relatively easy to parallelize especially since we packaged a loop as a function,
    but we have not implemented the parallelization (TODO).
    
    """
    # We only need to solve for the 0th Fourier mode to determine the flux
    # --------------------------------------------------------------------------------------------------------------------------
    outputs = _one_Fourier_mode(
        0,
        scaled_omega_arr,
        tau_arr,
        scaled_tau_arr_with_0,
        mu_arr_pos, mu_arr,
        M_inv, W,
        N, NQuad, NLeg,
        NLayers, NBDRF,
        weighted_scaled_Leg_coeffs,
        weighted_Leg_coeffs_BDRF,
        mu0, I0,
        b_pos, b_neg,
        scalar_b_pos, scalar_b_neg,
        s_poly_coeffs,
        Nscoeffs,
        use_sparse_NLayers
    )
    if I0 > 0:
        if Nscoeffs > 0:
            G_collect_0, C_0, K_collect_0, B_collect_0, G_inv_collect_0 = outputs
            GC_collect_0 = G_collect_0 * C_0[:, None, :]
        else:
            GC_collect_0, K_collect_0, B_collect_0 = outputs
    else:
        if Nscoeffs > 0:
            G_collect_0, C_0, K_collect_0, G_inv_collect_0 = outputs
            GC_collect_0 = G_collect_0 * C_0[:, None, :]
        else:
            GC_collect_0, K_collect_0 = outputs    
    # -------------------------------------------------------------------------------------------------------------------------- 
    
    if not only_flux:
        # To solve for the intensity, we need to loop NLoops times over the Fourier modes m
        # --------------------------------------------------------------------------------------------------------------------------
        GC_collect = np.empty((NLoops, NLayers, NQuad, NQuad))
        K_collect = np.empty((NLoops, NLayers, NQuad))
        GC_collect[0, :, :, :] = GC_collect_0
        K_collect[0, :, :] = K_collect_0
        if I0 > 0:
            B_collect = np.empty((NLoops, NLayers, NQuad))
            B_collect[0, :, :] = B_collect_0
        # TODO: Look into the "xarray.apply_ufunc" method or "dask delayed / bag" to parallelize this
        # May need to code an xarray wrapper. Dask works with Python code.
        for m in range(1, NLoops): 
            outputs = _one_Fourier_mode(
                m,
                scaled_omega_arr,
                tau_arr,
                scaled_tau_arr_with_0,
                mu_arr_pos, mu_arr,
                M_inv, W,
                N, NQuad, NLeg,
                NLayers, NBDRF,
                weighted_scaled_Leg_coeffs,
                weighted_Leg_coeffs_BDRF,
                mu0, I0,
                b_pos, b_neg,
                scalar_b_pos, scalar_b_neg,
                s_poly_coeffs,
                Nscoeffs,
                use_sparse_NLayers
            )
            if I0 > 0:
                GC_collect[m, :, :, :], K_collect[m, :, :], B_collect[m, :, :] = outputs
            else:
                GC_collect[m, :, :, :], K_collect[m, :, :] = outputs
                
        # --------------------------------------------------------------------------------------------------------------------------
        
        # Construct the intensity function
        # --------------------------------------------------------------------------------------------------------------------------
        def u(tau, phi, return_Fourier_error=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            
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
            if I0 > 0:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                ) + B_collect[:, l, :] * np.exp(-scaled_tau[None, :, None] / mu0)
            else:
                um = np.einsum(
                    "mtij, mtj -> mti", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
                )
            
            # Contribution from particular solution for isotropic internal sources
            if Nscoeffs > 0:
                _mathscr_v_contribution = _mathscr_v(
                    tau, l,
                    s_poly_coeffs,
                    Nscoeffs,
                    G_collect_0,
                    K_collect_0,
                    G_inv_collect_0,
                    mu_arr,
                )
                # The following line must be implemented differently for autograd to work on `u` with isotropic sources
                um[0, :, :] += _mathscr_v_contribution.T
                
            intensities = np.squeeze(
                np.einsum(
                    "mti, mp -> itp",
                    um,
                    np.cos(np.arange(NLoops)[:, None] * (phi0 - phi)[None, :]),
                    optimize=True,
                )
            )

            if return_Fourier_error:
                exponent = np.concatenate(
                    [
                        K_collect[-1, l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                        K_collect[-1, l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
                    ],
                    axis=-1,
                )
                if I0 > 0:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    ) + B_collect[-1, l, :].T * np.exp(-scaled_tau[None, :] / mu0)
                else:
                    ulast = np.einsum(
                        "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                    )
                Fourier_error = np.max(
                    np.abs(
                        (ulast[:, :, None] * np.cos((NLoops - 1) * (phi0 - phi))[None, None, :])
                        / np.clip(intensities, a_min=1e-15, a_max=None)
                    )
                )
                return intensities, Fourier_error
            else:
                return intensities
        # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct the flux functions
    # --------------------------------------------------------------------------------------------------------------------------
    GC_pos = GC_collect_0[:, :N, :]
    GC_neg = GC_collect_0[:, N:, :]
    if I0 > 0:
        B_pos = B_collect_0[:, :N].T
        B_neg = B_collect_0[:, N:].T


    def flux_up(tau):
        tau = np.atleast_1d(tau)
        
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

        if Nscoeffs > 0:
            _mathscr_v_contribution = _mathscr_v(
                tau, l,
                s_poly_coeffs,
                Nscoeffs,
                G_collect_0,
                K_collect_0,
                G_inv_collect_0,
                mu_arr,
            )[:N, :]
        else:
            _mathscr_v_contribution = 0

        if I0 > 0:
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
                  
        return np.squeeze(2 * pi * (mu_arr_pos * W) @ u0_pos)[()]


    def flux_down(tau):
        tau = np.atleast_1d(tau)

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

        if Nscoeffs > 0:
            _mathscr_v_contribution = _mathscr_v(
                tau, l,
                s_poly_coeffs,
                Nscoeffs,
                G_collect_0,
                K_collect_0,
                G_inv_collect_0,
                mu_arr,
            )[N:, :]
        else:
            _mathscr_v_contribution = 0
        
        if I0 > 0:
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
        
        return (
            np.squeeze(
                2 * pi * (mu_arr_pos * W) @ u0_neg
                + direct_beam_scaled
                - direct_beam
            )[()],
            np.squeeze(direct_beam)[()],
        )
        # --------------------------------------------------------------------------------------------------------------------------

    if only_flux:
        return flux_up, flux_down
    else:
        return flux_up, flux_down, u