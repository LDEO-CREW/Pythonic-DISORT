from PyDISORT import _one_Fourier_mode
from PyDISORT import subroutines
from math import pi
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np

def _loop_and_assemble_results(
    scaled_omega_arr,
    tau_arr,
    scaled_tau_arr_with_0,
    mu_arr_pos, mu_arr, weights_mu,
    M_inv, W,
    N, NQuad, NLeg, NLoops,
    NLayers, NBDRF,
    weighted_Leg_coeffs,
    weighted_Leg_coeffs_BDRF,
    mu0, I0, phi0,
    b_pos, b_neg,
    mathscr_vs,
    mathscr_vs_callable,
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
        scaled_tau_arr_with_0,
        mu_arr_pos, mu_arr,
        M_inv, W,
        N, NQuad, NLeg,
        NLayers, NBDRF,
        weighted_Leg_coeffs,
        weighted_Leg_coeffs_BDRF,
        mu0, I0,
        b_pos, b_neg,
        mathscr_vs,
        mathscr_vs_callable,
        scale_tau,
        use_sparse_NLayers
    )
    if mathscr_vs_callable:
        G_collect_0, C_0, K_collect_0, B_collect_0, G_inv_collect_0 = outputs
    else:
        GC_collect_0, K_collect_0, B_collect_0 = outputs
    # -------------------------------------------------------------------------------------------------------------------------- 
    
    if not only_flux:
        # To solve for the intensity, we need to loop NLoops times over the Fourier modes m
        # --------------------------------------------------------------------------------------------------------------------------
        GC_collect = np.empty((NLoops, NLayers, NQuad, NQuad))
        K_collect = np.empty((NLoops, NLayers, NQuad))
        B_collect = np.empty((NLoops, NLayers, NQuad))
        
        if mathscr_vs_callable:
            GC_collect[0, :, :, :] = G_collect_0 * C_0[:, None, :]
        else:
            GC_collect[0, :, :, :] = GC_collect_0
        K_collect[0, :, :] = K_collect_0
        B_collect[0, :, :] = B_collect_0
        # These loops are relatively easy to PARALLELIZE, but we have not implemented the parallelization (TODO)
        for m in range(1, NLoops):
            (
                GC_collect[m, :, :, :],
                K_collect[m, :, :],
                B_collect[m, :, :],
            ) = _one_Fourier_mode(
                m,
                scaled_omega_arr, 
                scaled_tau_arr_with_0,
                mu_arr_pos, mu_arr,
                M_inv, W,
                N, NQuad, NLeg,
                NLayers, NBDRF,
                weighted_Leg_coeffs,
                weighted_Leg_coeffs_BDRF,
                mu0, I0,
                b_pos, b_neg,
                mathscr_vs,
                mathscr_vs_callable,
                scale_tau,
                use_sparse_NLayers
            )
        # --------------------------------------------------------------------------------------------------------------------------
        
        # Construct the intensity function
        # --------------------------------------------------------------------------------------------------------------------------
        def u(tau, phi, return_Fourier_error=False):
            tau, phi = np.atleast_1d(tau, phi)
            # tau must be sorted in ascending order
            assert np.all(np.diff(tau) >= 0)
            # tau must be in the domain
            assert tau[-1] <= tau_arr[-1]
            assert tau[0] >= 0

            # Atmospheric layer indices
            l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
            l_unique = np.unique(l)
            scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
            scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

            # Delta-M scaling
            tau_dist_from_top = tau_arr[l] - tau
            scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
            scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top

            exponent = np.concatenate(
                (
                    K_collect[:, l, :N] * (scaled_tau - scaled_tau_arr_lm1)[None, :, None],
                    K_collect[:, l, N:] * (scaled_tau - scaled_tau_arr_l)[None, :, None],
                ),
                axis=-1,
            )
            um = np.einsum(
                "mtij, mtj -> itm", GC_collect[:, l, :, :], np.exp(exponent), optimize=True
            ) + B_collect[:, :, None] * np.exp(-tau[None, None, :] / mu0)
            
            # Contribution from particular solution for other internal sources
            if mathscr_vs_callable:
                mathscr_vs_contribution = np.concatenate(
                    [
                        (   # The current layer
                            mathscr_vs(
                                scaled_tau[l == l_uni],
                                scaled_tau_arr_with_0[l_uni + 1],
                                NQuad,
                                G_collect_0[l_uni, :, :],
                                K_collect_0[l_uni, :],
                                G_inv_collect_0[l_uni, :, :],
                            )[N:]
                            / scale_tau[l_uni]
                            # All the layers below
                            + np.sum(
                                [
                                    mat[N:] / scale_tau[l]
                                    for l, mat in enumerate(
                                        map(
                                            mathscr_vs,
                                            scaled_tau_arr_with_0[(l_uni + 1) : -1],
                                            scaled_tau_arr_with_0[(l_uni + 2) :],
                                            np.full((NLayers - l_uni - 1), NQuad),
                                            iter(G_collect_0[l_uni:-1, :, :]),
                                            iter(K_collect_0[l_uni:-1, :]),
                                            iter(G_inv_collect_0[l_uni:-1, :, :]),
                                        ),
                                        start=l_uni + 1,
                                    )
                                ],
                                axis=0,
                            )
                        )
                        for l_uni in l_unique
                    ],
                    axis=1,
                )
                um[:, :, 0] = um[:, :, 0] + mathscr_vs_contribution

            intensities = np.squeeze(
                np.einsum(
                    "itm, mp -> itp",
                    um,
                    np.cos(np.arange(NLoops)[:, None] * (phi0 - phi)[None, :]),
                    optimize=True,
                )
            )

            if return_Fourier_error:
                exponent = np.concatenate(
                    (
                        K_collect[-1, l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                        K_collect[-1, l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
                    ),
                    axis=-1,
                )
                ulast = np.einsum(
                    "tij, tj -> it", GC_collect[-1, l, :, :], np.exp(exponent), optimize=True
                ) + B_collect[-1, :, None] * np.exp(-scaled_tau[None, :] / mu0)
                Fourier_error = np.max(
                    np.abs(
                        ulast[:, :, None]
                        * np.cos((NLoops - 1) * (phi0 - phi))[None, None, :]
                        / intensities
                    )
                )
                return intensities, Fourier_error
            else:
                return intensities
        # --------------------------------------------------------------------------------------------------------------------------
    
    # Construct the flux functions
    # --------------------------------------------------------------------------------------------------------------------------
    GC_pos = GC_collect_0[:, :N, :]
    B_pos = B_collect_0[:, :N].T
    GC_neg = GC_collect_0[:, N:, :]
    B_neg = B_collect_0[:, N:].T


    def flux_up(tau):
        tau = np.atleast_1d(tau)
        # tau must be sorted in ascending order
        assert np.all(np.diff(tau) >= 0)
        # tau must be in the domain
        assert tau[-1] <= tau_arr[-1]
        assert tau[0] >= 0

        # Atmospheric layer indices
        l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
        scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
        scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

        # Delta-M scaling
        tau_dist_from_top = tau_arr[l] - tau
        scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
        scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top

        exponent = np.concatenate(
            (
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ),
            axis=-1,
        )
        u0_pos = np.einsum("tij, tj -> it", GC_pos, np.exp(exponent)) + B_pos[
            :, l
        ] * np.exp(-tau[None, :] / mu0)
        return np.squeeze(2 * pi * (mu_arr_pos * weights_mu) @ u0_pos)[()]


    def flux_down(tau):
        tau = np.atleast_1d(tau)
        # tau must be sorted in ascending order
        assert np.all(np.diff(tau) >= 0)
        # tau must be in the domain
        assert tau[-1] <= tau_arr[-1]
        assert tau[0] >= 0

        # Atmospheric layer indices
        l = np.argmax(tau[:, None] <= tau_arr[None, :], axis=1)
        l_unique = np.unique(l)
        scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
        scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]

        # Delta-M scaling
        tau_dist_from_top = tau_arr[l] - tau
        scaled_tau_dist_from_top = tau_dist_from_top * scale_tau[l]
        scaled_tau = scaled_tau_arr_l - scaled_tau_dist_from_top

        direct_beam = I0 * mu0 * np.exp(-tau / mu0)
        direct_beam_scaled = I0 * mu0 * np.exp(-scaled_tau / mu0)
        exponent = np.concatenate(
            (
                K_collect_0[l, :N] * (scaled_tau - scaled_tau_arr_lm1)[:, None],
                K_collect_0[l, N:] * (scaled_tau - scaled_tau_arr_l)[:, None],
            ),
            axis=-1,
        )

        if mathscr_vs_callable:
            mathscr_vs_contribution = np.concatenate(
                [
                    (   # The current layer
                        mathscr_vs(
                            scaled_tau[l == l_uni],
                            scaled_tau_arr_with_0[l_uni + 1],
                            NQuad,
                            G_collect_0[l_uni, :, :],
                            K_collect_0[l_uni, :],
                            G_inv_collect_0[l_uni, :, :],
                        )[N:]
                        / scale_tau[l_uni]
                        # All the layers below
                        + np.sum(
                            [
                                mat[N:] / scale_tau[l]
                                for l, mat in enumerate(
                                    map(
                                        mathscr_vs,
                                        scaled_tau_arr_with_0[(l_uni + 1) : -1],
                                        scaled_tau_arr_with_0[(l_uni + 2) :],
                                        np.full((NLayers - l_uni - 1), NQuad),
                                        iter(G_collect_0[l_uni:-1, :, :]),
                                        iter(K_collect_0[l_uni:-1, :]),
                                        iter(G_inv_collect_0[l_uni:-1, :, :]),
                                    ),
                                    start=l_uni + 1,
                                )
                            ],
                            axis=0,
                        )
                    )
                    for l_uni in l_unique
                ],
                axis=1,
            )
            u0_neg = (
                np.einsum("tij, tj -> it", GC_neg, np.exp(exponent))
                + B_neg[:, l] * np.exp(-scaled_tau[None, :] / mu0)
                + mathscr_vs_contribution
            )
        else:
            u0_neg = (
                np.einsum("tij, tj -> it", GC_neg, np.exp(exponent))
                + B_neg[:, l] * np.exp(-scaled_tau[None, :] / mu0)
            )
        return (
            np.squeeze(
                2 * pi * (mu_arr_pos * weights_mu) @ u0_neg
                + direct_beam_scaled
                - direct_beam
            )[()],
            direct_beam,
        )
        # --------------------------------------------------------------------------------------------------------------------------

    if only_flux:
        return flux_up, flux_down
    else:
        return flux_up, flux_down, u