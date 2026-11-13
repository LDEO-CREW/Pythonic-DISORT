from PythonicDISORT import subroutines

import numpy as np
import scipy as sc


def _NT_corrections(
    u_star,                             # Delta-M scaled, uncorrected intensity function
    NT_cor,                             # Perform NT corrections?
    interpolate_mu,                     # Does `u_star` take `mu` as its first argument?
    tau_arr,                            # Lower boundary of layers
    thickness_arr,                      # Thickness of each layer
    omega_arr,                          # Single-scattering albedos
    f_arr,                              # Fractional scattering into peak for each layer
    Leg_coeffs_all,                     # All available unweighted phase function Legendre coefficients
    weighted_Leg_coeffs_all,            # All available weighted phase function Legendre coefficients
    NLeg, NLeg_all,                     # Number of phase function Legendre coefficients 1) used by the solver; 2) available
    NLayers,                            # Number of layers
    is_atmos_multilayered,              # Is the atmosphere multilayered?
    mu_arr_pos, M_inv,                  # 1) Upper hemisphere quadrature nodes (cosine polar angles); 2) 1 / `mu_arr_pos`
    scale_tau,                          # Delta-scale factor for tau
    scaled_tau_arr_with_0,              # Delta-scaled lower boundary of layers with 0 inserted in front
    scaled_thickness_arr,               # Delta-scaled thickness of each layer
    scaled_omega_arr,                   # Delta-scaled single-scattering albedos
    mu0, phi0, I0_div_4pi,              # Direct beam parameters
    rescale_factor,                     # Rescale factor of the sources
    there_is_beam_source,               # Is there a beam source?
    autograd_compatible,                # Should the output functions be compatible with autograd?
):
    """Performs the Nakajima-Tanaka (NT) corrections on the intensity, but not the flux, if `NT_cor`
    and if the corrections are applicable. Returns the intensity function `u`, which is otherwise `u_star`.
    The corrections are evaluated at the quadrature nodes or, if `interpolate_mu`, at the `mu` values
    which are passed to `u`.
    This function is wrapped by the `pydisort` function.
    It should be called through `pydisort` and never directly.
    It has many seemingly redundant arguments to maximize precomputation in `pydisort`.
    See the Jupyter Notebook, especially section 3.7.2, for documentation, explanation and derivation.

    Arguments of _NT_corrections
    |            Variable            |       Type / Shape       |
    | ------------------------------ | ------------------------ |
    | `u_star`                       | function                 |
    | `NT_cor`                       | boolean                  |
    | `interpolate_mu`               | boolean                  |
    | `tau_arr`                      | `NLayers`                |
    | `thickness_arr`                | `NLayers`                |
    | `omega_arr`                    | `NLayers`                |
    | `f_arr`                        | `NLayers`                |
    | `Leg_coeffs_all`               | `NLayers x NLeg_all`     |
    | `weighted_Leg_coeffs_all`      | `NLayers x NLeg_all`     |
    | `NLeg`                         | scalar                   |
    | `NLeg_all`                     | scalar                   |
    | `NLayers`                      | scalar                   |
    | `is_atmos_multilayered`        | boolean                  |
    | `mu_arr_pos`                   | `NQuad/2`                |
    | `M_inv`                        | `NQuad/2`                |
    | `scale_tau`                    | `NLayers`                |
    | `scaled_tau_arr_with_0`        | `NLayers + 1`            |
    | `scaled_thickness_arr`         | `NLayers`                |
    | `scaled_omega_arr`             | `NLayers`                |
    | `mu0`                          | scalar                   |
    | `phi0`                         | scalar                   |
    | `I0_div_4pi`                   | scalar                   |
    | `rescale_factor`               | scalar                   |
    | `there_is_beam_source`         | boolean                  |
    | `autograd_compatible`          | boolean                  |

    Notable internal variables of _NT_corrections
    |          Variable            |        Type / Shape        |
    | ---------------------------- | -------------------------- |
    | `TMS_Leg_coeffs`             | `NLayers x NLeg_all`       |
    | `cumsum1`                    | `NLayers`                  |
    | `omega_avg`                  | `NLayers`                  |
    | `cumsum2`                    | `NLayers`                  |
    | `f_avg`                      | `NLayers`                  |
    | `omega_f_avg`                | `NLayers + 1`              |
    | `Leg_coeffs_residue`         | `NLayers x NLeg_all`       |
    | `Leg_coeffs_residue_avg`     | `(NLayers + 1) x NLeg_all` |
    | `IMS_Leg_coeffs`             | `(NLayers + 1) x NLeg_all` |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np

    # NT corrections require a beam source, delta-M scaling (`f_arr > 0`), scattering
    # (`omega_arr > 0`) and a truncated phase function (`NLeg < NLeg_all`)
    if not (NT_cor and there_is_beam_source and np.any(f_arr > 0) and NLeg < NLeg_all and np.any(omega_arr > 0)):
        return u_star

    ############################################################ Setup ############################################################

    # Quantities used by the NT corrections which are independent of the evaluation points

    # TMS corrections: weighted Legendre coefficients of `mathscr_B` for each layer, without its factor that depends on `mu`
    # --------------------------------------------------------------------------------------------------------------------------
    Leg_coeffs_diff = weighted_Leg_coeffs_all / (1 - f_arr)[:, None]
    Leg_coeffs_diff[:, :NLeg] = (
        (2 * np.arange(NLeg) + 1)[None, :] * (f_arr / (1 - f_arr))[:, None]
    )
    TMS_Leg_coeffs = (scaled_omega_arr * I0_div_4pi)[:, None] * Leg_coeffs_diff
    exp_scaled_tau_arr_with_0 = np.exp(-scaled_tau_arr_with_0 / mu0)
    is_delta_scaled = np.any(scale_tau != 1)
    # --------------------------------------------------------------------------------------------------------------------------

    # IMS corrections
    # --------------------------------------------------------------------------------------------------------------------------
    weights = omega_arr * thickness_arr
    cumsum1 = np.cumsum(weights)                                     # (NLayers,)
    cumsum2 = np.cumsum(f_arr * weights)                             # (NLayers,)
    # Guarded divisions: a zero cumulative sum means no scattering above
    # the boundary, where `IMS_scalar` and `IMS_Leg_coeffs` correctly vanish
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

    scaled_mu0 = mu0 / (1 - omega_f_avg)                             # (NLayers + 1,)
    IMS_scalar = I0_div_4pi * omega_f_avg**2 / (1 - omega_f_avg)     # (NLayers + 1,)
    # Weighted Legendre coefficients of the IMS correction at each layer boundary, including its scalar factor
    IMS_Leg_coeffs = (IMS_scalar[:, None] * (2 * np.arange(NLeg_all) + 1)[None, :]) * (
        2 * Leg_coeffs_residue_avg - Leg_coeffs_residue_avg**2
    )                                                                # (NLayers + 1, NLeg_all)
    tau_arr_with_0 = np.concatenate([np.zeros(1), tau_arr])
    # --------------------------------------------------------------------------------------------------------------------------

    ###############################################################################################################################
    ##################################################### Apply NT Corrections ####################################################

    def _NT_corrections_at(
        mu_pos, abs_mu_neg, tau, phi, is_antiderivative_wrt_tau, TMS_statics_pos=None, TMS_statics_neg=None
    ):
        """NT (TMS + IMS) corrections at the upward `mu_pos` and the downward `-abs_mu_neg`, in that order.
        Returns an (Npos + Nneg, Ntau, Nphi) array. The optional `_TMS_mu_statics` of the two hemispheres
        are only passed for the quadrature nodes, where they are precomputed.
        """
        if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")
        Nmu_pos = len(mu_pos)
        Nmu = Nmu_pos + len(abs_mu_neg)
        Nphi = len(phi)

        # Atmospheric layer indices: smallest l with tau <= tau_arr[l]
        l = np.searchsorted(tau_arr, tau, side="left")

        # Delta-M scaling
        if is_delta_scaled:
            scaled_tau = scaled_tau_arr_with_0[l + 1] - (tau_arr[l] - tau) * scale_tau[l]
        else:
            scaled_tau = tau

        # The scattering angle cosines are shared by the TMS and IMS corrections, and so are the Legendre
        # polynomials at them. The IMS correction applies only to downward directions and only requires the
        # series of the boundaries of the layers which contain the `tau` values.
        nu = np.reshape(
            subroutines.calculate_nu(np.concatenate([mu_pos, -abs_mu_neg]), phi, -mu0, phi0),
            Nmu * Nphi,
        )
        series = [(TMS_Leg_coeffs, 0)]
        if Nmu_pos < Nmu:
            boundaries = np.unique(np.concatenate([l, l + 1])) if is_atmos_multilayered else np.array([1])
            series.append((IMS_Leg_coeffs[boundaries], Nmu_pos * Nphi))
        series_values = _legendre_series(nu, series)

        # TMS correction for the intensity (see section 3.7.2)
        NT_corrections = _TMS_correction(
            mu_pos, abs_mu_neg,
            np.moveaxis(np.reshape(series_values[0], (NLayers, Nmu, Nphi)), 0, 1),
            l, scaled_tau,
            is_antiderivative_wrt_tau,
            mu0,
            scale_tau, scaled_thickness_arr,
            scaled_tau_arr_with_0, exp_scaled_tau_arr_with_0,
            NLayers, is_atmos_multilayered,
            autograd_compatible,
            TMS_statics_pos,
            TMS_statics_neg,
        )

        # IMS correction for the intensity (see section 3.7.2); it applies only to downward directions
        if Nmu_pos < Nmu:
            IMS_corrections = _IMS_correction(
                abs_mu_neg,
                np.reshape(series_values[1], (len(boundaries), Nmu - Nmu_pos, Nphi)), boundaries,
                tau, l,
                is_antiderivative_wrt_tau,
                scaled_mu0, tau_arr_with_0,
                autograd_compatible,
            )
            if autograd_compatible:
                NT_corrections = NT_corrections - np.concatenate(
                    [np.zeros((Nmu_pos, len(tau), Nphi)), IMS_corrections], axis=0
                )
            else:
                NT_corrections[Nmu_pos:, :, :] -= IMS_corrections

        return NT_corrections

    ###############################################################################################################################

    if interpolate_mu:

        # The corrected intensity: `u_star` plus the TMS and IMS corrections evaluated at the same `mu` values
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(mu, tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False):
            mu = np.atleast_1d(mu)
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            if not np.all(np.abs(mu) <= 1):
                raise ValueError("mu values must be between -1 and 1.")
            if np.any(np.abs(mu) < 1e-8):
                raise ValueError("NT corrections require `mu` values bounded away from 0 (|mu| >= 1e-8).")

            mask_pos = mu > 0
            Nmu_pos = np.count_nonzero(mask_pos)
            if Nmu_pos == len(mu):
                NT_corrections = _NT_corrections_at(mu, mu[:0], tau, phi, is_antiderivative_wrt_tau)
            elif Nmu_pos == 0:
                NT_corrections = _NT_corrections_at(mu[:0], -mu, tau, phi, is_antiderivative_wrt_tau)
            else:
                mask_neg = ~mask_pos
                unordered = _NT_corrections_at(mu[mask_pos], -mu[mask_neg], tau, phi, is_antiderivative_wrt_tau)
                # Return to the original `mu` ordering
                NT_corrections = np.empty(np.shape(unordered))
                NT_corrections[mask_pos] = unordered[:Nmu_pos]
                NT_corrections[mask_neg] = unordered[Nmu_pos:]
            NT_corrections = rescale_factor * np.squeeze(NT_corrections)[()]

            # The corrected intensity
            if return_Fourier_error:
                u_star_val, Fourier_error = u_star(mu, tau, phi, is_antiderivative_wrt_tau, True)
                return u_star_val + NT_corrections, Fourier_error
            else:
                return u_star(mu, tau, phi, is_antiderivative_wrt_tau) + NT_corrections

    else:

        # The contributions from other layers to the TMS correction are partly precomputed at the quadrature nodes
        if is_atmos_multilayered:
            TMS_statics_quad_pos = _TMS_mu_statics(M_inv, True, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0)
            TMS_statics_quad_neg = _TMS_mu_statics(M_inv, False, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0)
        else:
            TMS_statics_quad_pos = None
            TMS_statics_quad_neg = None

        # The corrected intensity: `u_star` plus the TMS and IMS corrections evaluated at the quadrature nodes
        # --------------------------------------------------------------------------------------------------------------------------
        def u_corrected(tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False):
            tau = np.atleast_1d(tau)
            phi = np.atleast_1d(phi)
            NT_corrections = rescale_factor * np.squeeze(
                _NT_corrections_at(
                    mu_arr_pos, mu_arr_pos, tau, phi, is_antiderivative_wrt_tau,
                    TMS_statics_quad_pos, TMS_statics_quad_neg,
                )
            )

            # The corrected intensity
            if return_Fourier_error:
                u_star_val, Fourier_error = u_star(tau, phi, is_antiderivative_wrt_tau, True)
                return u_star_val + NT_corrections, Fourier_error
            else:
                return u_star(tau, phi, is_antiderivative_wrt_tau) + NT_corrections

    # `u_corrected` wraps `u_star`, so it needs its own copy of the attribute
    # that `_assemble_intensity_and_fluxes` attaches to `u_star`
    u_corrected.tau_arr = tau_arr

    return u_corrected


############################################### NT-correction helper functions ###############################################

_LEGENDRE_TABLE_SIZE = 2**22
def _legendre_series(x, series):
    """Values at the points `x` (Npts,) of sets of Legendre series which share the Legendre polynomials at `x`.
    Each entry of `series` is a tuple of the weighted Legendre coefficients (Nseries, NLeg_all) of a set of series
    and the index of the first point of `x` at which that set is required.
    Returns an array of shape (Nseries, Npts - first index) for each set.
    """
    NLeg_all = np.shape(series[0][0])[1]
    Npts = len(x)
    outputs = [np.empty((np.shape(coeffs)[0], Npts - first)) for coeffs, first in series]
    step = max(1, _LEGENDRE_TABLE_SIZE // NLeg_all)
    for start in range(0, Npts, step):
        stop = min(start + step, Npts)
        P = sc.special.legendre_p_all(NLeg_all - 1, x[start:stop])[0]   # (NLeg_all, stop - start)
        for (coeffs, first), output in zip(series, outputs):
            if first < stop:
                skip = max(first - start, 0)
                output[:, start + skip - first : stop - first] = coeffs @ P[:, skip:]
    return outputs


def _TMS_mu_statics(M_inv_h, positive_hemisphere, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0):
    """Precomputes, for one hemisphere of H `mu` values, the per-layer factors of the contributions
    from other layers to the TMS correction (refer to section 3.7.2 of the Comprehensive Documentation),
    which `_TMS_correction` accumulates across layers. The factors are independent of `tau` and `phi`,
    so `_NT_corrections` precomputes them at the quadrature nodes.
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
    # Transmittance of each layer along each `mu`
    delta = scaled_thickness_arr[None, :] * M_inv_h[:, None]                        # (H, NLayers)
    decay = np.exp(-delta)                                                          # (H, NLayers)

    # The square-bracketed difference of exponentials of each layer's contribution, referenced to the boundary
    # of that layer which faces the evaluation points: its top for upward `mu`, its bottom for downward `mu`
    if positive_hemisphere:
        delta_pos = delta + (scaled_thickness_arr / mu0)[None, :]                   # (H, NLayers)
        fac_pos = -np.expm1(-delta_pos)                                             # (H, NLayers)
        coef = fac_pos * exp_scaled_tau_arr_with_0[:-1][None, :]                    # (H, NLayers)
    else:
        # The beam factor of `mathscr_B` is folded in (see `_TMS_correction`). Near its removable singularity
        # (|w| < 1) use phi1(-w); elsewhere factor out the larger exponential so that `expm1` sees non-positive arguments
        w = delta - (scaled_thickness_arr / mu0)[None, :]                           # (H, NLayers)
        y = M_inv_h - 1 / mu0                                                       # (H,)
        near_singularity = np.abs(w) < 1.0
        y_safe = np.where(y == 0.0, 1.0, y)[:, None]
        w_safe = np.where(near_singularity & (w != 0.0), w, 1.0)
        phi1_neg_w = np.where(w == 0.0, 1.0, -np.expm1(-w_safe) / w_safe)           # phi1(-w)
        em1 = np.expm1(-np.abs(w))                                                  # (H, NLayers)
        exp_x1 = exp_scaled_tau_arr_with_0[1:][None, :]                             # (1, NLayers)
        exp_x0 = decay * exp_scaled_tau_arr_with_0[:-1][None, :]                    # (H, NLayers)
        coef = M_inv_h[:, None] * np.where(
            near_singularity,
            exp_x1 * scaled_thickness_arr[None, :] * phi1_neg_w,
            np.where(w >= 0.0, -em1 * exp_x1, em1 * exp_x0) / y_safe,
        )                                                                           # (H, NLayers)

    return decay, coef


def _TMS_correction(
    mu_pos, abs_mu_neg,             # Positive-hemisphere `mu` values; absolute values of negative-hemisphere `mu`
    mathscr_B_series,               # `mathscr_B` of each layer without its factor that depends on `mu` (positive-hemisphere rows first)
    l, scaled_tau,                  # Layer indices and delta-scaled optical depths of the evaluation points
    is_antiderivative_wrt_tau,      # Switch to an antiderivative of the function?
    mu0,                            # Cosine of polar angle of the incident beam
    scale_tau,                      # Delta-scale factor for tau
    scaled_thickness_arr,           # Delta-scaled thickness of each layer
    scaled_tau_arr_with_0,          # Delta-scaled lower boundary of layers with 0 inserted in front
    exp_scaled_tau_arr_with_0,      # exp(-scaled_tau_arr_with_0 / mu0)
    NLayers,                        # Number of layers
    is_atmos_multilayered,          # Is the atmosphere multilayered?
    autograd_compatible,            # Should the output functions be compatible with autograd?
    statics_pos=None,               # Optional precomputed `_TMS_mu_statics` for `mu_pos` (multi-layer only)
    statics_neg=None,               # Optional precomputed `_TMS_mu_statics` for `abs_mu_neg` (multi-layer only)
):
    """TMS correction to the intensity at arbitrary `mu` evaluation points.
    Refer to section 3.7.2 of the Comprehensive Documentation.
    Returns an ndarray of shape (Npos + Nneg, Ntau, Nphi) with the positive-hemisphere rows first.
    This function is private and should be called only through the closures that `_NT_corrections` creates.

    Arguments of _TMS_correction
    |          Variable            |          Type / Shape           |
    | ---------------------------- | ------------------------------- |
    | `mu_pos`                     | `Npos`                          |
    | `abs_mu_neg`                 | `Nneg`                          |
    | `mathscr_B_series`           | `(Npos + Nneg) x NLayers x Nphi`|
    | `l`                          | `Ntau`                          |
    | `scaled_tau`                 | `Ntau`                          |
    | `scale_tau`                  | `NLayers`                       |
    | `scaled_thickness_arr`       | `NLayers`                       |
    | `scaled_tau_arr_with_0`      | `NLayers + 1`                   |
    | `exp_scaled_tau_arr_with_0`  | `NLayers + 1`                   |
    | `statics_pos`                | 2-tuple of `Npos x NLayers`     |
    | `statics_neg`                | 2-tuple of `Nneg x NLayers`     |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np

    Nmu_pos = len(mu_pos)
    Nmu_neg = len(abs_mu_neg)
    Nphi = np.shape(mathscr_B_series)[-1]

    scaled_tau_arr_l = scaled_tau_arr_with_0[l + 1]
    scaled_tau_arr_lm1 = scaled_tau_arr_with_0[l]
    exp_neg_scaled_tau_div_mu0 = np.exp(-scaled_tau / mu0)              # (Ntau,)
    if is_antiderivative_wrt_tau:
        # Antiderivatives with respect to the unscaled tau, hence the divisions by d tau* / d tau = `scale_tau`
        scale_tau_l = scale_tau[l]                                      # (Ntau,)
    if is_atmos_multilayered:
        # The layers which contain the `tau` values. If they are few, the contributions from the other layers
        # are summed directly for each of them, else they are accumulated by a single scan over the layers.
        l_uniq = np.unique(l)
        sum_directly = len(l_uniq) <= 2
        l_min = int(l_uniq[0])
        l_max = int(l_uniq[-1])

    solutions = []

    # Upward `mu`
    # --------------------------------------------------------------------------------------------------------------------------
    if Nmu_pos > 0:
        Bpos = (mu0 / (mu0 + mu_pos))[:, None, None] * mathscr_B_series[:Nmu_pos]  # mathscr_B; (Npos, NLayers, Nphi)

        # The square-bracketed differences of exponentials for the layer containing each `tau`
        if is_antiderivative_wrt_tau:
            TMS_correction_pos = (exp_neg_scaled_tau_div_mu0 / (-scale_tau_l / mu0))[None, :] - np.exp(
                (scaled_tau - scaled_tau_arr_l)[None, :] / mu_pos[:, None]
                - scaled_tau_arr_l[None, :] / mu0
            ) * (mu_pos[:, None] / scale_tau_l[None, :])
        else:
            TMS_correction_pos = exp_neg_scaled_tau_div_mu0[None, :] - np.exp(
                (scaled_tau - scaled_tau_arr_l)[None, :] / mu_pos[:, None]
                - scaled_tau_arr_l[None, :] / mu0
            )

        # Contribution from the layer containing each `tau`
        sol_pos = Bpos[:, l, :] * TMS_correction_pos[:, :, None]        # (Npos, Ntau, Nphi)

        # Contributions from the layers below, if some `tau` has layers below. The antiderivative factor
        # `mu / scale_tau` belongs to the layer containing `tau`, not to the contributing layer, and is
        # therefore applied after the contributions are accumulated.
        if is_atmos_multilayered and l_min < NLayers - 1:
            M_inv_pos = 1 / mu_pos
            if statics_pos is None:
                decay_pos, coef_pos = _TMS_mu_statics(
                    M_inv_pos, True, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0
                )
            else:
                decay_pos, coef_pos = statics_pos

            # Acc_r is the contribution of the layers below layer r, referenced to the bottom of layer r:
            # Acc_{r-1} = coef_pos[:, r] * Bpos[:, r, :] + decay_pos[:, r] * Acc_r
            if sum_directly:
                # Layer r contributes coef_pos[:, r] * Bpos[:, r, :], attenuated from its top to the bottom of layer k
                Acc_pos = np.stack(
                    [
                        np.einsum(
                            "ar, arp -> ap",
                            coef_pos[:, k + 1 :] * np.exp(
                                M_inv_pos[:, None]
                                * (scaled_tau_arr_with_0[k + 1] - scaled_tau_arr_with_0[k + 1 : -1])[None, :]
                            ),
                            Bpos[:, k + 1 :, :],
                        )
                        for k in l_uniq
                    ],
                    axis=1,
                )                                                       # (Npos, len(l_uniq), Nphi)
                Acc_index = np.searchsorted(l_uniq, l)
            elif autograd_compatible:
                acc0 = np.zeros((Nmu_pos, Nphi))
                acc = acc0
                cols_rev = [acc0]  # Acc_{NLayers-1} = 0
                for r in range(NLayers - 1, l_min, -1):
                    term = coef_pos[:, r][:, None] * Bpos[:, r, :]          # (Npos, Nphi)
                    acc = term + decay_pos[:, r][:, None] * acc
                    cols_rev.append(acc)
                Acc_pos = np.stack(cols_rev[::-1], axis=1)                  # (Npos, Nacc, Nphi)
                Acc_index = l - l_min
            else:
                Nacc = NLayers - l_min  # Only layers >= l_min are needed; layer r is stored at index r - l_min
                Acc_pos = np.empty((Nmu_pos, Nacc, Nphi))
                Acc_pos[:, Nacc - 1, :] = 0.0                               # Acc_{NLayers-1}
                for r in range(NLayers - 1, l_min, -1):
                    i_dest = r - 1 - l_min
                    i_source = r - l_min
                    np.multiply(coef_pos[:, r:r+1], Bpos[:, r, :],
                                out=Acc_pos[:, i_dest, :])
                    Acc_pos[:, i_dest, :] += decay_pos[:, r:r+1] * Acc_pos[:, i_source, :]
                Acc_index = l - l_min

            # Transmit from the bottom of the layer containing `tau` to `tau`
            expfac_pos = np.exp(M_inv_pos[:, None] * (scaled_tau - scaled_tau_arr_l)[None, :])  # (Npos, Ntau)
            contrib_pos = Acc_pos[:, Acc_index, :] * expfac_pos[:, :, None]                     # (Npos, Ntau, Nphi)

            if is_antiderivative_wrt_tau:
                contrib_pos = contrib_pos * (mu_pos[:, None] / scale_tau_l[None, :])[:, :, None]

            sol_pos = sol_pos + contrib_pos

        solutions.append(sol_pos)
    # --------------------------------------------------------------------------------------------------------------------------

    # Downward `mu`
    # --------------------------------------------------------------------------------------------------------------------------
    if Nmu_neg > 0:
        # For downward `mu` the beam factor mu0 / (mu0 - mu) = 1 / (mu * y_neg) of `mathscr_B` is singular at
        # mu = mu0 and is instead folded into the square-bracketed differences of exponentials (below and in
        # `_TMS_mu_statics`)
        Bneg = mathscr_B_series[Nmu_pos:]                                   # (Nneg, NLayers, Nphi)

        # Near the removable singularity (|w| < 1) the brackets are evaluated through phi1(-w),
        # which is exact at w = 0; elsewhere the direct form is used
        y_neg = 1 / abs_mu_neg - 1 / mu0                                    # (Nneg,)
        Delta = scaled_tau - scaled_tau_arr_lm1                             # (Ntau,)
        w = Delta[None, :] * y_neg[:, None]                                 # (Nneg, Ntau)
        near_singularity = np.abs(w) < 1.0
        y_safe = np.where(y_neg == 0.0, 1.0, y_neg)[:, None]
        w_safe = np.where(near_singularity & (w != 0.0), w, 1.0)
        phi1_neg_w = np.where(w == 0.0, 1.0, -np.expm1(-w_safe) / w_safe)   # phi1(-w)
        E1 = exp_neg_scaled_tau_div_mu0[None, :]
        E2 = np.exp(
            (scaled_tau_arr_lm1 - scaled_tau)[None, :] / abs_mu_neg[:, None]
            - scaled_tau_arr_lm1[None, :] / mu0
        )

        # The square-bracketed differences of exponentials for the layer containing each `tau`
        if is_antiderivative_wrt_tau:
            TMS_correction_neg = np.where(
                near_singularity,
                -(E1 / scale_tau_l[None, :]) * (Delta[None, :] * phi1_neg_w + mu0),
                (abs_mu_neg[:, None] * E2 - mu0 * E1)
                / (scale_tau_l[None, :] * abs_mu_neg[:, None] * y_safe),
            )
        else:
            TMS_correction_neg = np.where(
                near_singularity,
                (Delta[None, :] / abs_mu_neg[:, None]) * phi1_neg_w * E1,
                (E1 - E2) / (abs_mu_neg[:, None] * y_safe),
            )

        # Contribution from the layer containing each `tau`
        sol_neg = Bneg[:, l, :] * TMS_correction_neg[:, :, None]            # (Nneg, Ntau, Nphi)

        # Contributions from the layers above, if some `tau` has layers above
        if is_atmos_multilayered and l_max > 0:
            M_inv_neg = 1 / abs_mu_neg
            if statics_neg is None:
                decay_neg, coef_neg = _TMS_mu_statics(
                    M_inv_neg, False, scaled_thickness_arr, mu0, exp_scaled_tau_arr_with_0
                )
            else:
                decay_neg, coef_neg = statics_neg

            # Acc_r is the contribution of the layers above layer r, referenced to the top of layer r:
            # Acc_{r+1} = coef_neg[:, r] * Bneg[:, r, :] + decay_neg[:, r] * Acc_r
            if sum_directly:
                # Layer r contributes coef_neg[:, r] * Bneg[:, r, :], attenuated from its bottom to the top of layer k
                Acc_neg = np.stack(
                    [
                        np.einsum(
                            "ar, arp -> ap",
                            coef_neg[:, :k] * np.exp(
                                M_inv_neg[:, None]
                                * (scaled_tau_arr_with_0[1 : k + 1] - scaled_tau_arr_with_0[k])[None, :]
                            ),
                            Bneg[:, :k, :],
                        )
                        for k in l_uniq
                    ],
                    axis=1,
                )                                                       # (Nneg, len(l_uniq), Nphi)
                Acc_index = np.searchsorted(l_uniq, l)
            elif autograd_compatible:
                acc0 = np.zeros((Nmu_neg, Nphi))
                acc = acc0
                cols = [acc0]  # Acc_0 = 0
                for r in range(0, l_max):
                    term = coef_neg[:, r][:, None] * Bneg[:, r, :]          # (Nneg, Nphi)
                    acc = term + decay_neg[:, r][:, None] * acc
                    cols.append(acc)
                Acc_neg = np.stack(cols, axis=1)                            # (Nneg, Nacc, Nphi)
                Acc_index = l
            else:
                Nacc = l_max + 1  # Only layers <= l_max are needed
                Acc_neg = np.empty((Nmu_neg, Nacc, Nphi))
                Acc_neg[:, 0, :] = 0.0                                      # Acc_0
                for r in range(0, l_max):
                    np.multiply(coef_neg[:, r:r+1], Bneg[:, r, :],
                                out=Acc_neg[:, r + 1, :])
                    Acc_neg[:, r + 1, :] += decay_neg[:, r:r+1] * Acc_neg[:, r, :]
                Acc_index = l

            # Transmit from the top of the layer containing `tau` to `tau`
            expfac_neg = np.exp(M_inv_neg[:, None] * (scaled_tau_arr_lm1 - scaled_tau)[None, :])  # (Nneg, Ntau)
            contrib_neg = Acc_neg[:, Acc_index, :] * expfac_neg[:, :, None]                       # (Nneg, Ntau, Nphi)

            if is_antiderivative_wrt_tau:
                contrib_neg = -contrib_neg * (abs_mu_neg[:, None] / scale_tau_l[None, :])[:, :, None]

            sol_neg = sol_neg + contrib_neg

        solutions.append(sol_neg)
    # --------------------------------------------------------------------------------------------------------------------------

    if len(solutions) == 1:
        return solutions[0]
    if len(solutions) == 0:
        return np.zeros((0, len(l), Nphi))
    return np.concatenate(solutions, axis=0)


# 17 coefficients of the Taylor series of phi2(z) = (e^z - 1 - z)/z^2 = sum_k z^k / (k + 2)!
_PHI2_TAYLOR_COEFFS = 1.0 / np.cumprod(np.arange(2, 19))
def _IMS_correction(
    abs_mu_neg,                     # Absolute values of negative-hemisphere `mu`
    IMS_series,                     # Scalar factor times the square-bracketed Legendre series, for each of `boundaries`
    boundaries,                     # Sorted indices of the layer boundaries whose series are in `IMS_series`
    tau,                            # Optical depths of the evaluation points (NOT delta-scaled)
    l,                              # Atmospheric layer indices of the evaluation points
    is_antiderivative_wrt_tau,      # Switch to an antiderivative of the function?
    scaled_mu0,                     # mu0' at each layer boundary
    tau_arr_with_0,                 # Lower boundary of layers with 0 inserted in front
    autograd_compatible,            # Should the output functions be compatible with autograd?
):
    """IMS correction to the intensity at arbitrary downward `mu` evaluation points.
    Refer to section 3.7.2 of the Comprehensive Documentation.
    Returns an ndarray of shape (Nneg, Ntau, Nphi).
    This function is private and should be called only through the closures that `_NT_corrections` creates.
    The boundary statics are indexed k = 0, ..., NLayers where the k = 0 (top of atmosphere) set
    equals the k = 1 set. Layers are 0-indexed, so layer `l` is layer l + 1 of section 3.7.2
    and its correction interpolates between the boundary sets k = l and k = l + 1.

    Arguments of _IMS_correction
    |          Variable            |          Type / Shape        |
    | ---------------------------- | ---------------------------- |
    | `abs_mu_neg`                 | `Nneg`                       |
    | `IMS_series`                 | `Nboundaries x Nneg x Nphi`  |
    | `boundaries`                 | `Nboundaries`                |
    | `tau`                        | `Ntau`                       |
    | `l`                          | `Ntau`                       |
    | `scaled_mu0`                 | `NLayers + 1`                |
    | `tau_arr_with_0`             | `NLayers + 1`                |
    """
    if autograd_compatible:
        import autograd.numpy as np
    else:
        import numpy as np

    exp_tau_mu = np.exp(-tau[None, :] / abs_mu_neg[:, None])            # (Nneg, Ntau)

    def chi_kernels(smu0, need_chichi_anti=True):
        """chi or, on the antiderivative branch, chi_anti and optionally chichi_anti (the first and
        second tau-antiderivatives of chi), each of shape (Nneg, Ntau), for the `scaled_mu0` value
        `smu0` (Ntau,) of each evaluation point. Near the removable singularity at mu = smu0 (|z| < 1)
        each is evaluated through phi2(z), and phi2 through its Taylor series."""
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
                / (1 + x * smu0[None, :]),                              # 1 + x * smu0 = smu0 / mu > 0
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

    if len(scaled_mu0) == 2:
        # Single-layer atmosphere: the two boundary sets are identical so there is nothing to interpolate
        smu0 = scaled_mu0[np.full(len(tau), 1)]
        scalP = IMS_series[0][:, None, :]
        if is_antiderivative_wrt_tau:
            return scalP * chi_kernels(smu0, need_chichi_anti=False)[0][:, :, None]
        else:
            return scalP * chi_kernels(smu0)[:, :, None]

    # Interpolate between the correction functions of the top ("lo", k = l) and bottom ("hi", k = l + 1)
    # boundaries of the layer containing each `tau`
    t0 = tau_arr_with_0[l]                                              # (Ntau,)
    thickness_l = tau_arr_with_0[l + 1] - t0                            # (Ntau,)

    # The scalar factor times the square-bracketed Legendre series (DISORT's `PSPIKE`) of the two boundary sets
    scalP_lo = np.moveaxis(IMS_series[np.searchsorted(boundaries, l)], 0, 1)      # (Nneg, Ntau, Nphi)
    scalP_hi = np.moveaxis(IMS_series[np.searchsorted(boundaries, l + 1)], 0, 1)  # (Nneg, Ntau, Nphi)

    if is_antiderivative_wrt_tau:
        # Per-layer antiderivative of the interpolated correction (refer to section 3.7.2),
        # where Y1 and Y2 are the first and second antiderivatives of the boundary correction functions
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

##############################################################################################################################
