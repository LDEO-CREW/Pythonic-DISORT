from PythonicDISORT.subroutines import (
    _mathscr_v_coeffs,
    _poly_tau_derivative,
    _check_tau_derivative_order,
    _Fourier_error,
)
from PythonicDISORT._solve_for_gen_and_part_sols import _solve_for_gen_and_part_sols
from PythonicDISORT._solve_for_coeffs import _solve_for_coeffs
from PythonicDISORT._IotS import _build_IotS

import numpy as np
from math import pi

def _assemble_intensity_and_fluxes(
    scaled_omega_arr,                   # Delta-scaled single-scattering albedos
    tau_arr,                            # Lower boundary of layers
    scaled_tau_arr_with_0,              # Delta-scaled lower boundary of layers with 0 inserted in front
    mu_arr_pos,                         # Upper hemisphere quadrature nodes (cosine polar angles)
    M_inv, W,                           # 1) 1 / `mu_arr_pos`; 2) quadrature weights for each hemisphere
    N, NQuad, NLeg,                     # Number of 1) upper 2) both hemispheres quadrature nodes; 3) phase function Legendre coefficients
    NFourier, NLayers, NBDRF,           # Number of 1) intensity Fourier modes; 2) layers; 3) BDRF Fourier modes
    is_atmos_multilayered,              # Is the atmosphere multilayered?
    weighted_scaled_Leg_coeffs,         # Weighted and delta-scaled Legendre coefficients
    BDRF_Fourier_modes,                 # BDRF Fourier modes
    mu0, I0, I0_div_4pi,                # Primary direct beam parameters
    rescale_factor, phi0,               # Secondary direct beam parameters
    there_is_beam_source,               # Is there a beam source?
    mu0_arr,                            # Cosine of the direct beam in each layer
    slant_tau_arr_with_0,               # Slant optical depth of the direct beam at each layer boundary
    scaled_mu0_arr,                     # Cosine of the direct beam in each delta-scaled layer
    scaled_slant_tau_arr_with_0,        # Slant optical depth of the direct beam at each delta-scaled layer boundary
    b_pos, b_neg,                       # Dirichlet BCs
    b_pos_is_scalar, b_neg_is_scalar,   # Is each Dirichlet BCs scalar and so isotropic?
    b_pos_is_vector, b_neg_is_vector,   # Is each Dirichlet BCs vector?
    scaled_s_poly_coeffs,               # Delta-scaled polynomial coefficients of isotropic source
    there_is_iso_source,                # Is there an isotropic source?
    scale_tau,                          # Delta-scale factor for tau
    only_flux,                          # Only compute fluxes?
    use_banded_solver_NLayers,          # Number of layers above or equal which to use `scipy.linalg.solve_banded`
    cache_asso_leg,                     # Assoc. Legendre table cache mode: None, "no_mu0" or "mu0"
    interpolate_mu,                     # Should the intensity functions take `mu` as an argument?
):
    """Assembles the solution functions: intensity `u`, its zeroth Fourier mode `u0`, upward flux `flux_up`,
    downward flux `flux_down` from the previously solved eigenpairs and coefficients from the boundary conditions.
    Returns these solution functions. If `interpolate_mu`, the intensity functions are instead those of
    `_build_IotS`, which take `mu` as their first argument.
    This function is wrapped by the `pydisort` function.
    It should be called through `pydisort` and never directly.
    It has many seemingly redundant arguments, which are described below,
    to maximize precomputation in `pydisort`
    These arguments are passed on to the `_solve_for_gen_and_part_sols` and `_solve_for_coeffs` functions
    which this function wraps. See the Jupyter Notebook, especially section 3, for
    documentation, explanation and derivation.

    Every solution function is a sum of exponentials in `tau` and, if there are isotropic internal sources,
    a polynomial in `tau`. Its `tau`-derivative of order `tau_derivative_order` (an antiderivative if -1) therefore
    multiplies each exponential by its rate to the power of that order and differentiates the polynomial.

    Arguments of _assemble_intensity_and_fluxes
    |            Variable            |                 Type / Shape                 |
    | ------------------------------ | -------------------------------------------- |
    | `scaled_omega_arr`             | `NLayers`                                    |
    | `tau_arr`                      | `NLayers`                                    |
    | `scaled_tau_arr_with_0`        | `NLayers + 1`                                |
    | `mu_arr_pos`                   | `NQuad/2`                                    |
    | `M_inv`                        | `NQuad/2`                                    |
    | `W`                            | `NQuad/2`                                    |
    | `N`                            | scalar                                       |
    | `NQuad`                        | scalar                                       |
    | `NLeg`                         | scalar                                       |
    | `NFourier`                     | scalar                                       |
    | `NLayers`                      | scalar                                       |
    | `NBDRF`                        | scalar                                       |
    | `is_atmos_multilayered`        | boolean                                      |
    | `weighted_scaled_Leg_coeffs`   | `NLayers x NLeg`                             |
    | `BDRF_Fourier_modes`           | `NBDRF`                                      |
    | `mu0`                          | scalar                                       |
    | `I0`                           | scalar                                       |
    | `I0_div_4pi`                   | scalar                                       |
    | `rescale_factor`               | scalar                                       |
    | `phi0`                         | scalar                                       |
    | `there_is_beam_source`         | boolean                                      |
    | `mu0_arr`                      | `NLayers` or `None`                          |
    | `slant_tau_arr_with_0`         | `NLayers + 1` or `None`                      |
    | `scaled_mu0_arr`               | `NLayers` or `None`                          |
    | `scaled_slant_tau_arr_with_0`  | `NLayers + 1` or `None`                      |
    | `b_pos`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  |
    | `b_neg`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  |
    | `b_pos_is_scalar`              | boolean                                      |
    | `b_neg_is_scalar`              | boolean                                      |
    | `b_pos_is_vector`              | boolean                                      |
    | `b_neg_is_vector`              | boolean                                      |
    | `scaled_s_poly_coeffs`         | `NLayers x Nscoeffs`                         |
    | `there_is_iso_source`          | boolean                                      |
    | `scale_tau`                    | `NLayers`                                    |
    | `only_flux`                    | boolean                                      |
    | `use_banded_solver_NLayers`    | scalar                                       |
    | `cache_asso_leg`               | None, "no_mu0" or "mu0"                      |
    | `interpolate_mu`               | boolean                                      |

    Notable internal variables of _assemble_intensity_and_fluxes
    |      Variable      |             Type / Shape               |
    | ------------------ | -------------------------------------- |
    | `G_collect`        | `NFourier x NLayers x NQuad x NQuad`   |
    | `K_collect`        | `NFourier x NLayers x NQuad`           |
    | `B_collect`        | `NFourier x NLayers x NQuad` or `None` |
    | `mathscr_v_coeffs` | `NLayers x NQuad x Nscoeffs` or `None` |
    | `C_collect`        | `NFourier x NLayers x NQuad`           |
    | `GC_collect`       | `NFourier x NLayers x NQuad x NQuad`   |
    | `tau_rates`        | `NFourier x NLayers x NQuad`           | Rate of each exponential with respect to `tau`
    | `scaled_tau_refs`  | `NLayers x NQuad`                      | Delta-scaled optical depth at which each exponential equals 1
    """
    ################################## Assemble uncorrected intensity and flux functions #######################################

    # Compute all the necessary quantities
    # --------------------------------------------------------------------------------------------------------------------------
    G_collect, K_collect, B_collect, G_inv_collect_0, asso_leg_table = _solve_for_gen_and_part_sols(
        NFourier,
        scaled_omega_arr,
        mu_arr_pos,
        M_inv, W,
        N, NQuad, NLeg,
        NLayers,
        weighted_scaled_Leg_coeffs,
        mu0, I0_div_4pi,
        there_is_beam_source,
        scaled_mu0_arr,
        there_is_iso_source,
        cache_asso_leg,
    )
    # Particular solution for isotropic internal sources (refer to section 3.6.1 of the Comprehensive Documentation)
    if there_is_iso_source:
        mathscr_v_coeffs = _mathscr_v_coeffs(
            scaled_s_poly_coeffs,
            G_collect[0],
            K_collect[0],
            G_inv_collect_0 @ np.concatenate([M_inv, -M_inv]),
        )
    else:
        mathscr_v_coeffs = None

    C_collect = _solve_for_coeffs(
        NFourier,
        G_collect,
        K_collect,
        B_collect,
        mathscr_v_coeffs,
        scaled_tau_arr_with_0,
        mu_arr_pos, mu_arr_pos * W,
        N, NQuad,
        NLayers, NBDRF,
        is_atmos_multilayered,
        BDRF_Fourier_modes,
        mu0, I0_div_4pi,
        there_is_beam_source,
        scaled_slant_tau_arr_with_0,
        b_pos, b_neg,
        b_pos_is_scalar, b_neg_is_scalar,
        b_pos_is_vector, b_neg_is_vector,
        there_is_iso_source,
        use_banded_solver_NLayers,
    )

    GC_collect = G_collect * C_collect[:, :, None, :]
    tau_rates = scale_tau[None, :, None] * K_collect
    # The exponentials with negative eigenvalues equal 1 at the top of their layer and those with positive
    # eigenvalues at its bottom, so that none exceeds 1 within the layer
    scaled_tau_refs = np.concatenate(
        [np.tile(scaled_tau_arr_with_0[:-1, None], N), np.tile(scaled_tau_arr_with_0[1:, None], N)], axis=1
    )
    tau_arr_with_0 = np.concatenate([[0], tau_arr])
    # --------------------------------------------------------------------------------------------------------------------------

    # Value-keyed cache for the most recent `tau` evaluation grid, shared across the output
    # functions `u`, `u0`, `flux_up` and `flux_down` so that evaluating several outputs on the same
    # grid, e.g. `flux_up(tau); flux_down(tau)`, recomputes nothing.
    # --------------------------------------------------------------------------------------------------------------------------
    is_delta_scaled = np.any(scale_tau != 1)
    _tau_cache = {"tau": None}

    def _process_tau(tau):
        """Layer indices and delta-scaled `tau` for the evaluation grid `tau`, cached."""
        if _tau_cache["tau"] is None or not np.array_equal(tau, _tau_cache["tau"]):
            if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
                raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")
            # Atmospheric layer indices: smallest l with tau <= tau_arr[l]
            l = np.searchsorted(tau_arr, tau, side="left")
            if is_delta_scaled:
                scaled_tau = scaled_tau_arr_with_0[l + 1] - (tau_arr[l] - tau) * scale_tau[l]
            else:
                scaled_tau = tau
            _tau_cache.clear()
            _tau_cache.update(tau=np.array(tau, copy=True), l=l, scaled_tau=scaled_tau)
        return _tau_cache["l"], _tau_cache["scaled_tau"]

    def _exponentials(l, scaled_tau, Nm):
        """Exponentials of the homogeneous solution of the first `Nm` Fourier modes for the (cached)
        evaluation grid; (Nm, Ntau, NQuad)."""
        if len(_tau_cache.get("exponentials", ())) < Nm:
            _tau_cache["exponentials"] = np.exp(
                K_collect[:Nm, l, :] * (scaled_tau[:, None] - scaled_tau_refs[l, :])[None, :, :]
            )
        return _tau_cache["exponentials"][:Nm]

    def _scaled_beam(l, scaled_tau):
        """Transmittance of the direct beam in the delta-scaled atmosphere; (Ntau,)."""
        return np.exp(
            -(scaled_slant_tau_arr_with_0[l] + (scaled_tau - scaled_tau_arr_with_0[l]) / scaled_mu0_arr[l])
        )
    # --------------------------------------------------------------------------------------------------------------------------

    # Fourier modes of the intensity at the quadrature nodes (refer to section 3.7 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    def _um_at_nodes(tau, tau_derivative_order, Nm, nodes=slice(None)):
        """`tau`-derivative of order `tau_derivative_order` of the first `Nm` Fourier modes of the intensity
        at the quadrature `nodes` (a slice) and the optical depths `tau`; (Nm, number of nodes, Ntau)."""
        _check_tau_derivative_order(tau_derivative_order)
        l, scaled_tau = _process_tau(tau)

        um = np.einsum(
            "mtij, mtj -> mit",
            GC_collect[:Nm, l, nodes, :],
            _exponentials(l, scaled_tau, Nm) * tau_rates[:Nm, l, :] ** tau_derivative_order,
        )
        if there_is_beam_source:
            beam = _scaled_beam(l, scaled_tau) * (-scale_tau[l] / scaled_mu0_arr[l]) ** tau_derivative_order
            um += np.swapaxes(B_collect[:Nm, l, nodes], 1, 2) * beam[None, None, :]
        # Contribution from particular solution for isotropic internal sources
        if there_is_iso_source:
            um[0] += scale_tau[l] ** tau_derivative_order * (
                _poly_tau_derivative(mathscr_v_coeffs[l, nodes, :], scaled_tau, tau_derivative_order).T
            )
        return um

    def _direct_beam(tau, tau_derivative_order):
        """`tau`-derivative of order `tau_derivative_order` of the intensity of the direct beam at `tau` and of
        the same in the delta-scaled atmosphere; each (Ntau,). Their difference is the delta-scaled
        direct beam which is reclassified as diffuse (refer to section 3.8 of the Comprehensive Documentation)."""
        l, scaled_tau = _process_tau(tau)
        if not there_is_beam_source:
            return np.zeros(len(tau)), np.zeros(len(tau))
        direct_beam = I0 * (-1 / mu0_arr[l]) ** tau_derivative_order * np.exp(
            -(slant_tau_arr_with_0[l] + (tau - tau_arr_with_0[l]) / mu0_arr[l])
        )
        direct_beam_scaled = (
            I0 * (-scale_tau[l] / scaled_mu0_arr[l]) ** tau_derivative_order * _scaled_beam(l, scaled_tau)
        )
        return direct_beam, direct_beam_scaled
    # --------------------------------------------------------------------------------------------------------------------------

    # Construct the intensity function (refer to section 3.7 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    def u(tau, phi, tau_derivative_order=0, return_Fourier_error=False):
        """
        Intensity function with arguments `(tau, phi)` of types `(array or float, array or float)`.
        Returns an ndarray with axes corresponding to variation with `mu, tau, phi` respectively.
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        Pass `return_Fourier_error = True` (defaults to `False`) to return the
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
        """
        tau = np.atleast_1d(tau)
        phi = np.atleast_1d(phi)
        um = _um_at_nodes(tau, tau_derivative_order, NFourier)
        cosines = np.cos(np.arange(NFourier)[:, None] * (phi0 - phi)[None, :])
        u = np.einsum("mit, mp -> itp", um, cosines, optimize=True)

        if return_Fourier_error:
            return rescale_factor * np.squeeze(u), _Fourier_error(um[-1], cosines[-1], u)
        else:
            return rescale_factor * np.squeeze(u)
    # --------------------------------------------------------------------------------------------------------------------------

    # Construct u0
    # --------------------------------------------------------------------------------------------------------------------------
    def u0(tau, tau_derivative_order=0):
        """
        Zeroth Fourier mode of the intensity with argument `tau` (type: array or float).
        Returns an ndarray with axes corresponding to variation with `mu` and `tau` respectively.
        This function is useful for calculating actinic fluxes and other quantities of interest,
        but reclassification of delta-scaled flux and other corrections must be done manually
        (for actinic flux `generate_diff_act_flux_funcs` will automatically perform the reclassification).
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        """
        return rescale_factor * np.squeeze(_um_at_nodes(np.atleast_1d(tau), tau_derivative_order, 1)[0])


    def _act_dscale_reclassification(tau, tau_derivative_order=0):
        """Reclassification of delta-scaled actinic flux at `tau`; it complements `u0` in
        `subroutines.generate_diff_act_flux_funcs`. Attached to `u0` as an attribute of the same name.
        """
        direct_beam, direct_beam_scaled = _direct_beam(np.atleast_1d(tau), tau_derivative_order)
        return rescale_factor * (direct_beam_scaled - direct_beam)
    # --------------------------------------------------------------------------------------------------------------------------

    # Construct the flux functions (refer to section 3.8 of the Comprehensive Documentation)
    # --------------------------------------------------------------------------------------------------------------------------
    def flux_up(tau, tau_derivative_order=0):
        """
        (Energetic) Flux function with argument `tau` (type: array or float) for positive (upward) `mu` values.
        Returns the diffuse flux magnitudes (same type and size as `tau`).
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        """
        u0_pos = _um_at_nodes(np.atleast_1d(tau), tau_derivative_order, 1, slice(0, N))[0]
        flux = 2 * pi * (mu_arr_pos * W) @ u0_pos

        return rescale_factor * np.squeeze(flux)[()]


    def flux_down(tau, tau_derivative_order=0):
        """
        (Energetic) Flux function with argument `tau` (type: array or float) for negative (downward) `mu` values.
        Returns a tuple of the diffuse and direct flux magnitudes respectively where each entry is of the
        same type and size as `tau`.
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        """
        tau = np.atleast_1d(tau)
        u0_neg = _um_at_nodes(tau, tau_derivative_order, 1, slice(N, None))[0]
        direct_beam, direct_beam_scaled = _direct_beam(tau, tau_derivative_order)
        diffuse_flux = 2 * pi * (mu_arr_pos * W) @ u0_neg + mu0 * direct_beam_scaled - mu0 * direct_beam

        return (
            rescale_factor * np.squeeze(diffuse_flux)[()],
            rescale_factor * np.squeeze(mu0 * direct_beam)[()],
        )
    # --------------------------------------------------------------------------------------------------------------------------

    # Intensities at arbitrary polar angles by the integration of the source function (refer to section 3.7)
    # --------------------------------------------------------------------------------------------------------------------------
    u0._act_dscale_reclassification = _act_dscale_reclassification
    if interpolate_mu:
        u0_at_nodes = u0
        u0, u = _build_IotS(
            _process_tau,
            _um_at_nodes(tau_arr[-1:], 0, NFourier, slice(N, None))[:, :, 0],
            asso_leg_table,
            G_collect, K_collect, B_collect, C_collect, mathscr_v_coeffs,
            scaled_tau_arr_with_0, scale_tau,
            mu_arr_pos, W,
            N, NQuad, NLeg,
            NFourier, NLayers, NBDRF,
            scaled_omega_arr, weighted_scaled_Leg_coeffs,
            BDRF_Fourier_modes,
            mu0, I0_div_4pi, phi0,
            rescale_factor,
            there_is_beam_source,
            scaled_mu0_arr, scaled_slant_tau_arr_with_0,
            b_pos, b_neg,
            b_pos_is_scalar, b_neg_is_scalar,
            scaled_s_poly_coeffs,
            there_is_iso_source,
        )
        # `subroutines.generate_diff_act_flux_funcs` integrates the zeroth Fourier mode at the quadrature nodes
        u0._at_quad_nodes = u0_at_nodes
    # --------------------------------------------------------------------------------------------------------------------------

    # Attach other useful attributes
    flux_up.tau_arr = flux_down.tau_arr = u0.tau_arr = u.tau_arr = tau_arr

    if only_flux:
        return flux_up, flux_down, u0
    else:
        return flux_up, flux_down, u0, u
