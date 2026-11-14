from PythonicDISORT.subroutines import _poly_tau_derivative, _check_tau_derivative_order, _Fourier_error
from PythonicDISORT._solve_for_gen_and_part_sols import _asso_leg_table_by_offset

import numpy as np
import scipy as sc


def _phi1(x):
    """(1 - exp(-x)) / x for x >= 0, which is 1 at x = 0."""
    return np.divide(-np.expm1(-x), x, out=np.ones_like(x), where=x != 0)


def _poly_particular_solution(coeffs, mu, tau):
    """Particular solution along the cosines `mu` (Nmu,), at each of `tau` (Ntau,), for a source function which is
    a polynomial in tau with the coefficients `coeffs` (Ntau, Nmu, Ncoeffs): the sum over k of mu^k times the
    k-th derivative of the polynomial. Returns an array of shape (Ntau, Nmu)."""
    return sum(mu**k * _poly_tau_derivative(coeffs, tau, k) for k in range(np.shape(coeffs)[-1]))


def _build_IotS(
    process_tau,                        # Function for the layer index and delta-scaled optical depth of each `tau`
    um_bottom_neg,                      # Fourier modes of the downward intensity at the bottom boundary and quadrature nodes
    asso_leg_table,                     # Seminormalized associated Legendre table at `mu_arr_pos` (and `-mu0`) from the solver
    G_collect,                          # Eigenvector matrices
    K_collect,                          # Eigenvalues
    B_collect,                          # Coefficient vectors of the direct beam particular solutions
    C_collect,                          # Coefficients of the eigenvectors
    mathscr_v_coeffs,                   # Polynomial coefficients of the particular solution for isotropic internal sources
    scaled_tau_arr_with_0,              # Delta-scaled lower boundary of layers with 0 inserted in front
    scale_tau,                          # Delta-scale factor for tau
    mu_arr_pos, W,                      # Upper hemisphere quadrature nodes and weights
    N, NQuad, NLeg,                     # Number of 1) upper 2) both hemispheres quadrature nodes; 3) phase function Legendre coefficients
    NFourier, NLayers, NBDRF,           # Number of 1) intensity Fourier modes; 2) layers; 3) BDRF Fourier modes
    scaled_omega_arr,                   # Delta-scaled single-scattering albedos
    weighted_scaled_Leg_coeffs,         # Weighted and delta-scaled Legendre coefficients
    BDRF_Fourier_modes,                 # BDRF Fourier modes
    mu0, I0_div_4pi, phi0,              # Direct beam parameters (rescaled)
    rescale_factor,                     # Rescale factor of the sources
    there_is_beam_source,               # Is there a beam source?
    scaled_mu0_arr,                     # Cosine of the direct beam in each delta-scaled layer
    scaled_slant_tau_arr_with_0,        # Slant optical depth of the direct beam at each delta-scaled layer boundary
    b_pos, b_neg,                       # Dirichlet BCs (rescaled)
    b_pos_is_scalar, b_neg_is_scalar,   # Is each Dirichlet BC scalar and so isotropic?
    scaled_s_poly_coeffs,               # Delta-scaled polynomial coefficients of isotropic source (rescaled)
    there_is_iso_source,                # Is there an isotropic source?
):
    """Builds the intensity functions `u0` (the 0th Fourier mode) and `u` (all Fourier modes) at arbitrary polar
    angles, which `pydisort` returns if `interpolate_mu=True`, by the integration of the source function (IotS) of
    Stamnes (1982), as in the subroutines `TERPEV`, `TERPSO` and `USRINT` of Stamnes' DISORT.
    Refer to section 3.7 of the Comprehensive Documentation.
    This function is wrapped by the `_assemble_intensity_and_fluxes` function.

    For every Fourier mode and layer, the source function at a polar angle `mu` follows from the quadrature solution
    through the scattering integral, so it has the same terms as that solution: the exponentials in tau of the
    eigenvectors and of the direct beam and, for mode 0, the powers of tau of the isotropic source. The coefficients
    of these terms at `mu` are the product of the associated Legendre functions at `mu` with the Legendre moments
    of the terms of the quadrature solution. The transfer equation along `mu`,

        mu du/dt = u - (source function), t being the delta-scaled tau,

    is then integrated exactly from the boundary where the path starts, layer by layer. The same equation
    gives the `tau`-derivative and an antiderivative of `u` from `u` and the source function.
    The largest arrays, which have an entry for each Fourier mode, layer, `mu` and exponential, are work arrays
    which are reused from call to call, and the most recent result is cached.

    The eigenvector matrix consists of two blocks which swap between the upward and the downward quadrature nodes.
    The Legendre moments of the eigenvectors of positive eigenvalues are therefore those of the eigenvectors of
    negative eigenvalues with the sign of the odd degrees flipped, so only the latter are stored, with the
    even and odd degrees apart: the coefficients at `mu` of the two kinds are `even + odd` and `even - odd`.

    Notable internal variables of _build_IotS
    |     Variable      |              Type / Shape              |
    | ----------------- | -------------------------------------- |
    | `rates`           | `NFourier x NLayers x Nexp`            | Rate of each exponential term: the eigenvalues, then that of the direct beam
    | `refs`            | `NLayers x Nexp`                       | Delta-scaled optical depth at which each exponential term equals 1
    | `amplitudes`      | `NFourier x NLayers x Nexp`            | Coefficient of each exponential term: `C_collect`, then 1 for the direct beam
    | `moments_even`    | `Nm_driven x NLayers x NLeg/2 x NQuad/2` | Legendre moments of even offset (degree minus mode) of the eigenvectors
    | `moments_odd`     | `Nm_driven x NLayers x NLeg/2 x NQuad/2` | Same but of odd offset
    | `beam_moments`    | `Nm_driven x NLayers x NLeg`           | Legendre moments of the direct beam term
    | `poly_moments`    | `NLayers x NLeg x Nscoeffs`            | Legendre moments of the polynomial terms (mode 0)
    | `exp_boundaries`  | `NFourier x NLayers x 2 x Nexp`        | Each exponential term, times its amplitude, at the top and bottom of its layer
    | `coeffs`          | `Nm_src x Nlayers x Nmu x Nexp`        | Source function coefficients at `mu`, before `amplitudes`, of the exponentials
    | `integrals_whole` | `Nm_src x Nwhole x Nmu`                | Integral of the source function over each layer which is crossed whole
    | `integrals_last`  | `Nm x Ntau x Nmu`                      | Integral of the source function over the last segment of each path
    """
    st0 = scaled_tau_arr_with_0
    Nexp = NQuad + there_is_beam_source
    # The exponentials of the eigenvectors equal 1 at the top of their layer if their eigenvalue is negative, else at its bottom.
    # The direct beam is one more exponential, which equals 1 at the top of the layer; the transmittance of
    # the beam down to there is instead part of `beam_moments`.
    rates, amplitudes = K_collect, C_collect
    refs = np.concatenate([np.tile(st0[:-1, None], N), np.tile(st0[1:, None], N)], axis=1)
    if there_is_beam_source:
        rates = np.concatenate([rates, np.broadcast_to(-1 / scaled_mu0_arr[None, :, None], (NFourier, NLayers, 1))], axis=2)
        amplitudes = np.concatenate([amplitudes, np.ones((NFourier, NLayers, 1))], axis=2)
        refs = np.concatenate([refs, st0[:-1, None]], axis=1)
    # Fourier modes >= 1 vanish identically unless the direct beam or the BCs drive them
    Nm_driven = NFourier if there_is_beam_source or np.any(C_collect[1:]) else 1
    # Consecutive Fourier modes are multiplied together; these groups are short as mode m only has NLeg - m degrees
    Ngroup = max(1, NLeg // 8)
    # Each exponential term, times its amplitude, at the top and at the bottom of its layer; (NFourier, NLayers, 2, Nexp)
    exp_boundaries = amplitudes[:, :, None, :] * np.exp(
        rates[:, :, None, :] * (np.stack([st0[:-1], st0[1:]], axis=1)[:, :, None] - refs[:, None, :])
    )

    # The largest arrays are work arrays which later calls reuse, as allocating such arrays anew is slow
    _work = {}

    def _get_work_arrays(shape):
        """Three work arrays and a mask of the given shape, which are reused by every call with that shape."""
        if shape not in _work:
            if len(_work) >= 8:
                _work.clear()
            _work[shape] = (np.empty(shape), np.empty(shape), np.empty(shape), np.empty(shape, dtype=bool))
        return _work[shape]

    # Legendre moments of the terms of the quadrature solution, which are independent of the evaluation points.
    # Their product with the associated Legendre functions at `mu` gives the coefficients of the terms
    # of the source function at `mu`. Row r of the moments of Fourier mode m has degree m + r, and
    # the rows beyond degree NLeg - 1 vanish because the associated Legendre table does.
    # --------------------------------------------------------------------------------------------------------------------------
    degrees = np.minimum(np.arange(Nm_driven)[:, None] + np.arange(NLeg)[None, :], NLeg - 1)    # (Nm_driven, NLeg)
    # Coefficients of the scattering kernel as in `_solve_for_gen_and_part_sols`, which neglects
    # the scattering of the (mode, layer) pairs that fail the test below (its no-scattering shortcut)
    c = ((scaled_omega_arr / 2)[:, None] * weighted_scaled_Leg_coeffs)[:, degrees].transpose(1, 0, 2)
    c = c * np.any(np.abs(c) > 1e-8, axis=2, keepdims=True)                             # (Nm_driven, NLayers, NLeg)
    # The associated Legendre functions at the downward quadrature nodes are those at the upward nodes,
    # negated for the odd rows
    asso_leg = asso_leg_table[:Nm_driven, None, :, :]                                   # (Nm_driven, 1, NLeg, N or N + 1)
    c_asso_leg_W = c[:, :, :, None] * (asso_leg[:, :, :, :N] * W)                       # (Nm_driven, NLayers, NLeg, N)
    signs = (-1.0) ** np.arange(NLeg)

    G_same, G_swap = G_collect[:Nm_driven, :, :N, :N], G_collect[:Nm_driven, :, N:, :N]
    moments_even = c_asso_leg_W[:, :, 0::2, :] @ (G_same + G_swap)
    moments_odd = c_asso_leg_W[:, :, 1::2, :] @ (G_same - G_swap)

    # The direct beam and the isotropic source are themselves terms of the source function
    if there_is_beam_source:
        beam_moments = np.exp(-scaled_slant_tau_arr_with_0[:-1, None]) * (
            (c_asso_leg_W @ B_collect[:Nm_driven, :, :N, None])[:, :, :, 0]
            + signs * (c_asso_leg_W @ B_collect[:Nm_driven, :, N:, None])[:, :, :, 0]
            + 2 * I0_div_4pi * (2 - (np.arange(Nm_driven) == 0))[:, None, None] * c * asso_leg[:, :, :, N]
        )
    if there_is_iso_source:
        poly_moments = c_asso_leg_W[0] @ mathscr_v_coeffs[:, :N, :] + signs[:, None] * (
            c_asso_leg_W[0] @ mathscr_v_coeffs[:, N:, :]
        )
        poly_moments[:, 0, :] += scaled_s_poly_coeffs

    # Intensity at the boundary where the paths start
    # --------------------------------------------------------------------------------------------------------------------------
    def _boundary_modes(mu, upward, Nm):
        """The first `Nm` Fourier modes of the intensity which enters the atmosphere along the cosines `mu` (Nmu,)
        of one hemisphere: the Dirichlet BC plus, at the bottom boundary, the reflection by the BDRF of the
        downward intensity and of the direct beam. Returns an array of shape (Nm, Nmu)."""
        b, b_is_scalar = (b_pos, b_pos_is_scalar) if upward else (b_neg, b_neg_is_scalar)
        modes = np.zeros((Nm, len(mu)))
        if b_is_scalar:
            modes[0] = b
        else:
            # Vector BCs are only known at the quadrature nodes, so they are interpolated polynomially
            b = np.reshape(b, (N, -1))[:, :Nm]                                          # (N, 1) or (N, Nm)
            modes[: np.shape(b)[1]] = sc.interpolate.BarycentricInterpolator(mu_arr_pos, b)(np.abs(mu)).T
        if upward:
            for m in range(min(Nm, NBDRF)):
                BDRF = BDRF_Fourier_modes[m]
                at_nodes = BDRF if np.isscalar(BDRF) else BDRF(mu, mu_arr_pos)
                modes[m] += (1 + (m == 0)) * (at_nodes * (mu_arr_pos * W)[None, :]) @ um_bottom_neg[m]
                if there_is_beam_source:
                    at_mu0 = BDRF if np.isscalar(BDRF) else BDRF(mu, np.array([mu0]))[:, 0]
                    modes[m] += 4 * mu0 * I0_div_4pi * at_mu0 * np.exp(-scaled_slant_tau_arr_with_0[-1])
        return modes

    # Fourier modes of the intensity at (`mu`, `tau`)
    # --------------------------------------------------------------------------------------------------------------------------
    def _segment_integrals(coeffs, rate, exp_start, exp_end, path, attenuation, mu):
        """Integral over path segments of the exponential terms of the source function, attenuated along the path
        to the end of each segment. The integral of each exponential is
        (its value at the end - its attenuated value at the start) / (1 - mu * rate),
        which is evaluated in work arrays.

        Arguments of _segment_integrals
        |         Variable          |           Shape             |
        | ------------------------- | --------------------------- |
        | `coeffs`                  | `Nm_src x Nseg x Nmu x Nexp` | Coefficients of the exponentials in the source function, before `amplitudes`
        | `rate`                    | `Nm_src x Nseg x Nexp`      |
        | `exp_start`, `exp_end`    | `Nm_src x Nseg x Nexp`      | Each exponential, times its amplitude, at the start and end of the segment
        | `path`, `attenuation`     | `Nseg x Nmu`                | Optical length of the segment along `mu` and the attenuation along it

        Returns an array of shape `Nm_src x Nseg x Nmu`.
        """
        _, denom, exp_integrals, near_singularity = _get_work_arrays(np.shape(coeffs))
        np.multiply(mu[:, None], rate[:, :, None, :], out=denom)
        np.subtract(1, denom, out=denom)
        np.less(np.abs(denom, out=exp_integrals), 0.01, out=near_singularity)
        np.multiply(attenuation[:, :, None], exp_start[:, :, None, :], out=exp_integrals)
        np.subtract(exp_end[:, :, None, :], exp_integrals, out=exp_integrals)
        # Near the removable singularity the integral is evaluated through phi1, which is exact at the singularity.
        # Its argument `x` is the rate, per unit `path`, at which the integrand decays back along the path;
        # if `x` is negative the integrand is largest at the start of the segment and is referenced to there.
        i, s, a, k = np.nonzero(near_singularity)
        x = denom[i, s, a, k] * path[s, a]
        denom[i, s, a, k] = 1
        exp_integrals /= denom
        exp_integrals[i, s, a, k] = (path[s, a] * _phi1(np.abs(x))) * np.where(
            x >= 0, exp_end[i, s, k], attenuation[s, a] * exp_start[i, s, k]
        )
        return np.einsum("isak, isak -> isa", coeffs, exp_integrals)

    def _hemisphere_modes(mu, tau, tau_derivative_order, Nm):
        """`tau`-derivative of order `tau_derivative_order` of the first `Nm` Fourier modes of the intensity at the
        cosines `mu` (Nmu,) of one hemisphere and the optical depths `tau`; (Nm, Nmu, Ntau)."""
        upward = mu[0] > 0
        l, t = process_tau(tau)
        Ntau, Nmu = len(tau), len(mu)
        abs_mu = np.abs(mu)

        # A path along `mu` to `tau` starts at the bottom boundary if upward, else at the top boundary, crosses whole
        # layers and ends with a last segment within the layer of `tau`. `layers` are those which the paths cross
        # and `whole` those of them which some path crosses whole, i.e. all but the first if upward, else the last.
        start, end = (1, 0) if upward else (0, 1)    # Offsets of the boundaries of a layer where paths enter and exit it
        layers = slice(np.min(l), NLayers) if upward else slice(0, np.max(l) + 1)
        whole = slice(layers.start + 1, layers.stop) if upward else slice(layers.start, layers.stop - 1)
        whole_among_layers = slice(1, None) if upward else slice(0, -1)
        last_among_layers = l - layers.start
        # Delta-scaled optical depths at the start and end of the path across each whole layer and at the start of
        # each last segment, the optical length of each along `mu` and the attenuation along it
        t_whole_start = st0[whole.start + start : whole.stop + start]                   # (Nwhole,)
        t_whole_end = st0[whole.start + end : whole.stop + end]
        t_last_start = st0[l + start]                                                   # (Ntau,)
        path_whole = np.abs(t_whole_end - t_whole_start)[:, None] / abs_mu[None, :]     # (Nwhole, Nmu)
        path_last = np.abs(t - t_last_start)[:, None] / abs_mu[None, :]                 # (Ntau, Nmu)
        attenuation_whole, attenuation_last = np.exp(-path_whole), np.exp(-path_last)

        # Coefficients at `mu` of the exponential terms of the source function in each of `layers`, before `amplitudes`.
        # The Fourier modes >= 1 of the source function vanish at mu = +-1.
        Nm_src = min(Nm, Nm_driven) if np.any(abs_mu < 1) else 1
        asso_leg = _asso_leg_table_by_offset(mu, Nm_src, NLeg)                          # (NLeg, Nm_src, Nmu)
        asso_leg_even = np.ascontiguousarray(asso_leg[0::2].transpose(1, 2, 0))         # (Nm_src, Nmu, NLeg/2)
        asso_leg_odd = np.ascontiguousarray(asso_leg[1::2].transpose(1, 2, 0))
        coeffs = _get_work_arrays((Nm_src, layers.stop - layers.start, Nmu, Nexp))[0]
        for m in range(0, Nm_src, Ngroup):
            group = slice(m, min(m + Ngroup, Nm_src))
            Neven, Nodd = (NLeg - m + 1) // 2, (NLeg - m) // 2
            even = asso_leg_even[group, None, :, :Neven] @ moments_even[group, layers, :Neven, :]
            odd = asso_leg_odd[group, None, :, :Nodd] @ moments_odd[group, layers, :Nodd, :]
            np.add(even, odd, out=coeffs[group, :, :, :N])
            np.subtract(even, odd, out=coeffs[group, :, :, N:NQuad])
            if there_is_beam_source:
                coeffs[group, :, :, NQuad] = (
                    beam_moments[group, layers, : NLeg - m] @ asso_leg[: NLeg - m, group, :].transpose(1, 0, 2)
                )
        coeffs_last = coeffs[:, last_among_layers, :, :]                                # (Nm_src, Ntau, Nmu, Nexp)

        # Integrals of the exponential terms of the source function over the whole layers and over the last segments
        integrals_whole = _segment_integrals(
            coeffs[:, whole_among_layers, :, :],
            rates[:Nm_src, whole, :],
            exp_boundaries[:Nm_src, whole, start, :],
            exp_boundaries[:Nm_src, whole, end, :],
            path_whole, attenuation_whole, mu,
        )
        rate_last = rates[:Nm_src, l, :]                                                # (Nm_src, Ntau, Nexp)
        exp_last = amplitudes[:Nm_src, l, :] * np.exp(rate_last * (t[:, None] - refs[l, :]))
        integrals_last = np.zeros((Nm, Ntau, Nmu))
        integrals_last[:Nm_src] = _segment_integrals(
            coeffs_last, rate_last, exp_boundaries[:Nm_src, l, start, :], exp_last, path_last, attenuation_last, mu
        )
        # Source function at each `tau` (its antiderivative if `tau_derivative_order` is -1)
        sources = np.zeros((Nm, Ntau, Nmu))
        if tau_derivative_order != 0:
            sources[:Nm_src] = np.einsum(
                "itak, itk -> ita", coeffs_last, exp_last * rate_last ** min(tau_derivative_order, 0)
            )

        # The polynomial terms of the source function (mode 0); the constant of their antiderivative is such that
        # the antiderivative of the intensity is that of the quadrature solution at the quadrature nodes
        if there_is_iso_source:
            poly = asso_leg[:, 0, :].T @ poly_moments[layers]                           # (Nlayers, Nmu, Nscoeffs)
            poly_whole, poly_last = poly[whole_among_layers], poly[last_among_layers]
            integrals_whole[0] += _poly_particular_solution(poly_whole, mu, t_whole_end) - attenuation_whole * (
                _poly_particular_solution(poly_whole, mu, t_whole_start)
            )
            integrals_last[0] += _poly_particular_solution(poly_last, mu, t) - attenuation_last * (
                _poly_particular_solution(poly_last, mu, t_last_start)
            )
            if tau_derivative_order == 1:
                sources[0] += _poly_tau_derivative(poly_last, t, 0)
            elif tau_derivative_order == -1:
                sources[0] += _poly_tau_derivative(poly_last, t, -1) - mu * (
                    _poly_particular_solution(poly_last, mu, np.zeros(Ntau))
                )

        # Intensity at the start of the last segment of each path: the intensity which enters the atmosphere and
        # the integrals over the whole layers which the path crosses, each attenuated along the path from there.
        # `from_whole` is the distance in delta-scaled optical depth from the end of each whole layer to the start of
        # each last segment, and is negative if the path to that last segment does not cross that layer.
        from_boundary = np.abs(t_last_start - st0[-1 if upward else 0])                 # (Ntau,)
        from_whole = (t_last_start[:, None] - t_whole_end[None, :]) * (-1 if upward else 1)     # (Ntau, Nwhole)
        attenuation_from_whole = np.where(
            from_whole[:, :, None] >= 0, np.exp(-np.abs(from_whole)[:, :, None] / abs_mu[None, None, :]), 0
        )
        um = _boundary_modes(mu, upward, Nm)[:, None, :] * np.exp(-from_boundary[:, None] / abs_mu[None, :])
        um[:Nm_src] += np.einsum("mna, tna -> mta", integrals_whole, attenuation_from_whole)
        um = um * attenuation_last + integrals_last                                     # (Nm, Ntau, Nmu)

        # The transfer equation along `mu` gives the derivative and an antiderivative with respect to the delta-scaled tau
        if tau_derivative_order == 1:
            um = (um - sources) / mu
        elif tau_derivative_order == -1:
            um = mu * um + sources
        return (um * scale_tau[l][:, None] ** tau_derivative_order).transpose(0, 2, 1)

    # Value-keyed cache for the most recent evaluation points, so that calls which only differ in `phi` recompute nothing
    _cache = {}

    def _um(mu, tau, tau_derivative_order, Nm):
        """`tau`-derivative of order `tau_derivative_order` of the first `Nm` Fourier modes of the (rescaled)
        intensity at the polar angle cosines `mu` and the optical depths `tau`; (Nm, Nmu, Ntau)."""
        _check_tau_derivative_order(tau_derivative_order)
        if not np.all(np.abs(mu) <= 1):
            raise ValueError("mu values must be between -1 and 1.")
        if tau_derivative_order == 1 and np.any(np.abs(mu) < 1e-8):
            raise ValueError("The `tau`-derivative requires `mu` values bounded away from 0 (|mu| >= 1e-8).")
        if not (
            _cache
            and _cache["key"] == (tau_derivative_order, Nm)
            and np.array_equal(mu, _cache["mu"])
            and np.array_equal(tau, _cache["tau"])
        ):
            # `mu = 0` is evaluated as the limit from below, where the intensity is the source function,
            # except at the top boundary where it is the boundary condition
            mu_floored = np.where(np.abs(mu) < 1e-290, -1e-290, mu)
            um = np.empty((Nm, len(mu), len(tau)))
            for upward in (True, False):
                hemisphere = (mu_floored > 0) == upward
                if np.any(hemisphere):
                    um[:, hemisphere, :] = _hemisphere_modes(mu_floored[hemisphere], tau, tau_derivative_order, Nm)
            _cache.update(key=(tau_derivative_order, Nm), mu=np.array(mu, copy=True), tau=np.array(tau, copy=True), um=um)
        return _cache["um"]

    # The two intensity functions
    # --------------------------------------------------------------------------------------------------------------------------
    def u0(mu, tau, tau_derivative_order=0):
        """
        Zeroth Fourier mode of the intensity with arguments `(mu, tau)` each of type array or float.
        Returns an ndarray with axes corresponding to variation with `mu` and `tau` respectively.
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        """
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        return rescale_factor * np.squeeze(_um(mu, tau, tau_derivative_order, 1)[0])[()]

    def u(mu, tau, phi, tau_derivative_order=0, return_Fourier_error=False):
        """
        Intensity function with arguments `(mu, tau, phi)` each of type array or float.
        Returns an ndarray with axes corresponding to variation with `mu, tau, phi` respectively.
        Pass `tau_derivative_order = 1` or `-1` (defaults to `0`)
        to switch to the derivative or an antiderivative of this function with respect to `tau`.
        Pass `return_Fourier_error = True` (defaults to `False`) to return the
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
        """
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        phi = np.atleast_1d(phi)
        um = _um(mu, tau, tau_derivative_order, NFourier)
        cosines = np.cos(np.arange(NFourier)[:, None] * (phi0 - phi)[None, :])
        u = (um.reshape(NFourier, -1).T @ cosines).reshape(len(mu), len(tau), len(phi))

        if return_Fourier_error:
            return rescale_factor * np.squeeze(u)[()], _Fourier_error(um[-1], cosines[-1], u)
        else:
            return rescale_factor * np.squeeze(u)[()]

    return u0, u
