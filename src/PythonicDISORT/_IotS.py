from PythonicDISORT._solve_for_gen_and_part_sols import _seminormalized_asso_leg_table

import numpy as np
import scipy as sc
from math import factorial, log

_LN2 = log(2.0)
# `mu = 0` is evaluated as the limit from below (the intensity there is the source function, except at the top
# boundary where it is the boundary condition), which every kernel below attains if |mu| is floored at this value
_MU_FLOOR = 1e-290
# Fourier modes are processed in blocks whose largest arrays have about this many entries
_BLOCK_SIZE = 2**21


def _phi1_neg(y):
    """phi1(-y) = (1 - exp(-y)) / y for an array y >= 0, equal to 1 at y = 0."""
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(-np.expm1(-y) / y)
    out[np.asarray(y) == 0] = 1.0
    return out


def _dd(a, b, d):
    """(exp(-a d) - exp(-b d)) / (b - a) for d >= 0, evaluated without cancellation, overflow
    or a singularity at a = b by factoring out the larger of the two exponentials."""
    return np.exp(-np.minimum(a, b) * d) * d * _phi1_neg(np.abs(b - a) * d)


def _iso_part_sol_coeffs(Nscoeffs, s_poly_coeffs, G, K, G_inv_mu_inv):
    """Coefficients, in ascending powers of the delta-scaled tau, of the particular solution for the
    isotropic internal source at the quadrature nodes; the same computation as the general branch of
    `subroutines._mathscr_v`. Returns an array of shape (NLayers, NQuad, Nscoeffs)."""
    K_inv = 1 / K
    K_invP = np.cumprod(np.broadcast_to(K_inv[:, :, None], K_inv.shape + (Nscoeffs,)), axis=-1)
    ii = np.arange(Nscoeffs, dtype=int)[:, None]
    pp = np.arange(Nscoeffs, dtype=int)[None, :]
    jj = ii - pp
    fac = np.empty(Nscoeffs)
    fac[0] = 1
    if Nscoeffs > 1:
        fac[1:] = np.cumprod(np.arange(1, Nscoeffs))
    fac_rev = fac[::-1]
    weighted_mathscr_a = s_poly_coeffs[:, ::-1] * fac_rev[None, :]
    lower_tri = np.where((jj >= 0)[None, :, :], np.take(weighted_mathscr_a, jj, axis=1), 0)
    mathscr_b = G @ (
        (K_invP @ np.swapaxes(lower_tri, 1, 2))
        * (1 / fac_rev)[None, None, :]
        * G_inv_mu_inv[:, :, None]
    )
    return mathscr_b[:, :, ::-1]


def _angle_tables(x, a):
    """Tables for `_kernels` which depend on the path lengths `x` (P,) and the absolute direction cosines `a` (Na,)
    but not on the eigenvalues. Returns exp(-x / a) followed by the tuple that `_kernels` takes; all have shape (P, Na)."""
    y = x[:, None] / a[None, :]
    Ea = np.exp(-y)
    Ea1 = np.expm1(-y)
    big = y >= _LN2
    return Ea, (Ea1, np.where(big, Ea, Ea1), big, 0.1 / np.maximum(y, 1.0))


def _kernels(kappa, EK, EK1, x, a, angle_tables):
    """Integrals along a direction, over a path of optical length `x` within one layer, of the exponentials of the
    quadrature solution with eigenvalue magnitudes `kappa` (..., N), for the absolute direction cosines `a` (Na,).
    `x` and the `angle_tables` (see `_angle_tables`) broadcast against `kappa.shape[:-1]` and
    `kappa.shape[:-1] + (Na,)` respectively. Returns two arrays of shape `kappa.shape[:-1] + (Na, N)`:

    - (1 - exp(-(kappa + 1/a) x)) / (1 + a kappa) for the exponentials that decay along the path,
    - (exp(-kappa x) - exp(-x / a)) / (1 - a kappa) for those that grow along it.

    Only the tables `EK` = exp(-kappa x), `EK1` = expm1(-kappa x) and `angle_tables` require exponentials. They are
    combined in whichever of the `exp` and `expm1` forms does not cancel, and the few entries near the removable
    singularity at a kappa = 1 are recomputed through phi1.
    """
    Ea1, Ea_sel, big, thr = angle_tables
    aK = a[:, None] * kappa[..., None, :]
    decaying = EK[..., None, :] * (-Ea1)[..., :, None]
    decaying -= EK1[..., None, :]
    decaying /= aK + 1
    if big.all():
        growing = EK[..., None, :] - Ea_sel[..., :, None]
    elif not big.any():
        growing = EK1[..., None, :] - Ea_sel[..., :, None]
    else:
        growing = np.where(big[..., :, None], EK[..., None, :], EK1[..., None, :])
        growing -= Ea_sel[..., :, None]
    np.subtract(1, aK, out=aK)
    near_singularity = np.abs(aK) < thr[..., :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        growing /= aK
    if near_singularity.any():
        idx = np.nonzero(near_singularity)
        lead, ia, ik = idx[:-2], idx[-2], idx[-1]
        inv_a = 1 / a[ia]
        x_near = np.broadcast_to(x, kappa.shape[:-1])[lead]
        growing[idx] = inv_a * _dd(kappa[lead + (ik,)], inv_a, x_near)
    return decaying, growing


def _build_IotS(
    asso_leg_table,                     # Seminormalized associated Legendre table at `mu_arr_pos` (and `-mu0`) from the solver
    G_collect_0,                        # Eigenvector matrices of the 0th Fourier mode
    K_collect,                          # Eigenvalues
    B_collect,                          # Coefficient vectors of the direct beam particular solutions
    GC_collect,                         # Eigenvectors multiplied by their coefficients
    G_inv_mu_inv,                       # Inverse of the 0th mode eigenvector matrices @ (1 / `mu_arr`)
    tau_arr,                            # Lower boundary of layers
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
    b_pos, b_neg,                       # Dirichlet BCs (rescaled)
    b_pos_is_scalar, b_neg_is_scalar,   # Is each Dirichlet BC scalar and so isotropic?
    Nscoeffs,                           # Number of isotropic source polynomial coefficients
    scaled_s_poly_coeffs,               # Delta-scaled polynomial coefficients of isotropic source (rescaled)
    there_is_iso_source,                # Is there an isotropic source?
):
    """Builds the intensity functions at arbitrary polar angles for `subroutines.interpolate`, for `u` (all Fourier
    modes) and `u0` (the 0th Fourier mode), by the integration of the source function (IotS) of Stamnes (1982), as in
    the subroutines `TERPEV`, `TERPSO` and `USRINT` of Stamnes' DISORT. For every Fourier mode and layer the source
    function at a polar angle `mu` follows from the quadrature solution through the scattering integral, which is a sum
    of the same exponentials in tau as the solution itself. The transfer equation along `mu` is then integrated exactly.
    Refer to section 3.7 of the Comprehensive Documentation.
    Nothing is computed until the returned functions are first called.

    Notable internal variables of _build_IotS
    |     Variable      |              Type / Shape              |
    | ----------------- | -------------------------------------- |
    | `S`               | `B x nl x Nmu x NQuad`                 | Source function coefficients of a block of B Fourier modes in nl layers
    | `Z`               | `B x nl x Nmu`                         | Same for the direct beam exponential
    | `Jc`              | `nl x Nmu x Nscoeffs`                  | Same for the powers of tau of the isotropic source (0th mode)
    | `contrib`         | `Nm x NLayers x Nmu`                   | Integrals of the source function over each whole layer
    | `U`               | `Nm x (NLayers + 1) x Nup`             | Upward intensity at each boundary
    | `V`               | `Nm x (NLayers + 1) x Ndown`           | Downward intensity at each boundary
    """
    st0 = scaled_tau_arr_with_0
    thickness = np.diff(st0)                                                       # (NLayers,)
    is_delta_scaled = np.any(scale_tau != 1)
    all_layers = np.arange(NLayers)
    # Fourier modes >= 1 are identically zero unless the direct beam or azimuthally dependent BCs drive them
    Nm_driven = NFourier if (there_is_beam_source or np.ndim(b_pos) == 2 or np.ndim(b_neg) == 2) else 1

    # Quantities that are independent of the evaluation points, computed at most once
    # --------------------------------------------------------------------------------------------------------------------------
    _statics = {}

    def _solve_statics(Nm):
        """Statics for the first `Nm` Fourier modes; they are rebuilt for all modes if more are needed later."""
        if _statics and _statics["Nm"] >= Nm:
            return _statics
        Nm = Nm if not _statics else NFourier
        s = _statics
        s["Nm"] = Nm
        # Coefficients of the scattering kernel; the factor 1/2 and the Fourier normalization are as in
        # `_solve_for_gen_and_part_sols`
        c = (scaled_omega_arr / 2)[:, None] * weighted_scaled_Leg_coeffs           # (NLayers, NLeg)
        s["c"] = c
        # (mode, layer) pairs for which `_solve_for_gen_and_part_sols` did not take the no-scattering shortcut;
        # the source function of the others has neither a scattering nor a direct beam term
        c_tail_max = np.maximum.accumulate(np.abs(c)[:, ::-1], axis=1)[:, ::-1]    # max over l >= m
        s["ns"] = (c_tail_max[:, :Nm] > 1e-8).T                                    # (Nm, NLayers)
        # Associated Legendre functions at the quadrature nodes, multiplied by the quadrature weights. Those at the
        # downward nodes follow from the parity (-1)^(l - m) about mu = 0.
        table = asso_leg_table[:Nm]                                                # (Nm, NLeg, N or N + 1)
        PW_pos = table[:, :, :N] * W[None, None, :]                                # (Nm, NLeg, N)
        parity = np.where((np.arange(NLeg)[None, :] - np.arange(Nm)[:, None]) % 2 == 0, 1.0, -1.0)
        s["PW_pos"] = PW_pos
        s["PW"] = np.concatenate([PW_pos, PW_pos * parity[:, :, None]], axis=2)    # (Nm, NLeg, NQuad)
        if there_is_beam_source:
            fac = I0_div_4pi * (2 - (np.arange(Nm) == 0))
            # Legendre coefficients of the direct beam source of each mode and layer
            s["Q_coef"] = fac[:, None, None] * (2 * c)[None, :, :] * table[:, None, :, N]   # (Nm, NLayers, NLeg)
        if there_is_iso_source:
            s["vcoef"] = _iso_part_sol_coeffs(
                Nscoeffs, scaled_s_poly_coeffs, G_collect_0, K_collect[0], G_inv_mu_inv
            )                                                                       # (NLayers, NQuad, Nscoeffs)
        return s

    def _bottom_downward_intensity():
        """Fourier modes of the downward intensity at the bottom boundary at the quadrature nodes (NFourier, N),
        which the BDRF reflects."""
        if "u_bot_neg" not in _statics:
            E_bot = np.exp(
                np.concatenate([K_collect[:, -1, :N] * thickness[-1], np.zeros((NFourier, N))], axis=1)
            )
            u_bot_neg = np.einsum("mij, mj -> mi", GC_collect[:, -1, N:, :], E_bot)
            if there_is_beam_source:
                u_bot_neg = u_bot_neg + B_collect[:, -1, N:] * np.exp(-st0[-1] / mu0)
            if there_is_iso_source:
                u_bot_neg[0] += _statics["vcoef"][-1, N:, :] @ (st0[-1] ** np.arange(Nscoeffs))
            _statics["u_bot_neg"] = u_bot_neg
        return _statics["u_bot_neg"]

    def _BC_at_mu(b, b_is_scalar, Nm, abs_mu):
        """Dirichlet BC of each of the first `Nm` Fourier modes at the cosines `abs_mu` (Nm, Na).
        Vector BCs are only known at the quadrature nodes, so they are interpolated polynomially."""
        out = np.zeros((Nm, len(abs_mu)))
        if np.all(b == 0) or len(abs_mu) == 0:
            return out
        if b_is_scalar:
            out[0] = b
        else:
            b2 = np.reshape(b, (N, -1))                                             # (N, 1) or (N, NFourier)
            k = min(Nm, b2.shape[1])
            vals = sc.interpolate.BarycentricInterpolator(mu_arr_pos, b2[:, :k])(abs_mu)
            out[:k] = np.reshape(vals, (len(abs_mu), k)).T
        return out

    # Coefficients of the source function at the polar angles
    # --------------------------------------------------------------------------------------------------------------------------
    def _mode_blocks(Nm, Na, nl):
        """Blocks of Fourier modes `m0, m0 + 2, ...` below `Nm`, as slices. The modes of a block have the same parity,
        so the even and odd terms of the Legendre series are the same rows for all of them. The Legendre series of
        mode m starts at l = m, but those of a block start at l = m0, so the blocks of the costlier many-angle
        computation are kept short to limit the products of zeros."""
        kernel_first = 2 * Na < NLeg
        row_size = nl * NQuad * (Na if kernel_first else max(Na, NLeg))
        B = max(1, _BLOCK_SIZE // row_size)
        if not kernel_first:
            B = min(B, max(1, NLeg // 16))
        return kernel_first, [slice(m0, min(Nm, m0 + 2 * B), 2) for parity in (0, 1) for m0 in range(parity, Nm, 2 * B)]

    def _source_coeffs(blk, layers, P_user, kernel_first):
        """Coefficients of the source function of the Fourier modes `blk` (a slice of step 2) in `layers` (a slice or
        indices) at the polar angles for which `P_user` (Nm, NLeg, Na) holds the associated Legendre functions:
        `S` (B, nl, Na, NQuad) for the exponentials of the quadrature solution, `Z` (B, nl, Na) for the direct beam
        exponential and `Jc` (nl, Na, Nscoeffs) for the powers of tau of the isotropic source (0th mode, so only if
        `blk` starts at 0). The latter two are `None` if absent."""
        s = _statics
        m0 = blk.start
        # Row-major, since `matmul` is fastest with row-major operands
        P_T = np.ascontiguousarray(np.swapaxes(P_user[blk, m0:, :], 1, 2))[:, None, :, :]  # (B, 1, Na, L); zero for l < m
        c = s["c"][layers, m0:][None, :, None, :]                                  # (1, nl, 1, L)
        GC = GC_collect[blk][:, layers]                                            # (B, nl, NQuad, NQuad)
        Bc = B_collect[blk][:, layers] if there_is_beam_source else None           # (B, nl, NQuad)
        Kern0 = None
        if kernel_first:
            # Few angles: scattering kernel between the angles and the quadrature nodes (with weights) first
            Kern = (P_T * c) @ s["PW"][blk, None, m0:, :]                           # (B, nl, Na, NQuad)
            S = Kern @ GC
            Z = (Kern @ Bc[..., None])[..., 0] if Bc is not None else None
            if m0 == 0:
                Kern0 = Kern[0]
        else:
            # Many angles: Legendre moments of the quadrature solution first, then evaluation at the angles. The moments
            # of the even and odd terms only need the sum and difference of the two hemispheres respectively.
            PW_e = s["PW_pos"][blk, None, m0::2, :]                                # (B, 1, L_even, N)
            PW_o = s["PW_pos"][blk, None, m0 + 1::2, :]
            A_e = P_T[..., 0::2] * c[..., 0::2]                                    # (B, nl, Na, L_even)
            A_o = P_T[..., 1::2] * c[..., 1::2]
            S = A_e @ (PW_e @ (GC[..., :N, :] + GC[..., N:, :]))
            if A_o.shape[-1] > 0:
                S += A_o @ (PW_o @ (GC[..., :N, :] - GC[..., N:, :]))
            Z = None
            if Bc is not None:
                Z = (A_e @ (PW_e @ (Bc[..., :N] + Bc[..., N:])[..., None]))[..., 0]
                if A_o.shape[-1] > 0:
                    Z += (A_o @ (PW_o @ (Bc[..., :N] - Bc[..., N:])[..., None]))[..., 0]
            if m0 == 0 and there_is_iso_source:
                Kern0 = (P_T[0] * c[0]) @ s["PW"][0]
        ns = s["ns"][blk][:, layers]                                               # (B, nl)
        S *= ns[:, :, None, None]
        if Z is not None:
            Z += s["Q_coef"][blk][:, layers, m0:] @ P_user[blk, m0:, :]
            Z *= ns[:, :, None]
        Jc = None
        if m0 == 0 and there_is_iso_source:
            Jc = scaled_s_poly_coeffs[layers][:, None, :] + ns[0][:, None, None] * (Kern0 @ s["vcoef"][layers])
        return S, Z, Jc

    def _store_source_coeffs(cache, layers):
        """Computes the source function coefficients in `layers` (unique indices) for the evaluation within them."""
        Nm_src = cache["Nm_src"]
        n, Nmu = len(layers), len(cache["mu"])
        S_l = np.zeros((Nm_src, n, Nmu, NQuad))
        Z_l = np.zeros((Nm_src, n, Nmu)) if there_is_beam_source else None
        Jc_l = None
        kernel_first, blocks = _mode_blocks(Nm_src, Nmu, n)
        for blk in blocks:
            S, Z, Jc = _source_coeffs(blk, layers, cache["P_user"], kernel_first)
            S_l[blk] = S
            if Z is not None:
                Z_l[blk] = Z
            if Jc is not None:
                Jc_l = Jc
        if there_is_iso_source and Jc_l is None:
            Jc_l = np.repeat(scaled_s_poly_coeffs[layers][:, None, :], Nmu, axis=1)
        cache["layers"], cache["S"], cache["Z"], cache["Jc"] = layers, S_l, Z_l, Jc_l

    # Integrals of the polynomial part of the source function
    # --------------------------------------------------------------------------------------------------------------------------
    def _poly_integrals(upward, l, t, a, Jc_sub):
        """Integrals of the polynomial part of the 0th mode source function along the directions with absolute cosines
        `a` (Na,): from `t` down to the bottom of layer `l` if `upward`, else from the top of layer `l` down to `t`.
        `l` and `t` (n,) are layer indices and delta-scaled optical depths in those layers; `Jc_sub` is (n, Na, Nscoeffs).
        The polynomial is re-expanded in the distance from the boundary where the integral starts, so that its
        particular solution along the direction is a finite sum and its constant term needs only `expm1`.
        Returns an array of shape (n, Na)."""
        P = Nscoeffs
        if upward:
            t_ref = st0[l + 1]
            sigma = -1.0
            x = t_ref - t
        else:
            t_ref = st0[l]
            sigma = 1.0
            x = t - t_ref
        pp = np.arange(P)
        binom = sc.special.comb(pp[:, None], pp[None, :])                           # (p, q)
        pow_pq = pp[:, None] - pp[None, :]
        T = np.where(
            pow_pq[None, :, :] >= 0,
            binom[None, :, :] * t_ref[:, None, None] ** np.maximum(pow_pq, 0)[None, :, :],
            0.0,
        )                                                                           # (n, p, q)
        jt = np.einsum("nap, npq -> naq", Jc_sub, T) * (sigma ** pp)[None, None, :]  # (n, Na, q)
        wt = np.zeros_like(jt)
        for q in range(P):
            for k in range(P - q):
                wt[:, :, q] += (-a[None, :]) ** k * (factorial(q + k) / factorial(q)) * jt[:, :, q + k]
        result = wt[:, :, 0] * -np.expm1(-x[:, None] / a[None, :])
        for q in range(1, P):
            result = result + wt[:, :, q] * (x ** q)[:, None]
        return result

    # Quantities that depend on the polar angles `mu` but not on `tau` or `phi`
    # --------------------------------------------------------------------------------------------------------------------------
    _mu_caches = {}

    def _mu_statics(mu, Nm, layers):
        """Boundary intensities of the first `Nm` Fourier modes at the polar angles `mu`, and the source function
        coefficients in `layers` (unique indices); cached for the most recent `mu` of each `Nm`."""
        cache = _mu_caches.get(Nm)
        if cache is None or not np.array_equal(mu, cache["mu"]):
            cache = _new_mu_cache(mu, Nm, layers)
            _mu_caches[Nm] = cache
        elif not np.all(np.isin(layers, cache["layers"])):
            _store_source_coeffs(cache, np.union1d(layers, cache["layers"]) if len(cache["layers"]) < 8 else layers)
        return cache

    def _new_mu_cache(mu, Nm, layers):
        Nmu = len(mu)
        # Upward angles first, so that each hemisphere is a contiguous block of every array below
        mask_up = mu > 0
        nup = int(np.count_nonzero(mask_up))
        order = np.concatenate([np.flatnonzero(mask_up), np.flatnonzero(~mask_up)])
        mu_o = mu[order]
        a = np.maximum(np.abs(mu_o), _MU_FLOOR)
        # Fourier modes >= 1 of the source function vanish at mu = +-1
        Nm_src = min(Nm, Nm_driven) if np.any(np.abs(mu_o) < 1) else 1
        s = _solve_statics(Nm_src)
        cache = {
            "mu": np.array(mu, copy=True), "order": order, "nup": nup, "a": a, "mu_o": mu_o, "Nm_src": Nm_src,
            "P_user": _seminormalized_asso_leg_table(mu_o, Nm_src, NLeg),          # (Nm_src, NLeg, Nmu)
        }

        # Tables over the whole layers
        Ea, tables = _angle_tables(thickness, a)                                    # (NLayers, Nmu)
        if there_is_beam_source:
            inv_a = 1 / a
            d = thickness[:, None]
            Ib = np.empty_like(Ea)
            Ib[:, :nup] = -np.expm1(-(inv_a[None, :nup] + 1 / mu0) * d) / (1 + a[None, :nup] / mu0)
            Ib[:, nup:] = inv_a[None, nup:] * _dd(1 / mu0, inv_a[None, nup:], d)
            Ib *= np.exp(-st0[:-1] / mu0)[:, None]

        # Integrals of the source function over each whole layer, along every direction, by blocks of Fourier modes.
        # The source function coefficients in `layers` are kept for the evaluation within them.
        contrib = np.zeros((Nm, NLayers, Nmu))
        S_l = np.zeros((Nm_src, len(layers), Nmu, NQuad))
        Z_l = np.zeros((Nm_src, len(layers), Nmu)) if there_is_beam_source else None
        Jc_all = np.repeat(scaled_s_poly_coeffs[:, None, :], Nmu, axis=1) if there_is_iso_source else None
        kernel_first, blocks = _mode_blocks(Nm_src, Nmu, NLayers)
        for blk in blocks:
            if not s["ns"][blk].any():
                continue
            S, Z, Jc = _source_coeffs(blk, slice(None), cache["P_user"], kernel_first)
            S_l[blk] = S[:, layers]
            if Z is not None:
                Z_l[blk] = Z[:, layers]
            if Jc is not None:
                Jc_all = Jc

            kappa = K_collect[blk, :, N:]                                           # (B, NLayers, N)
            exponent = -kappa * thickness[:, None]
            decaying, growing = _kernels(kappa, np.exp(exponent), np.expm1(exponent), thickness, a, tables)
            # Upward: the exponentials of the negative eigenvalues decay along the direction; downward: the others
            vals = np.empty(S.shape[:3])
            vals[:, :, :nup] = (
                np.einsum("bnak, bnak -> bna", S[:, :, :nup, :N], decaying[:, :, :nup])
                + np.einsum("bnak, bnak -> bna", S[:, :, :nup, N:], growing[:, :, :nup])
            )
            vals[:, :, nup:] = (
                np.einsum("bnak, bnak -> bna", S[:, :, nup:, :N], growing[:, :, nup:])
                + np.einsum("bnak, bnak -> bna", S[:, :, nup:, N:], decaying[:, :, nup:])
            )
            if Z is not None:
                vals += Z * Ib[None, :, :]
            contrib[blk] = vals
        if there_is_iso_source:
            contrib[0, :, :nup] += _poly_integrals(True, all_layers, st0[:-1], a[:nup], Jc_all[:, :nup])
            contrib[0, :, nup:] += _poly_integrals(False, all_layers, st0[1:], a[nup:], Jc_all[:, nup:])
        cache["layers"], cache["S"], cache["Z"] = layers, S_l, Z_l
        cache["Jc"] = Jc_all[layers] if there_is_iso_source else None

        # Boundary conditions at `mu`
        mu_up = mu_o[:nup]
        U = np.empty((Nm, NLayers + 1, nup))                                        # Upward intensity at each boundary
        V = np.empty((Nm, NLayers + 1, Nmu - nup))                                  # Downward intensity at each boundary
        V[:, 0, :] = _BC_at_mu(b_neg, b_neg_is_scalar, Nm, np.abs(mu_o[nup:]))
        U[:, -1, :] = _BC_at_mu(b_pos, b_pos_is_scalar, Nm, mu_up)
        if NBDRF > 0 and nup > 0:
            u_bot_neg = _bottom_downward_intensity()
            for m in range(min(Nm, NBDRF)):
                BDRF_m = BDRF_Fourier_modes[m]
                if np.isscalar(BDRF_m):
                    R = (1 + (m == 0)) * BDRF_m * (mu_arr_pos * W)[None, :]
                    X = 4 * mu0 * I0_div_4pi * BDRF_m
                else:
                    R = (1 + (m == 0)) * BDRF_m(mu_up, mu_arr_pos) * (mu_arr_pos * W)[None, :]
                    if there_is_beam_source:
                        X = 4 * mu0 * I0_div_4pi * BDRF_m(mu_up, np.array([mu0]))[:, 0]
                U[m, -1, :] += R @ u_bot_neg[m]
                if there_is_beam_source:
                    U[m, -1, :] += X * np.exp(-st0[-1] / mu0)

        # Propagate through the layers, adding each layer's integral of the source function
        for l in range(NLayers - 1, -1, -1):
            U[:, l, :] = U[:, l + 1, :] * Ea[l, :nup] + contrib[:, l, :nup]
        for l in range(NLayers):
            V[:, l + 1, :] = V[:, l, :] * Ea[l, nup:] + contrib[:, l, nup:]
        cache["U"], cache["V"] = U, V
        return cache

    # Fourier modes of the intensity at (`mu`, `tau`)
    # --------------------------------------------------------------------------------------------------------------------------
    def _IotS_modes(mu, tau, is_antiderivative_wrt_tau, Nm):
        """Returns the first `Nm` Fourier modes of the (rescaled) intensity, or of its tau-antiderivative,
        at the polar angle cosines `mu` and optical depths `tau`, as an array of shape (Nm, Nmu, Ntau)."""
        if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")
        l = np.searchsorted(tau_arr, tau, side="left")
        if is_delta_scaled:
            t = st0[1:][l] - (tau_arr[l] - tau) * scale_tau[l]
        else:
            t = tau
        cache = _mu_statics(mu, Nm, np.unique(l))
        nup, a, Nm_src = cache["nup"], cache["a"], cache["Nm_src"]
        Nmu, Ntau = len(mu), len(tau)

        # Source function coefficients in the layers that contain `tau`
        pos = np.searchsorted(cache["layers"], l)
        S = cache["S"][:, pos]                                                      # (Nm_src, Ntau, Nmu, NQuad)
        if there_is_beam_source:
            Z = cache["Z"][:, pos]                                                  # (Nm_src, Ntau, Nmu)
        if there_is_iso_source:
            Jc = cache["Jc"][pos]                                                   # (Ntau, Nmu, Nscoeffs)

        # Exponentials of the quadrature solution over the distances to the two boundaries of the layer
        kappa = K_collect[:Nm_src, l, N:]                                           # (Nm_src, Ntau, N)
        x_top = t - st0[l]
        x_bot = st0[l + 1] - t
        exponent_top = -kappa * x_top[None, :, None]
        exponent_bot = -kappa * x_bot[None, :, None]
        E_top = np.exp(exponent_top)
        E_bot = np.exp(exponent_bot)

        um_o = np.empty((Nm, Nmu, Ntau))
        for upward, cols in ((True, slice(0, nup)), (False, slice(nup, Nmu))):
            a_h = a[cols]
            Na = len(a_h)
            if Na == 0:
                continue
            # Path within the layer: up from its bottom, or down from its top
            if upward:
                x, E_path, exponent_path, E_other = x_bot, E_bot, exponent_bot, E_top
            else:
                x, E_path, exponent_path, E_other = x_top, E_top, exponent_top, E_bot
            Ea, tables = _angle_tables(x, a_h)                                      # (Ntau, Na)
            decaying, growing = _kernels(kappa, E_path, np.expm1(exponent_path), x, a_h, tables)
            S_h = S[:, :, cols, :]
            # The exponentials which decay along the path are referenced to the other boundary of the layer
            ref = E_other[:, :, None, :]
            if upward:
                vals = (
                    np.einsum("mtak, mtak -> mta", S_h[..., :N] * ref, decaying)
                    + np.einsum("mtak, mtak -> mta", S_h[..., N:], growing)
                )
            else:
                vals = (
                    np.einsum("mtak, mtak -> mta", S_h[..., :N], growing)
                    + np.einsum("mtak, mtak -> mta", S_h[..., N:] * ref, decaying)
                )

            if there_is_beam_source:
                inv_a = 1 / a_h[None, :]
                if upward:
                    Ib = (
                        np.exp(-t / mu0)[:, None]
                        * -np.expm1(-(inv_a + 1 / mu0) * x[:, None]) / (1 + a_h[None, :] / mu0)
                    )
                else:
                    Ib = inv_a * np.exp(-st0[l] / mu0)[:, None] * _dd(1 / mu0, inv_a, x[:, None])
                vals += Z[:, :, cols] * Ib[None, :, :]
            if there_is_iso_source:
                vals[0] += _poly_integrals(upward, l, t, a_h, Jc[:, cols, :])

            # Add the attenuated intensity at the boundary where the path starts
            if upward:
                full = cache["U"][:, l + 1, :] * Ea[None, :, :]                     # (Nm, Ntau, Na)
            else:
                full = cache["V"][:, l, :] * Ea[None, :, :]
            full[:Nm_src] += vals

            if is_antiderivative_wrt_tau:
                # Along any direction, mu du/dt = u - J, hence mu * u + (an antiderivative of J) is an antiderivative
                # of u. The constant of the polynomial part is chosen to match the quadrature-node antiderivatives.
                mu_s = cache["mu_o"][cols]                                          # Signed
                E_over_K = np.concatenate([E_top / -kappa, E_bot / kappa], axis=-1)  # (Nm_src, Ntau, NQuad)
                J_anti = np.zeros((Nm, Ntau, Na))
                J_anti[:Nm_src] = np.einsum("mnak, mnk -> mna", S_h, E_over_K)
                if there_is_beam_source:
                    J_anti[:Nm_src] += Z[:, :, cols] * (-mu0 * np.exp(-t / mu0))[None, :, None]
                if there_is_iso_source:
                    pp = np.arange(Nscoeffs)
                    Jc_h = Jc[:, cols, :]
                    J_anti[0] += np.einsum(
                        "nap, np -> na", Jc_h, t[:, None] ** (pp + 1)[None, :] / (pp + 1)[None, :]
                    )
                    fact = np.array([factorial(k) for k in pp], dtype=float)
                    w0 = np.einsum("nap, ap -> na", Jc_h, mu_s[:, None] ** pp[None, :] * fact[None, :])
                    J_anti[0] -= mu_s[None, :] * w0
                full = (mu_s[None, None, :] * full + J_anti) / scale_tau[l][None, :, None]

            um_o[:, cols, :] = np.swapaxes(full, 1, 2)

        # Return to the original `mu` ordering
        um = np.empty_like(um_o)
        um[:, cache["order"], :] = um_o
        return um

    # The two intensity functions
    # --------------------------------------------------------------------------------------------------------------------------
    def IotS_u(mu, tau, phi, is_antiderivative_wrt_tau=False):
        """Intensity at (`mu`, `tau`, `phi`) by the integration of the source function.
        Returns the unsqueezed array of shape (Nmu, Ntau, Nphi)."""
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        phi = np.atleast_1d(phi)
        um = _IotS_modes(mu, tau, is_antiderivative_wrt_tau, NFourier)
        u = np.einsum(
            "mit, mp -> itp", um, np.cos(np.arange(NFourier)[:, None] * (phi0 - phi)[None, :]), optimize=True
        )
        return rescale_factor * u

    def IotS_u0(mu, tau, is_antiderivative_wrt_tau=False):
        """Zeroth Fourier mode of the intensity at (`mu`, `tau`) by the integration of the source function.
        Returns the unsqueezed array of shape (Nmu, Ntau)."""
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        return rescale_factor * _IotS_modes(mu, tau, is_antiderivative_wrt_tau, 1)[0]

    return IotS_u, IotS_u0
