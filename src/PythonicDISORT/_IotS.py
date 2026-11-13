from PythonicDISORT._solve_for_gen_and_part_sols import _asso_leg_table_by_offset

import numpy as np
import scipy as sc
from math import factorial, log

# REVIEW (merge): this module was rewritten when the interpolation was merged into `pydisort` (`interpolate_mu=True`).
# The method is unchanged (integration of the source function); the changes are for speed:
# - the Legendre moments of the eigenvectors, which do not depend on the evaluation points, are computed once and
#   every new set of `mu` values then needs only their product with the associated Legendre functions at `mu`;
# - only the layers between each `tau` and the boundary where its path starts are integrated, and nothing is
#   integrated within a layer for the `tau` values that are on a layer boundary, e.g. `tau = 0`;
# - the arrays are processed in pieces that fit in the cache of the processor, in work arrays that are reused,
#   with the longer of the angle and eigenvalue axes last.

_LN2 = log(2.0)
# `mu = 0` is evaluated as the limit from below (the intensity there is the source function, except at the top
# boundary where it is the boundary condition), which every kernel below attains if |mu| is floored at this value
_MU_FLOOR = 1e-290
# The largest arrays are processed in pieces of about this many entries, which fit in the cache of the processor
_CHUNK_SIZE = 2**16


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
    """Tables for `_integrate_exponentials` which depend on the path lengths `x` (P,) and the absolute direction
    cosines `a` (Na,) but not on the eigenvalues. Returns exp(-x / a) followed by the tuple that
    `_integrate_exponentials` takes; all have shape (P, Na)."""
    y = x[:, None] / a[None, :]
    Ea = np.exp(-y)
    Ea1 = np.expm1(-y)
    big = y >= _LN2
    return Ea, (-Ea1, np.where(big, Ea, Ea1), big, 0.01 / np.maximum(y, 1.0))


def _integrate_exponentials(
    upward, X_e, X_o, kappa, x, a, angle_tables, C_E_dec, C_E1_dec, C_grow, C_E_grow, C_E1_grow, get_work_arrays
):
    """Integrals along directions, over paths within one layer, of the part of the source function which consists of
    the exponentials of the quadrature solution. The directions have absolute cosines `a` (Na,) and the paths have
    optical lengths `x`; the integrals are evaluated for a stack (...) of Fourier modes and layers or optical depths,
    whose eigenvalue magnitudes are `kappa` (..., N). `x` and the `angle_tables` (see `_angle_tables`) broadcast
    against (...) and (..., Na) respectively. Returns an array of shape (..., Na).

    The source function coefficients are (`X_e` + `X_o`) C for the exponentials with negative eigenvalues and
    (`X_e` - `X_o`) C for those with positive ones, where `X_e` and `X_o` are (..., N, Na) and C are the
    coefficients of the eigenvectors. Along an upward direction the former decay and the latter grow; downward it
    is the opposite. The kernels of the two kinds are respectively

    - (1 - exp(-(kappa + 1/a) x)) / (1 + a kappa),
    - (exp(-kappa x) - exp(-x / a)) / (1 - a kappa),

    which only require the exponentials of the `angle_tables` and exp(-kappa x) and expm1(-kappa x). These two
    arrive multiplied by the coefficients C of the decaying kind as `C_E_dec` and `C_E1_dec`, and by those of the
    growing kind as `C_E_grow` and `C_E1_grow`; `C_grow` are the coefficients themselves. All are (..., N).
    The exponentials are combined in whichever of the `exp` and `expm1` forms does not cancel, and the few
    entries near the removable singularity at a kappa = 1 are recomputed through phi1.

    The arrays over both the eigenvalues and the angles have the longer of those two axes last, which is the
    faster arrangement, and are work arrays of that shape from `get_work_arrays`.
    """
    neg_Ea1, Ea_sel, big, thr = angle_tables
    Na, N = len(a), kappa.shape[-1]
    if Na >= N:
        static = lambda array: array[..., None]                                     # (..., N, 1)
        angular = lambda array: array[..., None, :]                                 # (..., 1, Na)
        subscripts = "...ka, ...ka -> ...a"
        aK, scratch, decaying, growing = get_work_arrays(kappa.shape + (Na,))
    else:
        static = lambda array: array[..., None, :]                                  # (..., 1, N)
        angular = lambda array: array[..., :, None]                                 # (..., Na, 1)
        subscripts = "...ka, ...ak -> ...a"
        aK, scratch, decaying, growing = get_work_arrays(kappa.shape[:-1] + (Na, N))

    np.multiply(static(kappa), angular(a), out=aK)
    np.add(aK, 1, out=scratch)
    np.multiply(static(C_E_dec), angular(neg_Ea1), out=decaying)
    decaying -= static(C_E1_dec)
    decaying /= scratch
    np.multiply(static(C_grow), angular(Ea_sel), out=growing)
    if big.all():
        np.subtract(static(C_E_grow), growing, out=growing)
    elif not big.any():
        np.subtract(static(C_E1_grow), growing, out=growing)
    else:
        np.subtract(np.where(angular(big), static(C_E_grow), static(C_E1_grow)), growing, out=growing)
    np.subtract(1, aK, out=aK)
    np.abs(aK, out=scratch)
    with np.errstate(divide="ignore", invalid="ignore"):
        growing /= aK
    near_singularity = np.nonzero(scratch < angular(thr))
    if len(near_singularity[0]) > 0:
        lead = near_singularity[:-2]
        ik, ia = near_singularity[-2:] if Na >= N else near_singularity[:-3:-1]
        inv_a = 1 / a[ia]
        x_near = np.broadcast_to(x, kappa.shape[:-1])[lead]
        growing[near_singularity] = C_grow[lead + (ik,)] * inv_a * _dd(kappa[lead + (ik,)], inv_a, x_near)

    result = np.einsum(subscripts, X_e, decaying) + np.einsum(subscripts, X_e, growing)
    odd = np.einsum(subscripts, X_o, decaying) - np.einsum(subscripts, X_o, growing)
    return result + odd if upward else result - odd


def _build_IotS(
    asso_leg_table,                     # Seminormalized associated Legendre table at `mu_arr_pos` (and `-mu0`) from the solver
    G_collect,                          # Eigenvector matrices
    K_collect,                          # Eigenvalues
    B_collect,                          # Coefficient vectors of the direct beam particular solutions
    C_collect,                          # Coefficients of the eigenvectors
    G_inv_mu_inv,                       # Inverse of the 0th mode eigenvector matrices @ (1 / `mu_arr`)
    tau_arr,                            # Lower boundary of layers
    scaled_tau_arr_with_0,              # Delta-scaled lower boundary of layers with 0 inserted in front
    scale_tau,                          # Delta-scale factor for tau
    mu_arr_pos, W,                      # Upper hemisphere quadrature nodes and weights
    N, NLeg,                            # Number of 1) upper hemisphere quadrature nodes; 2) phase function Legendre coefficients
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
    only_flux,                          # Only construct `u0`?
):
    """Builds the intensity functions `u0` (the 0th Fourier mode) and `u` (all Fourier modes) at arbitrary polar
    angles, which `pydisort` returns if `interpolate_mu=True`, by the integration of the source function (IotS) of
    Stamnes (1982), as in the subroutines `TERPEV`, `TERPSO` and `USRINT` of Stamnes' DISORT. For every Fourier mode
    and layer the source function at a polar angle `mu` follows from the quadrature solution through the scattering
    integral, which is a sum of the same exponentials in tau as the solution itself. The transfer equation along
    `mu` is then integrated exactly. Refer to section 3.7 of the Comprehensive Documentation.
    This function is wrapped by the `_assemble_intensity_and_fluxes` function.
    Nothing is computed until the returned functions are first called.

    The scattering integral is evaluated through the Legendre series of the phase function. The series is split
    into its terms that are even and odd in `mu`, which only need the sum and difference of the two blocks of the
    eigenvector matrix respectively. The degrees of each Fourier mode are indexed by their offset from the mode,
    as in `asso_leg_table`, so that the even and odd terms are the same rows for every mode.

    Notable internal variables of _build_IotS
    |     Variable      |              Type / Shape              |
    | ----------------- | -------------------------------------- |
    | `T_e`, `T_o`      | `Nm x NLayers x NQuad/2 x NLeg/2`      | Even and odd Legendre moments of the eigenvectors
    | `P_e`, `P_o`      | `Nm x NLeg/2 x Nmu`                    | Even and odd associated Legendre functions at `mu`
    | `X_e`, `X_o`      | `Nm x nl x NQuad/2 x Nmu`              | Source function coefficients (before `C_collect`) in nl layers
    | `Z`               | `Nm x nl x Nmu`                        | Same for the direct beam exponential
    | `Jc`              | `nl x Nmu x Nscoeffs`                  | Same for the powers of tau of the isotropic source (0th mode)
    | `integrals`       | `Nm x nl x Nmu`                        | Integrals of the source function over each whole layer
    """
    st0 = scaled_tau_arr_with_0
    thickness = np.diff(st0)                                                       # (NLayers,)
    is_delta_scaled = np.any(scale_tau != 1)
    # Fourier modes >= 1 are identically zero unless the direct beam or azimuthally dependent BCs drive them
    Nm_driven = NFourier if (there_is_beam_source or np.ndim(b_pos) == 2 or np.ndim(b_neg) == 2) else 1
    R_e, R_o = (NLeg + 1) // 2, NLeg // 2                                           # Number of even and odd terms

    # Quantities that are independent of the evaluation points, computed at most once
    # --------------------------------------------------------------------------------------------------------------------------
    _statics = {}
    _lazy = {}
    _work = {}

    def _get_work_arrays(shape):
        """Four work arrays of the given shape, which are reused by every piece of that shape."""
        if shape not in _work:
            if len(_work) >= 4:
                _work.clear()
            _work[shape] = tuple(np.empty(shape) for _ in range(4))
        return _work[shape]

    def _get_statics(Nm):
        """Statics for the first `Nm` Fourier modes; they are rebuilt for all the driven modes if more are needed later."""
        if _statics and _statics["Nm"] >= Nm:
            return _statics
        Nm = Nm if not _statics else Nm_driven
        s = _statics
        s.clear()
        s["Nm"] = Nm
        # Coefficients of the scattering kernel; the factor 1/2 and the Fourier normalization are as in
        # `_solve_for_gen_and_part_sols`
        c = (scaled_omega_arr / 2)[:, None] * weighted_scaled_Leg_coeffs           # (NLayers, NLeg)
        # (mode, layer) pairs for which `_solve_for_gen_and_part_sols` did not take the no-scattering shortcut;
        # the source function of the others has neither a scattering nor a direct beam term
        c_tail_max = np.maximum.accumulate(np.abs(c)[:, ::-1], axis=1)[:, ::-1]    # max over l >= m
        no_shortcut = (c_tail_max[:, :Nm] > 1e-8).T                                # (Nm, NLayers)
        # The same coefficients indexed by the offset r = l - m of the degree from the mode, as in `asso_leg_table`
        ell = np.arange(Nm)[:, None] + np.arange(NLeg)[None, :]                    # (Nm, NLeg)
        c_by_offset = np.where(
            (ell < NLeg)[:, None, :] & no_shortcut[:, :, None],
            c[:, np.minimum(ell, NLeg - 1)].transpose(1, 0, 2),
            0.0,
        )                                                                           # (Nm, NLayers, NLeg)
        # Associated Legendre functions at the upward quadrature nodes, multiplied by the quadrature weights.
        # Those at the downward nodes are the same for even offsets and the negatives for odd offsets.
        table = asso_leg_table[:Nm]                                                # (Nm, NLeg, N or N + 1)
        PW_T = (table[:, :, :N] * W).transpose(0, 2, 1)                             # (Nm, N, NLeg)
        sign = np.where(np.arange(NLeg) % 2 == 0, 1.0, -1.0)

        # Legendre moments of the eigenvectors. The eigenvector matrix consists of two blocks which swap between the
        # upward and downward nodes, so the even terms need only the sum of the blocks and the odd terms their difference.
        T_e = np.zeros((Nm, NLayers, N, R_e))
        T_o = np.zeros((Nm, NLayers, N, R_o))
        # Mode m has NLeg - m terms, so the modes are processed in short blocks which skip the terms that do not exist
        B = max(1, min(4 * _CHUNK_SIZE // (NLayers * R_e * N), NLeg // 8))
        for m0 in range(0, Nm, B):
            blk = slice(m0, min(m0 + B, Nm))
            R0 = NLeg - m0                                                          # The offsets of these modes are below R0
            G_same_T = G_collect[blk, :, :N, :N].transpose(0, 1, 3, 2)
            G_swap_T = G_collect[blk, :, :N, N:].transpose(0, 1, 3, 2)
            np.matmul(
                np.add(G_same_T, G_swap_T, order="C"),
                np.multiply(PW_T[blk, None, :, 0:R0:2], c_by_offset[blk, :, None, 0:R0:2], order="C"),
                out=T_e[blk, :, :, : (R0 + 1) // 2],
            )
            if R0 > 1:
                np.matmul(
                    np.subtract(G_same_T, G_swap_T, order="C"),
                    np.multiply(PW_T[blk, None, :, 1:R0:2], c_by_offset[blk, :, None, 1:R0:2], order="C"),
                    out=T_o[blk, :, :, : R0 // 2],
                )
        s["T_e"], s["T_o"] = T_e, T_o

        if there_is_beam_source:
            # Legendre moments of the direct beam particular solution plus those of the direct beam source
            fac = I0_div_4pi * (2 - (np.arange(Nm) == 0))
            Zm = c_by_offset * (
                B_collect[:Nm, :, :N] @ PW_T
                + sign * (B_collect[:Nm, :, N:] @ PW_T)
                + 2 * fac[:, None, None] * table[:, None, :, N]
            )                                                                       # (Nm, NLayers, NLeg)
            s["Zm_e"] = np.ascontiguousarray(Zm[:, :, 0::2])
            s["Zm_o"] = np.ascontiguousarray(Zm[:, :, 1::2])
        if there_is_iso_source:
            # Legendre moments of the particular solution for the isotropic source (0th mode)
            vcoef_T = _get_iso_part_sol_coeffs().transpose(0, 2, 1)                 # (NLayers, Nscoeffs, NQuad)
            Jm = c_by_offset[0][:, None, :] * (
                vcoef_T[:, :, :N] @ PW_T[0] + sign * (vcoef_T[:, :, N:] @ PW_T[0])
            )                                                                       # (NLayers, Nscoeffs, NLeg)
            s["Jm_e"] = np.ascontiguousarray(Jm[:, :, 0::2])
            s["Jm_o"] = np.ascontiguousarray(Jm[:, :, 1::2])

        # Exponentials of the quadrature solution over the whole layers, multiplied by the coefficients of the eigenvectors
        kappa = np.ascontiguousarray(K_collect[:Nm, :, N:])                         # (Nm, NLayers, N)
        exponent = -kappa * thickness[:, None]
        EK, EK1 = np.exp(exponent), np.expm1(exponent)
        C_neg, C_pos = C_collect[:Nm, :, :N], C_collect[:Nm, :, N:]
        s["kappa"], s["C_neg"], s["C_pos"] = kappa, np.ascontiguousarray(C_neg), np.ascontiguousarray(C_pos)
        s["whole_layers"] = {
            # In the order of the arguments of `_integrate_exponentials`: upward, the exponentials with negative
            # eigenvalues decay along the path and those with positive ones grow; downward it is the opposite
            True: (C_neg * EK, C_neg * EK1, s["C_pos"], C_pos * EK, C_pos * EK1),
            False: (C_pos * EK, C_pos * EK1, s["C_neg"], C_neg * EK, C_neg * EK1),
        }
        return s

    def _get_iso_part_sol_coeffs():
        if "vcoef" not in _lazy:
            _lazy["vcoef"] = _iso_part_sol_coeffs(
                Nscoeffs, scaled_s_poly_coeffs, G_collect[0], K_collect[0], G_inv_mu_inv
            )
        return _lazy["vcoef"]

    def _bottom_downward_intensity():
        """Fourier modes of the downward intensity at the bottom boundary at the quadrature nodes (NFourier, N),
        which the BDRF reflects."""
        if "u_bot_neg" not in _lazy:
            C_E_bot = C_collect[:, -1, :].copy()
            C_E_bot[:, :N] *= np.exp(K_collect[:, -1, :N] * thickness[-1])
            u_bot_neg = (G_collect[:, -1, N:, :] @ C_E_bot[:, :, None])[:, :, 0]
            if there_is_beam_source:
                u_bot_neg = u_bot_neg + B_collect[:, -1, N:] * np.exp(-st0[-1] / mu0)
            if there_is_iso_source:
                u_bot_neg[0] += _get_iso_part_sol_coeffs()[-1, N:, :] @ (st0[-1] ** np.arange(Nscoeffs))
            _lazy["u_bot_neg"] = u_bot_neg
        return _lazy["u_bot_neg"]

    # Boundary conditions at the polar angles
    # --------------------------------------------------------------------------------------------------------------------------
    def _BC_at_mu(b, b_is_scalar, Nm, abs_mu):
        """Dirichlet BC of each of the first `Nm` Fourier modes at the cosines `abs_mu` (Nm, Na).
        Vector BCs are only known at the quadrature nodes, so they are interpolated polynomially."""
        out = np.zeros((Nm, len(abs_mu)))
        if np.all(b == 0):
            return out
        if b_is_scalar:
            out[0] = b
        else:
            b2 = np.reshape(b, (N, -1))                                             # (N, 1) or (N, NFourier)
            k = min(Nm, b2.shape[1])
            vals = sc.interpolate.BarycentricInterpolator(mu_arr_pos, b2[:, :k])(abs_mu)
            out[:k] = np.reshape(vals, (len(abs_mu), k)).T
        return out

    def _bottom_BC_at_mu(Nm, mu_up):
        """Upward intensity at the bottom boundary of each of the first `Nm` Fourier modes at the cosines `mu_up`
        (Nm, Nup): the Dirichlet BC plus the reflection by the BDRF of the downward intensity and the direct beam."""
        U = _BC_at_mu(b_pos, b_pos_is_scalar, Nm, mu_up)
        if NBDRF > 0:
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
                U[m] += R @ u_bot_neg[m]
                if there_is_beam_source:
                    U[m] += X * np.exp(-st0[-1] / mu0)
        return U

    # Integrals of the direct beam and isotropic source parts of the source function
    # --------------------------------------------------------------------------------------------------------------------------
    def _beam_integrals(upward, x, a):
        """Integrals of the direct beam exponential along the directions with absolute cosines `a` (Na,) over paths
        of optical lengths `x` (n,) within one layer, relative to the exponential at the shallower end of each path.
        Returns an array of shape (n, Na)."""
        if upward:
            return -np.expm1(-(1 / a[None, :] + 1 / mu0) * x[:, None]) / (1 + a[None, :] / mu0)
        inv_a = 1 / a[None, :]
        return inv_a * _dd(1 / mu0, inv_a, x[:, None])

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
            x = np.maximum(t_ref - t, 0)
        else:
            t_ref = st0[l]
            sigma = 1.0
            x = np.maximum(t - t_ref, 0)
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

    # Quantities that depend on the polar angles `mu` but not on `tau` or `phi`, cached for the most recent `mu`
    # --------------------------------------------------------------------------------------------------------------------------
    _mu_caches = {}

    def _get_mu_cache(mu, Nm):
        """Associated Legendre functions at `mu` of the Fourier modes whose source function does not vanish there.
        If `mu` has both upward and downward angles, the upward ones are first, so that each hemisphere is a
        contiguous block."""
        cache = _mu_caches.get(Nm)
        if cache is not None and np.array_equal(mu, cache["mu"]):
            return cache
        mu = np.array(mu, copy=True)
        mask_up = mu > 0
        nup = int(np.count_nonzero(mask_up))
        if nup == 0 or nup == len(mu):
            order, mu_o = None, mu
        else:
            order = np.concatenate([np.flatnonzero(mask_up), np.flatnonzero(~mask_up)])
            mu_o = mu[order]
        a = np.maximum(np.abs(mu_o), _MU_FLOOR)
        # Fourier modes >= 1 of the source function vanish at mu = +-1
        Nm_src = min(Nm, Nm_driven) if np.any(a < 1) else 1
        P = _asso_leg_table_by_offset(a, Nm_src, NLeg)                              # (NLeg, Nm_src, Nmu)
        P_e = np.ascontiguousarray(P[0::2].transpose(1, 0, 2))                      # (Nm_src, R_e, Nmu)
        P_o = np.ascontiguousarray(P[1::2].transpose(1, 0, 2))                      # (Nm_src, R_o, Nmu)
        P_o[:, :, nup:] *= -1                                                       # Odd in mu
        cache = {
            "mu": mu, "order": order, "nup": nup, "a": a, "mu_o": mu_o, "Nm_src": Nm_src,
            "P_e": P_e, "P_o": P_o, "hemispheres": {},
        }
        _mu_caches[Nm] = cache
        return cache

    def _get_hemisphere(cache, upward, Nm):
        """The angles of one hemisphere and the intensity at the boundary where their paths start: the bottom
        boundary for the upward angles and the top boundary for the downward angles."""
        hem = cache["hemispheres"].get(upward)
        if hem is None:
            cols = slice(0, cache["nup"]) if upward else slice(cache["nup"], None)
            a = cache["a"][cols]
            hem = {
                "upward": upward, "a": a, "mu": cache["mu_o"][cols], "Nm_src": cache["Nm_src"],
                "P_e": cache["P_e"][:, :, cols], "P_o": cache["P_o"][:, :, cols],
                "BC": _bottom_BC_at_mu(Nm, a) if upward else _BC_at_mu(b_neg, b_neg_is_scalar, Nm, a),
                "lo": None, "src": np.empty(0, dtype=int),
            }
            cache["hemispheres"][upward] = hem
        return hem

    def _source_coeffs(s, blk, layers, P_e, P_o):
        """Coefficients of the source function, before the coefficients of the eigenvectors, of the Fourier modes `blk`
        (a slice) in `layers` (a slice or indices) at the angles of `P_e` and `P_o`; each is (B, nl, N, Na)."""
        R0 = NLeg - blk.start                                                       # The offsets of these modes are below R0
        Re0, Ro0 = (R0 + 1) // 2, R0 // 2
        X_e = s["T_e"][blk, layers, :, :Re0] @ P_e[blk, None, :Re0]
        X_o = s["T_o"][blk, layers, :, :Ro0] @ P_o[blk, None, :Ro0]
        return X_e, X_o

    def _beam_and_iso_coeffs(s, hem, layers):
        """Coefficients of the source function for the direct beam exponential (Nm_src, nl, Na) and for the powers of
        tau of the isotropic source (nl, Na, Nscoeffs) in `layers` (a slice or indices); `None` if absent."""
        Nm_src, P_e, P_o = hem["Nm_src"], hem["P_e"], hem["P_o"]
        Z = Jc = None
        if there_is_beam_source:
            Z = s["Zm_e"][:Nm_src, layers] @ P_e + s["Zm_o"][:Nm_src, layers] @ P_o
        if there_is_iso_source:
            Jc = (
                scaled_s_poly_coeffs[layers][:, :, None] + s["Jm_e"][layers] @ P_e[0] + s["Jm_o"][layers] @ P_o[0]
            ).transpose(0, 2, 1)
        return Z, Jc

    def _integrate_layers(hem, lo, hi):
        """Makes available the integrals of the source function over each whole layer from `lo` to `hi - 1`, along
        every direction of the hemisphere. If other layers were integrated before, every layer is integrated."""
        if hem["lo"] is not None:
            if lo >= hem["lo"] and hi <= hem["hi"]:
                return
            lo, hi = 0, NLayers
        upward, a, Nm_src = hem["upward"], hem["a"], hem["Nm_src"]
        nl, Na = hi - lo, len(a)
        d = thickness[lo:hi]
        Ea, tables = _angle_tables(d, a)                                            # (nl, Na)
        integrals = np.empty((Nm_src, nl, Na))
        if nl > 0:
            s = _get_statics(Nm_src)
            layers = slice(lo, hi)
            Na_c = max(1, _CHUNK_SIZE // (nl * N))
            for a0 in range(0, Na, Na_c):
                cols = slice(a0, a0 + Na_c)
                a_c = a[cols]
                tables_c = tuple(table[:, cols] for table in tables)
                P_e, P_o = hem["P_e"][:, :, cols], hem["P_o"][:, :, cols]
                B = max(1, _CHUNK_SIZE // (nl * N * len(a_c)))
                for m0 in range(0, Nm_src, B):
                    blk = slice(m0, min(m0 + B, Nm_src))
                    X_e, X_o = _source_coeffs(s, blk, layers, P_e, P_o)
                    integrals[blk, :, cols] = _integrate_exponentials(
                        upward, X_e, X_o, s["kappa"][blk, layers], d, a_c, tables_c,
                        *(array[blk, layers] for array in s["whole_layers"][upward]), _get_work_arrays,
                    )
            Z, Jc = _beam_and_iso_coeffs(s, hem, layers)
            if there_is_beam_source:
                integrals += Z * (_beam_integrals(upward, d, a) * np.exp(-st0[lo:hi] / mu0)[:, None])
            if there_is_iso_source:
                integrals[0] += _poly_integrals(
                    upward, np.arange(lo, hi), st0[lo:hi] if upward else st0[lo + 1 : hi + 1], a, Jc
                )
        hem.update(lo=lo, hi=hi, integrals=integrals, Ea=Ea)
        hem.pop("all_levels", None)

    def _store_source_coeffs(hem, layers):
        """Makes available the coefficients of the source function in `layers` (unique indices), for the evaluation
        within them."""
        if np.all(np.isin(layers, hem["src"])):
            return
        if len(hem["src"]) < 8:
            layers = np.union1d(layers, hem["src"])
        Nm_src, n, Na = hem["Nm_src"], len(layers), len(hem["a"])
        s = _get_statics(Nm_src)
        X_e = np.empty((Nm_src, n, N, Na))
        X_o = np.empty((Nm_src, n, N, Na))
        B = max(1, _CHUNK_SIZE // (n * N * Na))
        for m0 in range(0, Nm_src, B):
            blk = slice(m0, min(m0 + B, Nm_src))
            X_e[blk], X_o[blk] = _source_coeffs(s, blk, layers, hem["P_e"], hem["P_o"])
        Z, Jc = _beam_and_iso_coeffs(s, hem, layers)
        hem.update(src=layers, X_e=X_e, X_o=X_o, Z=Z, Jc=Jc)

    def _boundary_intensities(hem, levels, Nm):
        """Intensity of the first `Nm` Fourier modes at the layer boundaries `levels` (sorted unique indices, where
        boundary k is the top of layer k) along every direction of the hemisphere; (Nm, len(levels), Na).
        The intensity at the boundary where the paths start is attenuated and each layer's integral of the
        source function is added."""
        upward, a, Nm_src, lo, hi = hem["upward"], hem["a"], hem["Nm_src"], hem["lo"], hem["hi"]
        integrals, BC = hem["integrals"], hem["BC"]
        if len(levels) <= 2:
            # Few boundaries: sum over the layers directly
            out = np.empty((Nm, len(levels), len(a)))
            for i, k in enumerate(levels):
                if upward:
                    attenuation = np.exp((st0[k] - st0[k:])[:, None] / a[None, :])   # From the top of each layer below, and the bottom boundary
                    out[:, i] = BC * attenuation[-1]
                    out[:Nm_src, i] += np.einsum("mna, na -> ma", integrals[:, k - lo :], attenuation[:-1])
                else:
                    attenuation = np.exp((st0[: k + 1] - st0[k])[:, None] / a[None, :])  # From the top boundary, and the bottom of each layer above
                    out[:, i] = BC * attenuation[0]
                    out[:Nm_src, i] += np.einsum("mna, na -> ma", integrals[:, :k], attenuation[1:])
            return out
        # Many boundaries: propagate through the layers
        if "all_levels" not in hem:
            Ea = hem["Ea"]
            out = np.empty((Nm, hi - lo + 1, len(a)))
            if upward:
                out[:, -1] = BC
                for n in range(hi - lo - 1, -1, -1):
                    out[:, n] = out[:, n + 1] * Ea[n]
                    out[:Nm_src, n] += integrals[:, n]
            else:
                out[:, 0] = BC
                for n in range(hi - lo):
                    out[:, n + 1] = out[:, n] * Ea[n]
                    out[:Nm_src, n + 1] += integrals[:, n]
            hem["all_levels"] = out
        return hem["all_levels"][:, levels - lo]

    # Quantities that depend on `tau` but not on `mu` or `phi`, cached for the most recent `tau`
    # --------------------------------------------------------------------------------------------------------------------------
    _tau_cache = {"tau": None}

    def _process_tau(tau):
        if _tau_cache["tau"] is not None and np.array_equal(tau, _tau_cache["tau"]):
            return _tau_cache
        if np.any(tau < 0) or np.any(tau > tau_arr[-1]):
            raise ValueError("tau input outside the tau range specified for the atmosphere (check `tau_arr`).")
        # Atmospheric layer indices: smallest l with tau <= tau_arr[l]
        l = np.searchsorted(tau_arr, tau, side="left")
        if is_delta_scaled:
            t = st0[l + 1] - (tau_arr[l] - tau) * scale_tau[l]
        else:
            t = tau
        # On a layer boundary the intensity is that of the boundary; only elsewhere is there a path within a layer
        at_top = tau == 0
        at_bottom = tau == tau_arr[l]
        _tau_cache.update(
            tau=np.array(tau, copy=True), l=l, t=t,
            within=np.flatnonzero(~(at_top | at_bottom)),
            # Boundary at which the path to each `tau` enters its layer or, if `tau` is on a boundary, that boundary
            levels={True: np.unique(np.where(at_top, 0, l + 1), return_inverse=True),
                    False: np.unique(np.where(at_bottom, l + 1, l), return_inverse=True)},
            # Distances to the top and bottom of the layer
            x_top=np.maximum(t - st0[l], 0), x_bot=np.maximum(st0[l + 1] - t, 0),
            exps={},
        )
        return _tau_cache

    def _add_paths_within_layers(hem, tc, full):
        """For the `tau` values that are not on a layer boundary, attenuates the boundary intensities `full`
        (Nm, Ntau, Na) along the path within the layer and adds its integral of the source function."""
        upward, a, Nm_src = hem["upward"], hem["a"], hem["Nm_src"]
        idx = tc["within"]
        l, t = tc["l"][idx], tc["t"][idx]
        # Path within the layer: up from its bottom, or down from its top
        x, x_other = (tc["x_bot"][idx], tc["x_top"][idx]) if upward else (tc["x_top"][idx], tc["x_bot"][idx])
        exps = tc["exps"].get((upward, Nm_src))
        if exps is None:
            s = _get_statics(Nm_src)
            kappa = s["kappa"][:Nm_src, l]                                          # (Nm_src, n, N)
            exponent = -kappa * x[:, None]
            EK, EK1 = np.exp(exponent), np.expm1(exponent)
            # Upward, the exponentials with negative eigenvalues decay along the path and those with positive ones
            # grow; downward it is the opposite. Those which decay are referenced to the other boundary of the layer.
            C_dec, C_grow = (s["C_neg"], s["C_pos"]) if upward else (s["C_pos"], s["C_neg"])
            C_dec = C_dec[:Nm_src, l] * np.exp(-kappa * x_other[:, None])
            C_grow = C_grow[:Nm_src, l]
            exps = tc["exps"][(upward, Nm_src)] = (kappa, C_dec * EK, C_dec * EK1, C_grow, C_grow * EK, C_grow * EK1)

        pos = np.searchsorted(hem["src"], l)
        Ea, tables = _angle_tables(x, a)                                            # (n, Na)
        vals = np.empty((Nm_src, len(idx), len(a)))
        step = max(1, _CHUNK_SIZE // (Nm_src * len(a) * N))
        for i0 in range(0, len(idx), step):
            sub = slice(i0, i0 + step)
            vals[:, sub] = _integrate_exponentials(
                upward, hem["X_e"][:, pos[sub]], hem["X_o"][:, pos[sub]], exps[0][:, sub], x[sub], a,
                tuple(table[sub] for table in tables), *(array[:, sub] for array in exps[1:]), _get_work_arrays,
            )
        if there_is_beam_source:
            t_shallow = t if upward else st0[l]
            vals += hem["Z"][:, pos] * (_beam_integrals(upward, x, a) * np.exp(-t_shallow / mu0)[:, None])
        if there_is_iso_source:
            vals[0] += _poly_integrals(upward, l, t, a, hem["Jc"][pos])
        full[:, idx] *= Ea
        full[:Nm_src, idx] += vals

    def _antiderivative(hem, tc, full):
        """Antiderivative with respect to tau of the intensities `full` (Nm, Ntau, Na).
        Along any direction, mu du/dt = u - J, hence mu * u + (an antiderivative of J) is an antiderivative
        of u. The constant of the polynomial part is chosen to match the quadrature-node antiderivatives."""
        Nm_src, mu = hem["Nm_src"], hem["mu"]                                       # `mu` is signed
        l, t = tc["l"], tc["t"]
        weights = tc["exps"].get(("antiderivative", Nm_src))
        if weights is None:
            s = _get_statics(Nm_src)
            kappa = s["kappa"][:Nm_src, l]                                          # (Nm_src, Ntau, N)
            weights = tc["exps"][("antiderivative", Nm_src)] = (
                s["C_neg"][:Nm_src, l] * np.exp(-kappa * tc["x_top"][:, None]) / -kappa,
                s["C_pos"][:Nm_src, l] * np.exp(-kappa * tc["x_bot"][:, None]) / kappa,
            )
        pos = np.searchsorted(hem["src"], l)
        J_anti = np.zeros_like(full)
        step = max(1, _CHUNK_SIZE // (Nm_src * len(mu) * N))
        for i0 in range(0, len(l), step):
            sub = slice(i0, i0 + step)
            X_e, X_o = hem["X_e"][:, pos[sub]], hem["X_o"][:, pos[sub]]
            J_anti[:Nm_src, sub] = (
                np.einsum("mtka, mtk -> mta", X_e + X_o, weights[0][:, sub])
                + np.einsum("mtka, mtk -> mta", X_e - X_o, weights[1][:, sub])
            )
        if there_is_beam_source:
            J_anti[:Nm_src] += hem["Z"][:, pos] * (-mu0 * np.exp(-t / mu0))[None, :, None]
        if there_is_iso_source:
            pp = np.arange(Nscoeffs)
            Jc = hem["Jc"][pos]                                                     # (Ntau, Na, Nscoeffs)
            J_anti[0] += np.einsum("nap, np -> na", Jc, t[:, None] ** (pp + 1)[None, :] / (pp + 1)[None, :])
            fact = np.array([factorial(k) for k in pp], dtype=float)
            J_anti[0] -= mu[None, :] * np.einsum("nap, ap -> na", Jc, mu[:, None] ** pp[None, :] * fact[None, :])
        return (mu * full + J_anti) / scale_tau[l][:, None]

    # Fourier modes of the intensity at (`mu`, `tau`)
    # --------------------------------------------------------------------------------------------------------------------------
    def _hemisphere_modes(cache, upward, tc, is_antiderivative_wrt_tau, Nm):
        """Fourier modes at the angles of one hemisphere of `cache` and the optical depths of `tc`; (Nm, Ntau, Na)."""
        hem = _get_hemisphere(cache, upward, Nm)
        levels, inverse = tc["levels"][upward]
        # Upward paths start at the bottom boundary and downward paths at the top boundary
        if upward:
            _integrate_layers(hem, levels[0], NLayers)
        else:
            _integrate_layers(hem, 0, levels[-1])
        full = _boundary_intensities(hem, levels, Nm)[:, inverse]
        if is_antiderivative_wrt_tau:
            _store_source_coeffs(hem, np.unique(tc["l"]))
        elif len(tc["within"]) > 0:
            _store_source_coeffs(hem, np.unique(tc["l"][tc["within"]]))
        if len(tc["within"]) > 0:
            _add_paths_within_layers(hem, tc, full)
        if is_antiderivative_wrt_tau:
            full = _antiderivative(hem, tc, full)
        return full

    def _IotS_modes(mu, tau, is_antiderivative_wrt_tau, Nm):
        """Returns the first `Nm` Fourier modes of the (rescaled) intensity, or of its tau-antiderivative,
        at the polar angle cosines `mu` and optical depths `tau`, as an array of shape (Nm, Nmu, Ntau)."""
        if not np.all(np.abs(mu) <= 1):
            raise ValueError("mu values must be between -1 and 1.")
        tc = _process_tau(tau)
        cache = _get_mu_cache(mu, Nm)
        nup, order = cache["nup"], cache["order"]
        if order is None:
            return _hemisphere_modes(cache, nup > 0, tc, is_antiderivative_wrt_tau, Nm).transpose(0, 2, 1)
        um = np.empty((Nm, len(mu), len(tau)))
        um[:, order[:nup]] = _hemisphere_modes(cache, True, tc, is_antiderivative_wrt_tau, Nm).transpose(0, 2, 1)
        um[:, order[nup:]] = _hemisphere_modes(cache, False, tc, is_antiderivative_wrt_tau, Nm).transpose(0, 2, 1)
        return um

    # The two intensity functions
    # --------------------------------------------------------------------------------------------------------------------------
    def u0(mu, tau, is_antiderivative_wrt_tau=False):
        """
        Zeroth Fourier mode of the intensity with arguments `(mu, tau)` each of type array or float.
        Returns an ndarray with axes corresponding to variation with `mu` and `tau` respectively.
        Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
        to switch to an antiderivative of this function with respect to `tau`.
        """
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        return rescale_factor * np.squeeze(_IotS_modes(mu, tau, is_antiderivative_wrt_tau, 1)[0])[()]

    if only_flux:
        return u0, None

    def u(mu, tau, phi, is_antiderivative_wrt_tau=False, return_Fourier_error=False):
        """
        Intensity function with arguments `(mu, tau, phi)` each of type array or float.
        Returns an ndarray with axes corresponding to variation with `mu, tau, phi` respectively.
        Pass `return_Fourier_error = True` (defaults to `False`) to return the
        Cauchy / Fourier convergence evaluation (type: float) for the last Fourier term.
        Pass `is_antiderivative_wrt_tau = True` (defaults to `False`)
        to switch to an antiderivative of this function with respect to `tau`.
        """
        mu = np.atleast_1d(mu)
        tau = np.atleast_1d(tau)
        phi = np.atleast_1d(phi)
        um = _IotS_modes(mu, tau, is_antiderivative_wrt_tau, NFourier)             # (NFourier, Nmu, Ntau)
        cosines = np.cos(np.arange(NFourier)[:, None] * (phi0 - phi)[None, :])     # (NFourier, Nphi)
        u = (um.reshape(NFourier, -1).T @ cosines).reshape(len(mu), len(tau), len(phi))

        if return_Fourier_error:
            u_abs = np.abs(u)
            Fourier_error = np.max(
                np.divide(
                    np.abs(um[-1][:, :, None] * cosines[-1][None, None, :]),
                    u_abs,
                    out=np.zeros_like(u_abs),
                    where=u_abs > 1e-8,
                )
            )
            return rescale_factor * np.squeeze(u)[()], Fourier_error
        return rescale_factor * np.squeeze(u)[()]

    return u0, u
