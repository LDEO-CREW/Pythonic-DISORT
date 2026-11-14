import numpy as np
import warnings
from functools import lru_cache

from PythonicDISORT.subroutines import Gauss_Legendre_quad


@lru_cache(maxsize=32)
def _asso_leg_recurrence_coeffs(NFourier, NLeg):
    """Coefficients of the three-term recurrence of `_asso_leg_table_by_offset`,
    each of shape (NLeg, NFourier, 1), and its diagonal seed factors of shape (NFourier, 1)."""
    ms_all = np.arange(NFourier)
    ells = ms_all[None, :] + np.arange(NLeg)[:, None]                                # Degree l = m + r; (NLeg, NFourier)
    with np.errstate(invalid="ignore", divide="ignore"):                             # Unused rows r < 2
        c1 = (2 * ells - 1) / np.sqrt((ells - ms_all) * (ells + ms_all))
        c2 = np.sqrt((ells - 1 - ms_all) * (ells - 1 + ms_all) / ((ells - ms_all) * (ells + ms_all)))
    # The rows r < 2 are seeds, and the degrees l >= NLeg are not part of the table and are left as zeros
    c1[:2] = c2[:2] = 0
    c1[ells >= NLeg] = 0
    c2[ells >= NLeg] = 0
    # Diagonal seeds Pbar_m^m = -sqrt((2m - 1) / (2m)) * sqrt(1 - x^2) * Pbar_{m-1}^{m-1}
    # for all orders at once via a cumulative product over m
    diag_factors = np.cumprod(
        np.concatenate([[1.0], -np.sqrt((2 * ms_all[1:] - 1) / (2 * ms_all[1:]))])
    )
    # First recurrence step Pbar_{m+1}^m = sqrt(2m + 1) * x * Pbar_m^m
    first_step = np.sqrt(2 * ms_all + 1) * (ms_all + 1 < NLeg)
    outputs = (c1[:, :, None], c2[:, :, None], diag_factors[:, None], first_step[:, None])
    for array in outputs:
        array.setflags(write=False)
    return outputs


def _asso_leg_table_by_offset(x, NFourier, NLeg):
    """Table of seminormalized associated Legendre functions
    Pbar_l^m(x) = sqrt((l - m)! / (l + m)!) * P_l^m(x)
    (Condon-Shortley phase included, matching `scipy.special.lpmv` up to the normalization)
    for all orders m < NFourier and degrees l < NLeg, built in a single pass by the standard,
    numerically stable three-term recurrences.

    Returns
    -------
    3darray of shape (NLeg, NFourier, len(x)) whose entry [r, m, :] is the function of degree l = m + r
    (zero if m + r >= NLeg). The functions of even r are even in `x` and those of odd r are odd.
    """
    c1, c2, diag_factors, first_step = _asso_leg_recurrence_coeffs(NFourier, NLeg)
    table = np.empty((NLeg, NFourier, len(x)))
    table[0] = diag_factors * np.sqrt(1 - x**2)[None, :] ** np.arange(NFourier)[:, None]
    if NLeg > 1:
        table[1] = first_step * x[None, :] * table[0]

    # March up in degree, vectorized over all orders simultaneously.
    # Seminormalized form of (l - m) P_l^m = (2l - 1) x P_{l-1}^m - (l + m - 1) P_{l-2}^m.
    c1_x = c1 * x                                                                    # (NLeg, NFourier, Nx)
    buffer = np.empty(table.shape[1:])
    for r in range(2, NLeg):
        np.multiply(c1_x[r], table[r - 1], out=table[r])
        np.multiply(c2[r], table[r - 2], out=buffer)
        table[r] -= buffer
    return table


def _seminormalized_asso_leg_table(x, NFourier, NLeg):
    """`_asso_leg_table_by_offset` with the order as the zeroth axis.

    Returns
    -------
    3darray of shape (NFourier, NLeg, len(x)) whose entry [m, r, :] is the function of degree l = m + r
    (zero if m + r >= NLeg), so that `table[m, :NLeg - m, :]` holds all the degrees of order m.
    """
    return np.ascontiguousarray(_asso_leg_table_by_offset(x, NFourier, NLeg).transpose(1, 0, 2))


# In-process caches for the seminormalized associated Legendre tables (see the `cache_asso_leg`
# argument of `pydisort`). The tables at the quadrature nodes are fully determined by `N` so they can
# be reused across the `pydisort` calls of a batch run with a fixed
# discretization. Since `pydisort` enforces `NFourier <= NLeg`, the caches are built at the maximal 
# order `NFourier = NLeg`. The cached arrays are returned read-only. `maxsize` bounds the memory
# footprint; realistic batch runs use a single discretization (one live entry).

@lru_cache(maxsize=32)
def _cached_pos_asso_leg_table(N, NLeg):
    """`_seminormalized_asso_leg_table` at the positive quadrature nodes `mu_arr_pos`, which depend
    only on `N`, built at the maximal order `NFourier = NLeg`. Keyed on `(N, NLeg)` and hence shared
    across all `mu0` and all `NFourier <= NLeg` (callers slice `[:NFourier]`).
    """
    mu_arr_pos, _ = Gauss_Legendre_quad(N)
    table = _seminormalized_asso_leg_table(mu_arr_pos, NLeg, NLeg)
    table.setflags(write=False)
    return table


@lru_cache(maxsize=32)
def _cached_full_asso_leg_table(N, mu0, NLeg):
    """`_seminormalized_asso_leg_table` at `[mu_arr_pos, -mu0]`, i.e. including the direct-beam
    column, built at the maximal order `NFourier = NLeg`. Keyed on `(N, mu0, NLeg)`; only worthwhile
    when `mu0` is (near-)constant across many calls, since a distinct `mu0` misses the cache and
    rebuilds the whole table. Callers slice `[:NFourier]`.
    """
    mu_arr_pos, _ = Gauss_Legendre_quad(N)
    table = _seminormalized_asso_leg_table(np.append(mu_arr_pos, -mu0), NLeg, NLeg)
    table.setflags(write=False)
    return table


def _solve_for_gen_and_part_sols(
    NFourier,                    # Number of intensity Fourier modes
    scaled_omega_arr,            # Delta-scaled single-scattering albedos
    mu_arr_pos,                  # Upper hemisphere quadrature nodes (cosine polar angles)
    M_inv, W,                    # 1) 1 / `mu_arr_pos`; 2) quadrature weights for each hemisphere
    N, NQuad, NLeg,              # Number of 1) upper 2) both hemispheres quadrature nodes; 3) phase function Legendre coefficients 
    NLayers,                     # Number of layers
    weighted_scaled_Leg_coeffs,  # Weighted and delta-scaled Legendre coefficients
    mu0, I0_div_4pi,             # Primary properties of the direct beam
    there_is_beam_source,        # Is there a beam source?
    scaled_mu0_arr,              # Cosine of the direct beam in each delta-scaled layer
    there_is_iso_source,         # Is there an isotropic source?
    cache_asso_leg,              # Assoc. Legendre table cache mode: None, "no_mu0" or "mu0"
):
    """
    Diagonalizes the coefficient matrix of the system of ordinary differential equations (ODEs) 
    for each Fourier mode and returns the eigenpairs which give the general solution up to unknown coefficients.
    Also solves for the particular solution to each system of ODEs and returns its coefficient vector.
    Returns `G_collect`, `K_collect`, `B_collect` (`None` without a beam source), `G_inv_collect_0`
    (`None` without an isotropic source) and the associated Legendre table `asso_leg_table`.
    This function is wrapped by the `_assemble_intensity_and_fluxes` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    Arguments of _solve_for_gen_and_part_sols
    |            Variable            |            Type / Shape            |
    | ------------------------------ | ---------------------------------- |
    | `NFourier`                     | scalar                             |
    | `scaled_omega_arr`             | `NLayers`                          |
    | `mu_arr_pos`                   | `NQuad/2`                          |
    | `M_inv`                        | `NQuad/2`                          |
    | `W`                            | `NQuad/2`                          |
    | `N`                            | scalar                             |
    | `NQuad`                        | scalar                             |
    | `NLeg`                         | scalar                             |
    | `NLayers`                      | scalar                             |
    | `weighted_scaled_Leg_coeffs`   | `NLayers x NLeg`                   |
    | `mu0`                          | scalar                             |
    | `I0_div_4pi`                   | scalar                             |
    | `there_is_beam_source`         | boolean                            |
    | `scaled_mu0_arr`               | `NLayers` or `None`                |
    | `there_is_iso_source`          | boolean                            |
    | `cache_asso_leg`               | None, "no_mu0" or "mu0"            |

    Notable internal variables of _solve_for_gen_and_part_sols
    |       Variable      |             Type / Shape               |
    | ------------------- | -------------------------------------- |
    | `G_collect`         | `NFourier*NLayers x NQuad x NQuad`     | Reshaped to NFourier x NLayers x NQuad x NQuad
    | `K_collect`         | `NFourier*NLayers x NQuad`             | Reshaped to NFourier x NLayers x NQuad
    | `alpha_arr`         | `Nns x NQuad/2 x NQuad/2`              | Nns is the number of (mode, layer) pairs without the no-scattering shortcut
    | `beta_arr`          | `Nns x NQuad/2 x NQuad/2`              |
    | `B_collect`         | `NFourier*NLayers x NQuad` or `None`   | Reshaped to NFourier x NLayers x NQuad
    | `eigenvecs_GpG_arr` | `Nns x NQuad/2 x NQuad/2`              |
    | `eigenvecs_GmG_arr` | `Nns x NQuad/2 x NQuad/2`              |
    | `inv_mu0_arr`       | `Nns`                                  | Rate at which the direct beam decays in the layer of each pair
    | `G_inv_collect_0`   | `NLayers x NQuad x NQuad` or `None`    |
    | `asso_leg_table`    | `NFourier x NLeg x (NQuad/2 or NQuad/2 + 1)` | Entry [m, r, :] has degree l = m + r; the last column is at `-mu0` (if there is a beam)

    """
    ############################### Assemble system and diagonalize coefficient matrix #########################################
    ########################### Refer to section 3.4.2 of the Comprehensive Documentation  #####################################
    
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------------
    signs_all = np.ones(NLeg)
    signs_all[1::2] = -1
    
    G_trivial = np.zeros((NQuad, NQuad))
    stride = NQuad + 1
    G_trivial.ravel()[N : N + N * stride : stride] = 1
    G_trivial.ravel()[NQuad * N : NQuad * N + N * stride : stride] = 1
    
   # Flat (NFourier * NLayers, ...) storage so a single integer index list can address
   # every (m, l) pair across Fourier modes for the cross-mode batched `eig`.
    G_collect = np.empty((NFourier * NLayers, NQuad, NQuad))
    K_collect = np.empty((NFourier * NLayers, NQuad))
    A_arr = np.empty((NFourier * NLayers, NQuad, NQuad))
    no_shortcut_indices = []
    no_shortcut_indices_0 = []

    if there_is_beam_source:
        if cache_asso_leg == "mu0":
            # Cache the full table, including the direct-beam `-mu0` column (mu0 assumed ~constant).
            asso_leg_table = _cached_full_asso_leg_table(N, mu0, NLeg)[:NFourier]
        elif cache_asso_leg == "no_mu0":
            # Cache only the mu0-independent part and rebuild the single `-mu0` column per call.
            # The recurrence is separable over the evaluation points, so appending the freshly built
            # column to the (sliced) cached quadrature-node table reproduces the full table exactly.
            asso_leg_table = np.concatenate(
                (
                    _cached_pos_asso_leg_table(N, NLeg)[:NFourier],
                    _seminormalized_asso_leg_table(np.array([-mu0]), NFourier, NLeg),
                ),
                axis=2,
            )
        else:
            asso_leg_table = _seminormalized_asso_leg_table(np.append(mu_arr_pos, -mu0), NFourier, NLeg)

        X_arr = np.empty((NFourier * NLayers, NQuad))
        B_collect = np.zeros((NFourier * NLayers, NQuad))
    else:
        # Without the direct beam there is no `-mu0` column, so "no_mu0" and "mu0" both reduce to the
        # mu0-independent quadrature-node table.
        if cache_asso_leg is None:
            asso_leg_table = _seminormalized_asso_leg_table(mu_arr_pos, NFourier, NLeg)
        else:
            asso_leg_table = _cached_pos_asso_leg_table(N, NLeg)[:NFourier]
    if there_is_iso_source:
        G_inv_collect_0 = np.empty((NLayers, NQuad, NQuad))
    # --------------------------------------------------------------------------------------------------------------------------

    # Loop over NFourier Fourier modes
    # These can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    # --------------------------------------------------------------------------------------------------------------------------
    for m in range(NFourier):
        # Setup
        # --------------------------------------------------------------------------------------------------------------------------
        m_equals_0 = (m == 0)
        signs = signs_all[:NLeg - m]

        if there_is_beam_source:
            asso_leg_m = asso_leg_table[m, :NLeg - m, :]
            asso_leg_term_pos = asso_leg_m[:, :-1]
            scalar_fac_asso_leg_term_mu0 = I0_div_4pi * (2 - m_equals_0) * asso_leg_m[:, -1]
        else:
            asso_leg_term_pos = asso_leg_table[m, :NLeg - m, :]

        asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
        fac_asso_leg_term_posT = asso_leg_term_pos.T
        # --------------------------------------------------------------------------------------------------------------------------

        # Vectorized over NLayers atmospheric layers
        # --------------------------------------------------------------------------------------------------------------------------
        omega_times_Leg = (scaled_omega_arr[:, None] / 2) * weighted_scaled_Leg_coeffs[:, m:]   # (NLayers, NLeg-m)
        no_shortcut = np.any(np.abs(omega_times_Leg) > 1e-8, axis=1)                           # (NLayers,)
        ns_ind = np.flatnonzero(no_shortcut)                                                        
        sh_ind = np.flatnonzero(~no_shortcut)                                                       
        ind_base = m * NLayers                                                                  

        # Shortcut layers: trivial diagonalization (no multiple scattering)
        if sh_ind.size:
            G_collect[ind_base + sh_ind] = G_trivial
            K_collect[ind_base + sh_ind, :N] = -M_inv
            K_collect[ind_base + sh_ind, N:] = M_inv
            if m_equals_0 and there_is_iso_source:
                G_inv_collect_0[sh_ind] = G_trivial

        # Non-shortcut layers: full computation, batched over NLayers
        if ns_ind.size:
            D_temp = omega_times_Leg[ns_ind][:, None, :] * fac_asso_leg_term_posT[None, :, :]     # (Nns, N, NLeg-m)
            D_pos = D_temp @ asso_leg_term_pos                                                    # (Nns, N, N)
            D_neg = D_temp @ asso_leg_term_neg                                                    # (Nns, N, N)
            DW = D_pos * W[None, None, :]                                                         
            DW.reshape(-1, N * N)[:, ::N + 1] -= 1                                                # batched diag-sub
            inds = ind_base + ns_ind                                                             
            A_arr[inds, N:, N:] = M_inv[None, :, None] * DW                                       # alpha
            A_arr[inds, N:, :N] = M_inv[None, :, None] * D_neg * W[None, None, :]                 # beta
            no_shortcut_indices.extend(inds.tolist())                                            
            if m_equals_0 and there_is_iso_source:
                no_shortcut_indices_0.extend(ns_ind.tolist())                                      
            if there_is_beam_source:
                X_temp = (                                                                        
                    scalar_fac_asso_leg_term_mu0                                                  
                    * scaled_omega_arr[ns_ind, None]                                                  
                    * weighted_scaled_Leg_coeffs[ns_ind, m:]                                          
                )                                                                                
                # Signs flipped due to the extra minus sign
                X_arr[inds, :N] =  M_inv * (X_temp @ asso_leg_term_pos)                           
                X_arr[inds, N:] = -M_inv * (X_temp @ asso_leg_term_neg)                           

    if len(no_shortcut_indices) > 0:
        A_arr = A_arr[no_shortcut_indices, :, :]
        alpha_arr = A_arr[:, N:, N:]
        beta_arr = A_arr[:, N:, :N]
        
        # Diagonalization of coefficient matrix (refer to section 3.5 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        apb = alpha_arr + beta_arr
        amb = alpha_arr - beta_arr
        K_squared_arr, eigenvecs_GmG_arr = np.linalg.eig(apb @ amb)

        # In theory K^2 is real and positive, but `eig` may return complex eigenpairs either as
        # numerical artifacts or due to the truncated phase function genuinely having complex eigenvalues.
        # We warn instead of raising an error as `pydisort` may be called an enormous number of times in long runs,
        # and the real parts have been observed to sometimes still produce reasonable outputs, then proceed with the real parts.
        # Note that Python prints each distinct warning only once per process by default.
        # In NumPy >= 2.5 `eig` always returns complex arrays, thus we test the imaginary parts, not the dtype.
        if np.iscomplexobj(K_squared_arr):
            if np.any(K_squared_arr.imag != 0):
                warnings.warn(
                    "Some eigenvalues of the coefficient matrices are incorrectly complex and their"
                    + " imaginary parts will be discarded. The results MAY still be reasonable"
                    + " but should be checked."
                )
            K_squared_arr = np.real(K_squared_arr)
            eigenvecs_GmG_arr = np.real(eigenvecs_GmG_arr)
        if np.any(K_squared_arr <= 0):
            warnings.warn(
                "Some squared eigenvalues of the coefficient matrices are non-positive."
                + " Their square roots, and consequently some outputs, will be NaN."
            )

        # Eigenvalues arranged negative then positive
        K_arr_pos = np.sqrt(K_squared_arr)
        K_arr = np.concatenate((-K_arr_pos, K_arr_pos), axis=1)
        
        # Build eigenvector matrix from four blocks
        eigenvecs_GmG_arr /= 2
        eigenvecs_GpG_arr = amb @ (eigenvecs_GmG_arr / K_arr_pos[:, None, :])
        GpG_p_GmG_div_2 = eigenvecs_GpG_arr + eigenvecs_GmG_arr
        GpG_m_GmG_div_2 = eigenvecs_GpG_arr - eigenvecs_GmG_arr
        
        G_collect[no_shortcut_indices, :N, :N] = GpG_p_GmG_div_2
        G_collect[no_shortcut_indices, N:, N:] = GpG_p_GmG_div_2
        G_collect[no_shortcut_indices, :N, N:] = GpG_m_GmG_div_2
        G_collect[no_shortcut_indices, N:, :N] = GpG_m_GmG_div_2
        
        K_collect[no_shortcut_indices, :] = K_arr
        len_no_shortcut_indices_0 = len(no_shortcut_indices_0)
        if len_no_shortcut_indices_0 > 0: # If there is no isotropic source this list will be empty
            G_inv_collect_0[no_shortcut_indices_0, :, :] = np.linalg.inv(
                G_collect[no_shortcut_indices_0, :, :]
            )

        # --------------------------------------------------------------------------------------------------------------------------
        
        if there_is_beam_source:
            
            # The direct beam decays at the rate 1/mu0 in the layer of each (mode, layer) pair
            inv_mu0_arr = 1 / scaled_mu0_arr[np.array(no_shortcut_indices) % NLayers]
            resonant = np.any(np.abs(K_arr + inv_mu0_arr[:, None]) < 1e-8, axis=1)

            # Warn if -1/mu0 is close to an eigenvalue (see section 3.6.1 of the Comprehensive Documentation)
            if np.any(resonant & ~np.any(np.abs(M_inv[None, :] - inv_mu0_arr[:, None]) < 1e-8, axis=1)):
                warnings.warn(
                    "`-1/mu0` is close to an eigenvalue so the results may be inaccurate."
                    + " Consider slightly perturbing `mu0`."
                )

            # Solve for particular solution w.r.t. direct beam source (refer to section 3.6.1 of the Comprehensive Documentation)
            # --------------------------------------------------------------------------------------------------------------------------
            if len_no_shortcut_indices_0 > 0: # In this case we have `G_inv_collect_0`, so use it
                B_collect[no_shortcut_indices_0, :] = (
                    # Note that the minus sign has been absorbed into `X_arr`
                    (
                        G_collect[no_shortcut_indices_0, :, :]
                        / (inv_mu0_arr[:len_no_shortcut_indices_0, None, None] + K_collect[no_shortcut_indices_0, None, :])
                    )
                    @ (G_inv_collect_0[no_shortcut_indices_0, :, :] @ X_arr[no_shortcut_indices_0, :, None])
                ).squeeze(-1)
                no_shortcut_indices = no_shortcut_indices[len_no_shortcut_indices_0:]
                inv_mu0_arr = inv_mu0_arr[len_no_shortcut_indices_0:]
                resonant = resonant[len_no_shortcut_indices_0:]
            
            if len(no_shortcut_indices) > 0:
                A_arr = A_arr[len_no_shortcut_indices_0:]
                A_arr[:, :N, N:] = -A_arr[:, N:, :N]
                A_arr[:, :N, :N] = -A_arr[:, N:, N:]
                A_arr.reshape(-1, NQuad * NQuad)[:, ::NQuad + 1] += inv_mu0_arr[:, None]
                X_arr = X_arr[no_shortcut_indices, :, None]
                # If -1/mu0 equals an eigenvalue to rounding, e.g. when `mu0` is a quadrature node which a
                # high Fourier mode barely scatters into or out of, the matrix is singular to working precision and
                # `solve` returns a huge, inaccurate `B`. The (mode, layer) pairs which trip the test of the warning above
                # are therefore solved by a pseudoinverse instead, which drops the directions that are singular to rounding.
                # Each row is first scaled to unit size so that the cutoff of the pseudoinverse is relative to that row.
                # Note that the minus sign has been absorbed into `X_arr`
                if np.any(resonant):
                    B_arr = np.empty((len(no_shortcut_indices), NQuad))
                    row_scale = 1 / (np.concatenate([M_inv, M_inv])[None, :, None] + inv_mu0_arr[resonant, None, None])
                    B_arr[resonant] = (
                        np.linalg.pinv(row_scale * A_arr[resonant], 1e-13) @ (row_scale * X_arr[resonant])
                    ).squeeze(-1)
                    if not np.all(resonant):
                        B_arr[~resonant] = np.linalg.solve(A_arr[~resonant], X_arr[~resonant]).squeeze(-1)
                    B_collect[no_shortcut_indices, :] = B_arr
                else:
                    B_collect[no_shortcut_indices, :] = np.linalg.solve(A_arr, X_arr).squeeze(-1)

            # --------------------------------------------------------------------------------------------------------------------------

    return (
        G_collect.reshape((NFourier, NLayers, NQuad, NQuad)),
        K_collect.reshape((NFourier, NLayers, NQuad)),
        B_collect.reshape((NFourier, NLayers, NQuad)) if there_is_beam_source else None,
        G_inv_collect_0 if there_is_iso_source else None,
        asso_leg_table,
    )