from PythonicDISORT.subroutines import _mathscr_v

import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy as sc


def _solve_for_coeffs(
    NFourier,                                 # Number of intensity Fourier modes
    G_collect,                                # Eigenvector matrices
    K_collect,                                # Eigenvalues
    B_collect,                                # Coefficients vectors for particular solutions
    G_inv_mu_inv,                             # Inverse of eigenvector matrix for the 0th Fourier mode
    scaled_tau_arr_with_0,                    # Delta-scaled lower boundary of layers with 0 inserted in front
    mu_arr_pos, mu_arr_pos_times_W,           # Upper hemisphere quadrature nodes (cosine polar angles); same but multiplied by weights
    N, NQuad,                                 # Number of 1) upper 2) both hemispheres quadrature nodes
    NLayers, NBDRF,                           # Number of 1) layers; 2) BDRF Fourier modes
    is_atmos_multilayered,                    # Is the atmosphere multilayered?
    BDRF_Fourier_modes,                       # BDRF Fourier modes
    mu0, I0_div_4pi,                          # Primary properties of the direct beam
    there_is_beam_source,                     # Is there a beam source?
    b_pos, b_neg,                             # Dirichlet BCs
    b_pos_is_scalar, b_neg_is_scalar,         # Is each Dirichlet BCs scalar and so isotropic?
    b_pos_is_vector, b_neg_is_vector,         # Is each Dirichlet BCs vector?
    Nscoeffs,                                 # Number of isotropic source polynomial coefficients
    scaled_s_poly_coeffs,                     # Polynomial coefficients of isotropic source           
    there_is_iso_source,                      # Is there an isotropic source?
    use_banded_solver_NLayers,                # Number of layers above or equal which to use `scipy.linalg.solve_banded`
):
    """
    Uses the boundary conditions to solve for the unknown coefficients 
    of the general solution to the system of ordinary differential equations for each Fourier mode. 
    Returns the product of the coefficients and the eigenvectors.
    This function is wrapped by the `_assemble_intensity_and_fluxes` function.
    It has many seemingly redundant arguments to maximize precomputation in the `pydisort` function.
    See the Jupyter Notebook, especially section 3, for documentation, explanation and derivation.
    The labels in this file reference labels in the Jupyter Notebook, especially sections 3 and 4.

    Arguments of _solve_for_coeffs
    |            Variable            |                  Type / Shape                |
    | ------------------------------ | -------------------------------------------- |
    | `NFourier`                     | scalar                                       |
    | `G_collect`                    | `NFourier x NLayers x NQuad x NQuad`         |
    | `K_collect`                    | `NFourier x NLayers x NQuad`                 |
    | `B_collect`                    | `NFourier x NLayers x NQuad` or `None`       |
    | `G_inv_mu_inv`                 | `NLayers x NQuad x NQuad` or `None`          |
    | `scaled_tau_arr_with_0`        | `NLayers + 1`                                |
    | `mu_arr_pos`                   | `NQuad/2`                                    |
    | `mu_arr_pos_times_W`           | `NQuad/2`                                    |
    | `N`                            | scalar                                       |
    | `NQuad`                        | scalar                                       |
    | `NLayers`                      | scalar                                       |
    | `NBDRF`                        | scalar                                       |
    | `is_atmos_multilayered`        | boolean                                      |
    | `BDRF_Fourier_modes`           | `NBDRF`                                      |
    | `mu0`                          | scalar                                       |
    | `I0_div_4pi`                   | scalar                                       |
    | `there_is_beam_source`         | boolean                                      |
    | `b_pos`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  |
    | `b_neg`                        | `NQuad/2 x NFourier` or `NQuad/2` or scalar  | 
    | `b_pos_is_scalar`              | boolean                                      |
    | `b_neg_is_scalar`              | boolean                                      |
    | `b_pos_is_vector`              | boolean                                      |
    | `b_neg_is_vector`              | boolean                                      |
    | `Nscoeffs`                     | scalar                                       |
    | `scaled_s_poly_coeffs`         | `NLayers x Nscoeffs`                         |
    | `there_is_iso_source`          | boolean                                      |
    | `use_banded_solver_NLayers`    | scalar                                       |
    
    Notable internal variables of _solve_for_coeffs
    |   Variable   |                 Shape                |
    | ------------ | ------------------------------------ |
    | `GC_collect` | `NFourier x NLayers x NQuad x NQuad` |

    """
    ################################## Solve for coefficients of homogeneous solution ##########################################
    ############# Refer to section 3.6.2 (single-layer) and 4 (multi-layer) of the Comprehensive Documentation #################
    
    use_banded_solver = (NLayers >= use_banded_solver_NLayers)
    dim = NLayers * NQuad
    C_collect = np.empty((NFourier, dim))
    thickness = np.diff(scaled_tau_arr_with_0)                                                       # (NLayers,)

    RHS = np.empty(dim)
    i_bot = dim - N # Equals `j_bot_right`
    j_bot_left = dim - NQuad

    if use_banded_solver:
        Nsupsubdiags = 3 * N - 1
        LHS_dof = np.zeros((6 * N - 1, dim), order="F") # Diagonal ordered form
        s0, s1 = LHS_dof.strides
        col_stride = s1 - s0  # Stride when moving +1 col and -1 row

        def _view_slanted(i0, j0, nrows):
            """Single slanted view into `LHS_dof` anchored at conceptual LHS coords (i0, j0).
            Returns a 2D view of shape (nrows, N). Used for top/bottom boundary-condition rows.

            Uses `numpy.lib.stride_tricks.as_strided` (zero-copy but unsafe), which is why
            this helper is not exposed in `PythonicDISORT.subroutines`.
            See https://numpy.org/devdocs/reference/generated/numpy.lib.stride_tricks.as_strided.html.

            Assume the following exist in the outer scope:
                `LHS_dof`, `Nsupsubdiags`, `N`, `s0`, `col_stride`.
            """
            r0 = Nsupsubdiags + i0 - j0
            base = LHS_dof[r0 : r0 + nrows, j0 : j0 + N]
            return as_strided(base, shape=(nrows, N), strides=(s0, col_stride))

        def _view_slanted_batched(i0, j0, nrows):
            """Batched (NLayers-1, nrows, N) slanted view across interlayer continuity rows.
            Anchored at conceptual LHS coords (i0, j0) for layer 0; layer l advances both
            i and j by `NQuad` along the diagonal.

            Same `(i0, j0)` convention as `_view_slanted` (slant arithmetic done internally).
            Assume the following exist in the outer scope:
                `LHS_dof`, `Nsupsubdiags`, `N`, `NQuad`, `NLayers`, `s0`, `s1`, `col_stride`.
            """
            r0 = Nsupsubdiags + i0 - j0 # matches `_view_slanted`'s internal slant
            base = LHS_dof[r0 : r0 + nrows, j0 : j0 + N + (NLayers - 2) * NQuad]
            return as_strided(base, shape=(NLayers - 1, nrows, N),
                              strides=(NQuad * s1, s0, col_stride))

    else:
        LHS = np.zeros((dim, dim), order="F")
        sR, sC = LHS.strides
        # Stacks for a single batched `np.linalg.solve` over all Fourier modes after the
        # assembly loop; this removes the per-mode LAPACK dispatch overhead. The banded
        # path keeps its per-mode loop as `scipy.linalg.solve_banded` cannot batch.
        LHS_batch = np.empty((NFourier, dim, dim))
        RHS_batch = np.empty((NFourier, dim))
        def _dense_slab(r0, c0, nrows, ncols):
            """Batched (NLayers-1, nrows, ncols) view across interlayer blocks of `LHS`.
            Anchored at LHS coords (r0, c0) for layer 0; layer l advances both row and column
            anchors by `NQuad` along the diagonal. Each slab has shape (nrows, ncols)."""
            base = LHS[r0 : r0 + nrows + (NLayers - 2) * NQuad,                                       
                       c0 : c0 + ncols + (NLayers - 2) * NQuad]                                       
            return as_strided(base, shape=(NLayers - 1, nrows, ncols),                                
                              strides=(NQuad * (sR + sC), sR, sC))                                    
    
    # The following loops can easily be parallelized, but the speed-up is unlikely to be worth the overhead
    for m in range(NFourier):
        m_equals_0 = (m == 0)
        there_is_BDRF_mode = (NBDRF > m) 
        
        G_collect_m = G_collect[m, :, :, :]
        K_collect_m = K_collect[m, :, :]
            
        # Generate mathscr_D and mathscr_X (BDRF terms)
        # Just for this part, refer to section 3.4.2 of the Comprehensive Documentation 
        # --------------------------------------------------------------------------------------------------------------------------
        if there_is_BDRF_mode:
            BDRF_Fourier_modes_m = BDRF_Fourier_modes[m]
            if np.isscalar(BDRF_Fourier_modes_m):
                mathscr_D_neg = (1 + m_equals_0 * 1) * BDRF_Fourier_modes_m
                R = mathscr_D_neg * mu_arr_pos_times_W[None, :]
                if there_is_beam_source:
                    mathscr_X_pos = (mu0 * I0_div_4pi * 4) * BDRF_Fourier_modes_m
            else:
                mathscr_D_neg = (1 + m_equals_0 * 1) * BDRF_Fourier_modes_m(mu_arr_pos, mu_arr_pos)
                R = mathscr_D_neg * mu_arr_pos_times_W[None, :]
                if there_is_beam_source:
                    mathscr_X_pos = (mu0 * I0_div_4pi * 4) * BDRF_Fourier_modes_m(
                        mu_arr_pos, np.array([mu0])
                    )[:, 0]
                    
        # --------------------------------------------------------------------------------------------------------------------------
    
    
        # Assemble RHS
        # --------------------------------------------------------------------------------------------------------------------------
        # Ensure the Dirchlet BCs are of the correct shape
        if b_pos_is_scalar and m_equals_0:
            b_pos_m = np.full(N, b_pos)
        elif b_pos_is_vector and m_equals_0:
            b_pos_m = b_pos
        elif (b_pos_is_scalar or b_pos_is_vector) and not m_equals_0:
            b_pos_m = np.zeros(N)
        else:
            b_pos_m = b_pos[:, m]
            
        if b_neg_is_scalar and m_equals_0:
            b_neg_m = np.full(N, b_neg)
        elif b_neg_is_vector and m_equals_0:
            b_neg_m = b_neg
        elif (b_neg_is_scalar or b_neg_is_vector) and not m_equals_0:
            b_neg_m = np.zeros(N)
        else:
            b_neg_m = b_neg[:, m]
        
        # Top Dirichlet BC
        RHS[:N] = b_neg_m
        # Bottom Dirichlet BC
        RHS[-N:] = b_pos_m
        
        # Reset the rest of `RHS` to 0
        RHS[N:-N] = 0
        
        if m_equals_0 and there_is_iso_source:
            # Top boundary
            RHS[:N] -= _mathscr_v(
                        np.array([0]),
                        None,
                        np.array([0]),
                        Nscoeffs,
                        scaled_s_poly_coeffs[[0], :],
                        G_collect_m[[0], N:, :],
                        K_collect_m[[0], :],
                        G_inv_mu_inv[[0]],
                    ).ravel()
        
            if is_atmos_multilayered:
                indices = np.arange(NLayers - 1)
                indicesp1 = indices + 1
                RHS[N:-N] += (
                    _mathscr_v(
                        scaled_tau_arr_with_0[1:-1],
                        None,
                        indices,
                        Nscoeffs,
                        scaled_s_poly_coeffs[indicesp1],
                        G_collect_m[indicesp1],
                        K_collect_m[indicesp1],
                        G_inv_mu_inv[indicesp1],
                    )
                    - _mathscr_v(
                        scaled_tau_arr_with_0[1:-1],
                        None,
                        indices,
                        Nscoeffs,
                        scaled_s_poly_coeffs[indices],
                        G_collect_m[indices],
                        K_collect_m[indices],
                        G_inv_mu_inv[indices],
                    )
                ).ravel(order='F')
                
            # Bottom boundary
            RHS[-N:] -= _mathscr_v(
                scaled_tau_arr_with_0[[-1]],
                None,
                np.array([0]),
                Nscoeffs,
                scaled_s_poly_coeffs[[-1], :],
                G_collect_m[[-1], :N, :],
                K_collect_m[[-1], :],
                G_inv_mu_inv[[-1]],
            ).ravel()
            
            if there_is_BDRF_mode:
                RHS[-N:] += (
                    R
                    @ _mathscr_v(
                        scaled_tau_arr_with_0[[-1]],
                        None,
                        np.array([0]),
                        Nscoeffs,
                        scaled_s_poly_coeffs[[-1], :],
                        G_collect_m[[-1], N:, :],
                        K_collect_m[[-1], :],
                        G_inv_mu_inv[[-1]],
                    ).ravel()
                )
        
        if there_is_beam_source:
            B_collect_m = B_collect[m, :, :]
        
            # Top boundary
            RHS[:N] -= B_collect_m[0, N:]
        
            if is_atmos_multilayered:
                indicesp1 = np.arange(1, NLayers)
                RHS[N:-N] += (
                    (B_collect_m[indicesp1, :] - B_collect_m[indicesp1 - 1, :])
                    * np.exp(-scaled_tau_arr_with_0 / mu0)[indicesp1, None]
                ).ravel()
            
            # Bottom boundary
            if there_is_BDRF_mode:
                RHS[-N:] += (
                    mathscr_X_pos + R @ B_collect_m[-1, N:] - B_collect_m[-1, :N]
                ) * np.exp(-scaled_tau_arr_with_0[-1] / mu0)

            else:
                RHS[-N:] -= B_collect_m[-1, :N] * np.exp(-scaled_tau_arr_with_0[-1] / mu0)
            
        # --------------------------------------------------------------------------------------------------------------------------
        
        
        # Assemble LHS and solve (the old version of this code is replicated in section 4 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        G_0 = G_collect_m[0]
        G_0_nn = G_0[N:, :N]
        G_0_np = G_0[N:, N:]
        E_01 = np.exp(K_collect_m[0, :N] * scaled_tau_arr_with_0[1])[None, :]

        G_L = G_collect_m[-1]
        G_L_pn = G_L[:N, :N]
        G_L_nn = G_L[N:, :N]
        G_L_pp = G_L[:N, N:]
        G_L_np = G_L[N:, N:]
        E_Lm1L = np.exp(
            K_collect_m[-1, :N] * (scaled_tau_arr_with_0[-1] - scaled_tau_arr_with_0[-2])
        )[None, :]
        # Per-layer thickness exponentials, computed once per Fourier mode and reused below.
        # If looping over `l in range(NLayers)`, the line below would be `E_pos_full[l] = exp(-thickness[l] * K_pos[l])`
        E_pos_full = np.exp(-thickness[:, None] * K_collect_m[:, N:])                                # (NLayers, N)
        
        ################################## Use `sc.linalg.solve_banded` ##################################
        if use_banded_solver:       
        
            # Directly assemble diagonal ordered form of LHS matrix
            # ---------------- Top BC ----------------
            G_0 = G_collect_m[0]
            G_0_nn = G_0[N:, :N]     # (N,N)
            G_0_np = G_0[N:, N:]     # (N,N)

            _view_slanted(0, 0, N)[:] = G_0_nn  # `_view_slanted` was defined with respect to `LHS_dof`
            _view_slanted(0, N, N)[:] = G_0_np * E_01

            # ---------------- Bottom BC ----------------
            if there_is_BDRF_mode:
                _view_slanted(i_bot, j_bot_left, N)[:] = (G_L_pn - R @ G_L_nn) * E_Lm1L
                _view_slanted(i_bot, i_bot, N)[:] = G_L_pp - R @ G_L_np
            else:
                _view_slanted(i_bot, j_bot_left, N)[:] = G_L_pn * E_Lm1L
                _view_slanted(i_bot, i_bot, N)[:] = G_L_pp

            # ---------------- Interlayer / continuity BCs (batched across layers) ----------------
            E_lm1l = E_pos_full[:-1, None, :]                                                        # (NLayers-1, 1, N)
            E_llp1 = E_pos_full[1:,  None, :]                                                        # (NLayers-1, 1, N)
            G_lo = G_collect_m[:-1]                                                                  # (NLayers-1, NQuad, NQuad)
            G_hi = G_collect_m[1:]                                                                   # (NLayers-1, NQuad, NQuad)

            _view_slanted_batched(N,       0,       N)[:]    = G_lo[:, :N, :N] * E_lm1l            
            _view_slanted_batched(2 * N,   0,       N)[:]    = G_lo[:, N:, :N] * E_lm1l            
            _view_slanted_batched(N,       N,       NQuad)[:] = G_lo[:, :,  N:]                    
            _view_slanted_batched(N,       NQuad,   NQuad)[:] = -G_hi[:, :,  :N]                   
            _view_slanted_batched(N,       3 * N,   N)[:]    = -G_hi[:, :N, N:] * E_llp1           
            _view_slanted_batched(2 * N,   3 * N,   N)[:]    = -G_hi[:, N:, N:] * E_llp1           
        
            # Solve the system
            C_collect[m, :] = sc.linalg.solve_banded(
                (Nsupsubdiags, Nsupsubdiags),
                LHS_dof,
                RHS,
                False,
                True,
                True,
            )
        
        ##################################################################################################
        ##################################### Use `np.linalg.solve` ######################################
        else:
        
            # ---------------- Top BC ----------------
            LHS[:N, :N] = G_0_nn
            LHS[:N, N : NQuad] = G_0_np * E_01
            
            # ---------------- Bottom BC ----------------
            if there_is_BDRF_mode:
                LHS[-N:, -NQuad : -N] = (G_L_pn - R @ G_L_nn) * E_Lm1L
                LHS[-N:, -N:] = G_L_pp - R @ G_L_np
            else:
                LHS[-N:, -NQuad : -N] = G_L_pn * E_Lm1L
                LHS[-N:, -N:] = G_L_pp

            # ---------------- Interlayer / continuity BCs (batched across layers) ----------------
            E_lm1l = E_pos_full[:-1, None, :]                                                        # (NLayers-1, 1, N)
            E_llp1 = E_pos_full[1:,  None, :]                                                        # (NLayers-1, 1, N)
            G_lo = G_collect_m[:-1]                                                                  
            G_hi = G_collect_m[1:]                                                                   

            _dense_slab(N,       0,       N,     N)[:]    = G_lo[:, :N, :N] * E_lm1l                 
            _dense_slab(2 * N,   0,       N,     N)[:]    = G_lo[:, N:, :N] * E_lm1l                 
            _dense_slab(N,       N,       NQuad, N)[:]    = G_lo[:, :,  N:]                          
            _dense_slab(N,       NQuad,   NQuad, N)[:]    = -G_hi[:, :,  :N]                         
            _dense_slab(N,       3 * N,   N,     N)[:]    = -G_hi[:, :N, N:] * E_llp1                
            _dense_slab(2 * N,   3 * N,   N,     N)[:]    = -G_hi[:, N:, N:] * E_llp1                
                
            # Defer to the single batched solve over all Fourier modes below
            LHS_batch[m] = LHS
            RHS_batch[m] = RHS

            ##################################################################################################
            # --------------------------------------------------------------------------------------------------------------------------

    if not use_banded_solver:
        # One batched call for all Fourier modes
        C_collect = np.linalg.solve(LHS_batch, RHS_batch[:, :, None])[:, :, 0]

    return G_collect * C_collect.reshape((NFourier, NLayers, 1, NQuad))
