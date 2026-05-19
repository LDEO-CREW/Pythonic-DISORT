import numpy as np
import scipy as sc


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
    there_is_iso_source,         # Is there an isotropic source?
):
    """
    Diagonalizes the coefficient matrix of the system of ordinary differential equations (ODEs) 
    for each Fourier mode and returns the eigenpairs which give the general solution up to unknown coefficients.
    Also solves for the particular solution to each system of ODEs and returns its coefficient vector.
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
    | `there_is_iso_source`          | boolean                            |
    
    Notable internal variables of _solve_for_gen_and_part_sols
    |       Variable      |             Type / Shape               |
    | ------------------- | -------------------------------------- |
    | `ells_all`          | `NLeg`                                 |
    | `G_collect`         | `NFourier*NLayers x NQuad x NQuad`     | Reshaped to NFourier x NLayers x NQuad x NQuad
    | `K_collect`         | `NFourier*NLayers x NQuad`             | Reshaped to NFourier x NLayers x NQuad
    | `alpha_arr`         | `NFourier*NLayers x NQuad/2 x NQuad/2` | Reshaped to NFourier x NLayers x NQuad/2 x NQuad/2
    | `beta_arr`          | `NFourier*NLayers x NQuad/2 x NQuad/2` | Reshaped to NFourier x NLayers x NQuad/2 x NQuad/2
    | `B_collect`         | `NFourier*NLayers x NQuad` or `None`   | Reshaped to NFourier x NLayers x NQuad
    | `eigenvecs_GpG_arr` | `NFourier*NLayers x NQuad/2 x NQuad    |
    | `eigenvecs_GmG_arr` | `NFourier*NLayers x NQuad/2 x NQuad    |
    | `G_inv_collect_0`   | `NLayers x NQuad x NQuad` or `None`    |

    """
    ############################### Assemble system and diagonalize coefficient matrix #########################################
    ########################### Refer to section 3.4.2 of the Comprehensive Documentation  #####################################
    
    # Initialization
    # --------------------------------------------------------------------------------------------------------------------------
    ells_all = np.arange(NLeg)
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
        mu_arr_pos_mu0 = np.append(mu_arr_pos, -mu0)
        
        X_arr = np.empty((NFourier * NLayers, NQuad))
        B_collect = np.zeros((NFourier * NLayers, NQuad))
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
        ells = ells_all[m:]
        poch = sc.special.poch(ells + m + 1, -2 * m)
        signs = signs_all[:NLeg - m]
        
        if there_is_beam_source:
            degree_tile = np.broadcast_to(ells[:, None], (NLeg - m, N + 1))
            lpmv_all = sc.special.lpmv(m, degree_tile, mu_arr_pos_mu0)
            asso_leg_term_pos = lpmv_all[:, :-1]
            scalar_fac_asso_leg_term_mu0 = I0_div_4pi * (2 - m_equals_0) * poch * lpmv_all[:, -1]
        else:
            degree_tile = np.broadcast_to(ells[:, None], (NLeg - m, N))
            asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
            
        asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
        fac_asso_leg_term_posT = poch[None, :] * asso_leg_term_pos.T
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
        
        # Diagonalization of coefficient matrix (refer to section 3.4.2 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        apb = alpha_arr + beta_arr
        amb = alpha_arr - beta_arr
        K_squared_arr, eigenvecs_GpG_arr = np.linalg.eig(
            np.einsum("lij, ljk -> lik", amb, apb, optimize=True)
        )
        
        # Eigenvalues arranged negative then positive, from largest to smallest magnitude
        K_arr_pos = np.sqrt(K_squared_arr)
        K_arr = np.concatenate((-K_arr_pos, K_arr_pos), axis=1)
        
        # Build eigenvector matrix from four blocks
        eigenvecs_GpG_arr /= 2
        eigenvecs_GmG_arr = np.einsum("lij, ljk -> lik", apb, eigenvecs_GpG_arr / K_arr_pos[:, None, :], optimize=True)
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
        
            # Solve for particular solution w.r.t. direct beam source (refer to section 3.6.1 of the Comprehensive Documentation)
            # --------------------------------------------------------------------------------------------------------------------------
            if len_no_shortcut_indices_0 > 0: # In this case we have `G_inv_collect_0`, so use it
                B_collect[no_shortcut_indices_0, :] = np.einsum(
                    "bij, bjk, bk -> bi",
                    # Note that the minus sign has been absorbed into `X_arr`
                    G_collect[no_shortcut_indices_0, :, :]
                    / (1 / mu0 + K_collect[no_shortcut_indices_0, None, :]),
                    G_inv_collect_0[no_shortcut_indices_0, :, :],
                    X_arr[no_shortcut_indices_0, :],
                    optimize=True,
                )
                no_shortcut_indices = no_shortcut_indices[len_no_shortcut_indices_0:]
            
            if len(no_shortcut_indices) > 0:
                A_arr = A_arr[len_no_shortcut_indices_0:]
                A_arr[:, :N, N:] = -A_arr[:, N:, :N]
                A_arr[:, :N, :N] = -A_arr[:, N:, N:]
                A_arr.reshape(-1, NQuad * NQuad)[:, ::NQuad + 1] += 1 / mu0
                # Note that the minus sign has been absorbed into `X_arr`
                B_collect[no_shortcut_indices, :] = np.linalg.solve(A_arr, X_arr[no_shortcut_indices, :, None]).squeeze(-1)
            
            # --------------------------------------------------------------------------------------------------------------------------

    outputs = (
        G_collect.reshape((NFourier, NLayers, NQuad, NQuad)),
        K_collect.reshape((NFourier, NLayers, NQuad)),
    )
    if there_is_beam_source:
        outputs += (B_collect.reshape((NFourier, NLayers, NQuad)),)
    if there_is_iso_source:
        outputs += (G_inv_collect_0,)
    return outputs