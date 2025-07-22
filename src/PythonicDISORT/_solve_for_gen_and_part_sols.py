import numpy as np
import scipy as sc
from math import pi


def _solve_for_gen_and_part_sols(
    NFourier,                    # Number of intensity Fourier modes
    scaled_omega_arr,            # Delta-scaled single-scattering albedos
    mu_arr_pos, mu_arr,          # Quadrature nodes for 1) upper 2) both hemispheres
    M_inv, W,                    # 1) 1 / mu; 2) quadrature weights for each hemisphere
    N, NQuad, NLeg,              # Number of 1) upper 2) both hemispheres quadrature nodes; 3) phase function Legendre coefficients 
    NLayers,                     # Number of layers
    weighted_scaled_Leg_coeffs,  # Weighted and delta-scaled Legendre coefficients
    mu0, I0,                     # Properties of the direct beam
    there_is_beam_source,        # Is there a beam source?
    Nscoeffs,                    # Number of isotropic source polynomial coefficients
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
    | `mu_arr`                       | `NQuad`                            |
    | `M_inv`                        | `NQuad/2`                          |
    | `W`                            | `NQuad/2`                          |
    | `N`                            | scalar                             |
    | `NQuad`                        | scalar                             |
    | `NLeg`                         | scalar                             |
    | `NLayers`                      | scalar                             |
    | `weighted_scaled_Leg_coeffs`   | `NLayers x NLeg`                   |
    | `mu0`                          | scalar                             |
    | `I0`                           | scalar                             |
    | `there_is_beam_source`         | boolean                            |
    | `Nscoeffs`                     | scalar                             |
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
    ind = 0
    
    G_collect = np.empty((NFourier * NLayers, NQuad, NQuad))
    K_collect = np.empty((NFourier * NLayers, NQuad))
    alpha_arr = np.empty((NFourier * NLayers, N, N))
    beta_arr = np.empty((NFourier * NLayers, N, N))
    no_shortcut_indices = []
    no_shortcut_indices_0 = []
    
    if there_is_beam_source:
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
        degree_tile = np.tile(ells, (N, 1)).T
        fac = sc.special.poch(ells + m + 1, -2 * m)
        signs = np.ones(NLeg - m)
        signs[1::2] = -1

        asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
        asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
        asso_leg_term_mu0 = sc.special.lpmv(m, ells, -mu0)
        # --------------------------------------------------------------------------------------------------------------------------

        # Loop over NLayers atmospheric layers
        # --------------------------------------------------------------------------------------------------------------------------          
        for l in range(NLayers):
            weighted_scaled_Leg_coeffs_l = weighted_scaled_Leg_coeffs[l, :][ells]
            scaled_omega_l = scaled_omega_arr[l]
            omega_times_Leg_coeffs = (scaled_omega_l / 2) * weighted_scaled_Leg_coeffs_l
            
            if np.any(np.abs(omega_times_Leg_coeffs) > 1e-8): # There are shortcuts if multiple scattering is insignificant
                
                # Generate D
                # --------------------------------------------------------------------------------------------------------------------------
                D_temp = omega_times_Leg_coeffs[None, :] * (fac[None, :] * asso_leg_term_pos.T)
                D_pos = D_temp @ asso_leg_term_pos
                D_neg = D_temp @ asso_leg_term_neg
                
                # --------------------------------------------------------------------------------------------------------------------------
                

                # Assemble the coefficient matrix and additional terms
                # --------------------------------------------------------------------------------------------------------------------------
                DW = D_pos * W[None, :]
                np.fill_diagonal(DW, np.diag(DW) - 1)
                alpha = M_inv[:, None] * DW
                beta = M_inv[:, None] * D_neg * W[None, :]
                
                # --------------------------------------------------------------------------------------------------------------------------
                
                # Particular solution for the direct beam source (refer to section 3.6.1 of the Comprehensive Documentation)
                # --------------------------------------------------------------------------------------------------------------------------
                if there_is_beam_source:
                    # Generate X
                    X_temp = (
                        (scaled_omega_l * I0 * (2 - (m == 0)) / (4 * pi))
                        * weighted_scaled_Leg_coeffs_l
                        * (fac * asso_leg_term_mu0)
                    )
                    X_pos = X_temp @ asso_leg_term_pos
                    X_neg = X_temp @ asso_leg_term_neg
                    X_tilde = np.concatenate([-M_inv * X_pos, M_inv * X_neg])
                    
                    A = np.concatenate(
                        [
                            np.concatenate([-alpha, -beta], axis=1),
                            np.concatenate([beta, alpha], axis=1),
                        ],
                        axis=0,
                    )
                    np.fill_diagonal(A, np.diag(A) + 1 / mu0)
                    B_collect[ind, :] = -np.linalg.solve(A, X_tilde) # We moved the minus sign out
                    
                # --------------------------------------------------------------------------------------------------------------------------
                
                alpha_arr[ind, :, :] = alpha
                beta_arr[ind, :, :] = beta
                no_shortcut_indices.append(ind)
                if m_equals_0 and there_is_iso_source: # Keep the list empty if there is no isotropic source
                    no_shortcut_indices_0.append(l)
                
            else:
                # This is a shortcut to the diagonalization results
                G = np.zeros((NQuad, NQuad))
                np.fill_diagonal(G[N:, :N], 1)
                np.fill_diagonal(G[:N, N:], 1)
                
                G_collect[ind, :, :] = G
                K_collect[ind, :] = -1 / mu_arr
                if m_equals_0 and there_is_iso_source:
                    G_inv_collect_0[l, :, :] = G
                    
            ind += 1
                
    if len(no_shortcut_indices) > 0:
    
        # Diagonalization of coefficient matrix (refer to section 3.4.2 of the Comprehensive Documentation)
        # --------------------------------------------------------------------------------------------------------------------------
        alpha_arr = alpha_arr[no_shortcut_indices, :, :]
        beta_arr = beta_arr[no_shortcut_indices, :, :]

        K_squared_arr, eigenvecs_GpG_arr = np.linalg.eig(
            np.einsum(
                "lij, ljk -> lik", alpha_arr - beta_arr, alpha_arr + beta_arr, optimize=True
            ),
        )
        
        # Eigenvalues arranged negative then positive, from largest to smallest magnitude
        K_arr = np.concatenate((-np.sqrt(K_squared_arr), np.sqrt(K_squared_arr)), axis=1)
        eigenvecs_GpG_arr = np.concatenate((eigenvecs_GpG_arr, eigenvecs_GpG_arr), axis=2)
        eigenvecs_GmG_arr = (
            np.einsum(
                "lij, ljk -> lik", alpha_arr + beta_arr, eigenvecs_GpG_arr, optimize=True
            )
            / K_arr[:, None, :]
        )

        # Eigenvector matrices
        G_arr = np.concatenate(
            (
                (eigenvecs_GpG_arr - eigenvecs_GmG_arr) / 2,
                (eigenvecs_GpG_arr + eigenvecs_GmG_arr) / 2,
            ),
            axis=1,
        )
        
        G_collect[no_shortcut_indices, :, :] = G_arr
        K_collect[no_shortcut_indices, :] = K_arr
        if len(no_shortcut_indices_0) > 0: # If there is no isotropic source this list will be empty
            G_inv_collect_0[no_shortcut_indices_0, :, :] = np.linalg.inv(
                G_collect[no_shortcut_indices_0, :, :]
            )

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