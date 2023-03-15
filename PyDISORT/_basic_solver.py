from PyDISORT import subroutines

try:
    import autograd.numpy as np
except ImportError:
    import numpy as np
try:
    import parfor
except ImportError:
    None

def _basic_solver(
    tau_arr, omega_arr,
    N, NQuad, NLeg, NLoops,
    weighted_Leg_coeffs,
    mu0, I0, phi0,
    b_pos, b_neg,
    only_flux,
    NBDRF,
    weighted_Leg_coeffs_BDRF,
    mathscr_vs,
    parfor_Fourier,
    mu_arr_pos, weights_mu,
    scale_tau
):  
    """This function is wrapped by the `pydisort` function and should be called through `pydisort`.
    It has many seemingly redundant arguments to maximize precomputation in `pydisort`.
    We do not have an indepth docstring for this function but documentation can be found in the accompanying Jupyter Notebook,
    and many of its arguments are shared with the `pydisort` function.
    
    """
    # If we want to solve for the intensity we need to solve for NLoops Fourier modes
    # If we only want to solve for the flux we only need to solve for the 0th Fourier mode
    if not only_flux:
        GC_collect = np.empty((L, NQuad, NQuad, NLoops))
        eigenvals_collect = np.empty((L, NQuad, NLoops))
        B_collect = np.empty((L, NQuad, NLoops))
    else:
        GC_collect = np.empty((L, NQuad, NQuad))
        eigenvals_collect = np.empty((L, NQuad))
        B_collect = np.empty((L, NQuad))
        
    # Loop over NLoops Fourier modes
    for m in range(NLoops):
        # --------------------------------------------------------------------------------------------------------------------------
        # Generate D and X (phase function terms)
        # --------------------------------------------------------------------------------------------------------------------------
        ells = np.arange(m, NLeg)
        degree_tile = np.tile(ells, (N, 1)).T
        fac = np.prod(ells[:, None] + np.arange(-m + 1, m + 1)[None, :], axis=-1)
        asso_weights = np.divide(1, fac, out=np.zeros(NLeg - m), where=(fac != 0))
        signs = np.empty(NLeg - m)
        signs[::2] = 1
        signs[1::2] = -1
        if m == 0:
            prefactor = omegal * I0 / (4 * pi)
        else:
            prefactor = omegal * I0 / (2 * pi)

        asso_leg_term_pos = sc.special.lpmv(m, degree_tile, mu_arr_pos)
        asso_leg_term_neg = asso_leg_term_pos * signs[:, None]
        asso_leg_term_mu0 = sc.special.lpmv(m, ells, -mu0)
        weighted_asso_Leg_coeffs = weighted_Leg_coeffs[ells] * asso_weights

        D_temp = weighted_asso_Leg_coeffs[None, :] * asso_leg_term_pos.T
        D_pos = (omegal / 2) * D_temp @ asso_leg_term_pos
        D_neg = (omegal / 2) * D_temp @ asso_leg_term_neg

        X_temp = prefactor * weighted_asso_Leg_coeffs * asso_leg_term_mu0
        X_pos = X_temp @ asso_leg_term_pos
        X_neg = X_temp @ asso_leg_term_neg
        # --------------------------------------------------------------------------------------------------------------------------
        # Generate mathscr_D and mathscr_X (BDRF terms)
        # --------------------------------------------------------------------------------------------------------------------------
        if m < NBDRF:
            if m == 0:
                prefactor = mu0 * I0 / pi
            else:
                prefactor = mu0 * I0 * 2 / pi

            asso_leg_term_pos = asso_leg_term_pos[:NBDRF, :]
            asso_leg_term_neg = asso_leg_term_neg[:NBDRF, :]
            asso_leg_term_mu0 = asso_leg_term_mu0[:NBDRF]
            weighted_asso_Leg_coeffs = (
                weighted_Leg_coeffs_BDRF[ells[: (NBDRF - m)]] * asso_weights[: (NBDRF - m)]
            )

            mathscr_D_temp = (
                weighted_asso_Leg_coeffs[None, :] * asso_leg_term_pos.T * mu_arr_pos[:, None]
            )
            mathscr_D_neg = 2 * mathscr_D_temp @ asso_leg_term_neg

            mathscr_X_temp = prefactor * weighted_asso_Leg_coeffs * asso_leg_term_mu0
            mathscr_X_pos = mathscr_X_temp @ asso_leg_term_pos
        # --------------------------------------------------------------------------------------------------------------------------
        # Assemble the coefficient matrix and additional terms
        # --------------------------------------------------------------------------------------------------------------------------
        M_inv = 1 / mu_arr_pos
        W = weights_mu[None, :]
        alpha = M_inv[:, None] * (D_pos * W - np.eye(N))
        beta = M_inv[:, None] * D_neg * W
        A = np.vstack((np.hstack((-alpha, -beta)), np.hstack((beta, alpha))))

        weighted_mathscr_D_neg = mathscr_D_neg * W
        X_tilde = np.concatenate((-M_inv * X_pos, M_inv * X_neg))
        # --------------------------------------------------------------------------------------------------------------------------
        # Diagonalization of coefficient matrix
        # --------------------------------------------------------------------------------------------------------------------------
        ##### Computation of G^{-1} #####
        K_squared, eigenvecs_GpG = np.linalg.eig((alpha - beta) @ (alpha + beta))

        # Eigenvalues arranged negative then positive, from largest to smallest magnitude
        K = np.concatenate((-np.sqrt(K_squared), np.sqrt(K_squared)))
        eigenvecs_GpG = np.hstack((eigenvecs_GpG, eigenvecs_GpG))
        eigenvecs_GmG = (alpha + beta) @ eigenvecs_GpG / K

        # Eigenvector matrix
        G_pos = (eigenvecs_GpG - eigenvecs_GmG) / 2
        G_neg = (eigenvecs_GpG + eigenvecs_GmG) / 2
        G = np.vstack((G_pos, G_neg))
        
        ##### Computation of H^{-1} #####
        eigenvecs_HpH = np.linalg.eig(((alpha - beta) @ (alpha + beta)).T)[1]
        eigenvecs_HpH = np.hstack((eigenvecs_HpH, eigenvecs_HpH))
        eigenvecs_HmH = (alpha - beta).T @ eigenvecs_HpH / K

        # Inverse of eigenvector matrix
        H_pos = (eigenvecs_HpH - eigenvecs_HmH) / 2
        H_neg = (eigenvecs_HpH + eigenvecs_HmH) / 2
        G_inv = np.vstack((H_pos, H_neg)).T
        G_inv = G_inv / np.diag(G_inv @ G)[:, None]
        
        # Solve for the coefficients of the particular solution, B
        X_pos, X_neg = PyDISORT.subroutines.generate_Xs(m, Leg_coeffs, w0, mu0, I0, mu_arr_pos, ells, degree_tile)
        X_tilde = np.concatenate((-M_inv * X_pos, M_inv * X_neg))
        # -------------------------------------------
        # Particular solution for the sunbeam source
        # -------------------------------------------
        B = np.linalg.solve(-(np.eye(NQuad) / mu0 + A), X_tilde) # coefficient vector
        # --------------------------------------------------------------------------------------------------------------------------
        
        
    # Solve for the coefficients of the homogeneous solution
    B_pos, B_neg = B[:N], B[N:]
    K_tau0_neg = np.exp(eigenvals[:N] * tau0)

    LHS = np.vstack((G_neg, G_pos)) # LHS is a re-arranged COPY of matrix G
    LHS[:N, N:] *= K_tau0_neg[None, :]
    LHS[N:, :N] *= K_tau0_neg[None, :]
    RHS = np.concatenate((b_neg[:, m] - B_neg, b_pos[:, m] - B_pos * np.exp(-tau0 / mu0)))
    C = np.linalg.solve(LHS, RHS)

    if only_flux:
        return PyDISORT.subroutines.generate_flux_functions(
            mu0, I0, tau0,
            G_pos * C[None, :], G_neg * C[None, :],
            eigenvals, N,
            B_pos, B_neg,
            mu_arr_pos, weights_mu,
            scale_tau,
        )
    
    GC_collect[:, m, :] = G * C[None, :]
    eigenvals_collect[:, m] = eigenvals
    B_collect[:, m] = B
    
    # Construct the intensity function
    def u(tau, phi):
        tau = scale_tau * np.atleast_1d(tau) # Delta-M scaling
        phi = np.atleast_1d(phi)
        exponent = np.concatenate(
            (
                eigenvals_collect[:N, :, None] * tau[None, None, :],
                eigenvals_collect[N:, :, None] * (tau - tau0)[None, None, :],
            ),
            axis=0,
        )
        um = np.einsum(
            "imj, jmt -> imt", GC_collect, np.exp(exponent), optimize=True
        ) + B_collect[:, :, None] * np.exp(-tau[None, None, :] / mu0)
        return np.squeeze(
            np.einsum(
                "imt, mp -> itp",
                um,
                np.cos(np.arange(NLoops)[:, None] * (phi0 - phi)[None, :]),
                optimize=True,
            )
        )

    return PyDISORT.subroutines.generate_flux_functions(
        mu0, I0, tau0,
        GC_collect[:N, 0, :], GC_collect[N:, 0, :],
        eigenvals_collect[:, 0], N,
        B_collect[:N, 0], B_collect[N:, 0],
        mu_arr_pos, weights_mu,
        scale_tau,
    ) + (u, )