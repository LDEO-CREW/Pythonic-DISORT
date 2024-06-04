import numpy as np

def basic_solver(
    b_pos, b_neg,
    only_flux,
    N, NQuad, NLeg,
    Leg_coeffs,
    mu_arr_pos, weights_mu,
    tau0, w0,
    mu0, phi0, I0, 
    scale_tau, scale_beam,
):  # This function has many redundant arguments to maximize precomputation in the wrapper function
    
    """Basic radiative transfer solver which performs no corrections
    
    :Input:
     - *b_pos / neg* (matrix) - Boundary conditions for the upward / downward directions
     - *only_flux* (boolean) - Flag for whether to compute the intensity function
     - *N* (integer) - Equals NQuad // 2
     - *NQuad* (integer) - Number of mu quadrature points
     - *NLeg* (integer) - Number of phase function Legendre coefficients to use in the basic solver
     - *Leg_coeffs* (vector) - Long vector of weighted phase function Legendre coefficients
     - *mu_arr_pos* (vector) - Positive mu (quadrature) values
     - *weights_mu* (vector) - Weights for mu quadrature
     - *tau0* (float) - Optical depth
     - *w0* (float) - Single-scattering albedo
     - *mu0* (float) - Polar angle of the direct beam
     - *phi0* (float) - Azimuthal angle of the direct beam
     - *I0* (float) - Intensity of the direct beam
     - *scale_tau* (float) - Delta-M scale factor for tau
     - *scale_beam* (float) - Delta-M scale factor for the direct beam
     
     
    :Output:
     - *flux_up* (function) - Flux function with argument tau for positive (upward) mu values
     - *flux_down* (vector) - Flux function with argument tau for negative (downward) mu values
     :Optional:
     - *u* (function) - Intensity function with arguments (tau, phi); the output is in the order (mu, tau, phi)
    """
    
    # If we want to solve for the intensity we need to solve for NLeg Fourier modes
    # If we only want to solve for the flux we only need to solve for the 0th Fourier mode
    if not only_flux:
        GC_collect = np.empty((NQuad, NLeg, NQuad))
        eigenvals_collect = np.empty((NQuad, NLeg))
        B_collect = np.empty((NQuad, NLeg))

    # Loop over the number of Fourier modes
    for m in range(NLeg):
        ells = np.arange(m, NLeg)
        degree_tile = np.tile(ells, (N, 1)).T
        
        D_pos, D_neg = generate_Ds(m, Leg_coeffs, mu_arr_pos, w0, ells, degree_tile)
        M_inv = 1 / mu_arr_pos
        W = weights_mu[None, :]
        alpha = M_inv[:, None] * (D_pos * W - np.eye(N))
        beta = M_inv[:, None] * D_neg * W
        # This is the discretization of the multiple scattering (integral) term
        A = np.vstack((np.hstack((-alpha, -beta)), np.hstack((beta, alpha))))

        eigenvals_squared, eigenvecs_GpG = np.linalg.eig(
            (alpha - beta) @ (alpha + beta)
        )
        # Eigenvalues arranged negative then positive, from largest absolute value to smallest
        eigenvals = np.concatenate((-np.sqrt(eigenvals_squared), np.sqrt(eigenvals_squared)))
        eigenvecs_GpG = np.hstack((eigenvecs_GpG, eigenvecs_GpG))
        eigenvecs_GmG = (alpha + beta) @ eigenvecs_GpG / -eigenvals
        
        # Eigenvectors
        G_pos = (eigenvecs_GpG + eigenvecs_GmG) / 2
        G_neg = (eigenvecs_GpG - eigenvecs_GmG) / 2
        G = np.vstack((G_pos, G_neg))
        
        # Solve for the coefficients of the particular solution, B
        X_pos, X_neg = generate_Xs(m, Leg_coeffs, w0, mu0, I0, mu_arr_pos, ells, degree_tile)
        X_tilde = np.concatenate((-M_inv * X_pos, M_inv * X_neg))
        B = np.linalg.solve(-(np.eye(NQuad) / mu0 + A), X_tilde)
        
        # Solve for the coefficients of the homogeneous solution
        B_pos, B_neg = B[:N], B[N:]
        K_tau0_neg = np.exp(eigenvals[:N] * tau0)

        LHS = np.vstack((G_neg, G_pos))
        LHS[:N, N:] *= K_tau0_neg[None, :]
        LHS[N:, :N] *= K_tau0_neg[None, :]
        RHS = np.concatenate((b_neg[:, m] - B_neg, b_pos[:, m] - B_pos * np.exp(-tau0 / mu0)))
        C = np.linalg.solve(LHS, RHS)

        if only_flux:
            return generate_flux_functions(
                I0, mu0, tau0,
                G_pos * C[None, :], G_neg * C[None, :],
                eigenvals, N,
                B_pos, B_neg,
                mu_arr_pos, weights_mu,
                scale_tau, scale_beam,
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
                    np.cos(np.arange(NLeg)[:, None] * (phi0 - phi)[None, :]),
                    optimize=True,
                )
            )

    return (u, ) + generate_flux_functions(
        I0, mu0, tau0,
        GC_collect[:N, 0, :], GC_collect[N:, 0, :],
        eigenvals_collect[:, 0], N,
        B_collect[:N, 0], B_collect[N:, 0],
        mu_arr_pos, weights_mu,
        scale_tau, scale_beam,
    )