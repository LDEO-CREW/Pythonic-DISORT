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
    Leg_coeffs_BDRF,
    mathscr_vs,
    parfor_Fourier,
    mu_arr_pos, weights_mu,
    scale_tau
):  
    """Radiative transfer solver for a multi-layer atmosphere but excludes corrections and input checks.
    
    This function is wrapped by the `pydisort` function and should be called through `pydisort`,
    it has many seemingly redundant arguments to maximize precomputation in `pydisort`.
    We do not have an indepth docstring for this function but documentation can be found in the accompanying Jupyter Notebook,
    and many of its arguments are shared with  the `pydisort` function.
    
    """
    # If we want to solve for the intensity we need to solve for NLeg Fourier modes
    # If we only want to solve for the flux we only need to solve for the 0th Fourier mode
    if not only_flux:
        GC_collect = np.empty((NQuad, NLoops, NQuad))
        eigenvals_collect = np.empty((NQuad, NLoops))
        B_collect = np.empty((NQuad, NLoops))

    # Loop over NLoops Fourier modes
    for m in range(NLoops):
        ells = np.arange(m, NLeg)
        degree_tile = np.tile(ells, (N, 1)).T
        
        D_pos, D_neg = PyDISORT.subroutines.generate_Ds(m, Leg_coeffs, mu_arr_pos, w0, ells, degree_tile)
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
        X_pos, X_neg = PyDISORT.subroutines.generate_Xs(m, Leg_coeffs, w0, mu0, I0, mu_arr_pos, ells, degree_tile)
        X_tilde = np.concatenate((-M_inv * X_pos, M_inv * X_neg))
        B = np.linalg.solve(-(np.eye(NQuad) / mu0 + A), X_tilde)
        
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