import scipy as sc
from math import pi
from numpy.polynomial.legendre import leggauss
try:
    import autograd.numpy as np
except ImportError:
    import numpy as np


def transform_interval(arr, c, d, a=-1, b=1):
    """Affine transformation of an array from interval [a, b] to [c, d].

    Parameters
    ----------
    arr : array
        The 1D array to transform.
    c : float
        The beginning of interval [c, d].
    d : float
        The end of interval [c, d].
    a : float, optional
        The beginning of interval [a, b].
    b : float, optional
        The end of interval [a, b].

    Returns
    -------
    array
        The transformed 1D array.

    """
    return (((arr - a) * (d - c)) / (b - a)) + c


def transform_weights(weights, c, d, a=-1, b=1):
    """Transforms quadrature weights from interval [a, b] to [c, d].

    Parameters
    ----------
    weights : array
        The weights to transform.
    c : float
        The beginning of interval [c, d].
    d : float
        The end of interval [c, d].
    a : float, optional
        The beginning of interval [a, b].
    b : float, optional
        The end of interval [a, b].

    Returns
    -------
    array
        The transformed weights.

    """
    return weights * (d - c) / (b - a)


def calculate_nu(mu, phi, mu_p, phi_p):
    """Calculates the cosine of the scattering angle nu 
    between incident angle (mu_p, phi_p) and scattering angle (mu, phi).

    Parameters
    ----------
    mu : array
        Cosine of outgoing polar angles.
    phi : array
        Outgoing azimuthal angles.
    mu_p : array
        Cosine of incident polar angles.
    phi_p : array
        Incident azimuthal angles.

    Returns
    -------
    ndarray
        Cosine of scattering angles which axes capture variation with mu, phi, mu_p, phi_p respectively.

    """
    mu, phi, mu_p, phi_p = np.atleast_1d(mu, phi, mu_p, phi_p)
    nu = mu_p[None, None, :, None] * mu[:, None, None, None] + np.sqrt(1 - mu_p**2)[
        None, None, :, None
    ] * np.sqrt(1 - mu**2)[:, None, None, None] * np.cos(
        phi_p[None, None, None, :] - phi[None, :, None, None]
    )
    return np.squeeze(nu)


def Gauss_Legendre_quad(N, c=0, d=1):
    """Generates Gauss-Legendre quadrature zero points and weights for integration from c to d.

    Parameters
    ----------
    N : int
        Number of quadrature nodes.
    c : float, optional
        Start of integration interval.
    d : float, optional
        End of integration interval.

    Returns
    -------
    array
        Quadrature zero points.
    array
        Quadrature weights.

    """
    mu_arr_pos, W = leggauss(int(N))

    return transform_interval(mu_arr_pos, c, d), transform_weights(W, c, d)


def Clenshaw_Curtis_quad(Nphi, c=0, d=(2 * pi)):
    """Generates Clenshaw_Curtis quadrature weights and zero points for integration from c to d.

    Parameters
    ----------
    Nphi : int
        Number of quadrature nodes.
    c : float, optional
        Start of integration interval.
    d : float, optional
        End of integration interval.

    Returns
    -------
    array
        Quadrature zero points.
    array
        Quadrature weights.

    """
    # Ensure that the number of nodes is odd and greater than 2
    assert Nphi > 2
    assert Nphi % 2 == 1

    Nphi -= 1  # The extra index corresponds to the point 0 which we will add later
    Nphi_pos = Nphi // 2
    phi_arr_pos = np.cos(pi * np.arange(Nphi_pos) / Nphi)
    phi_arr = np.concatenate([-phi_arr_pos, [0], np.flip(phi_arr_pos)])
    diff = np.concatenate([[2], 2 / (1 - 4 * np.arange(1, Nphi_pos + 1) ** 2)])
    weights_phi_pos = sc.fft.idct(diff, type=1)
    weights_phi_pos[0] /= 2
    full_weights_phi = np.hstack((weights_phi_pos, np.flip(weights_phi_pos[:-1])))

    return transform_interval(phi_arr, c, d), transform_weights(full_weights_phi, c, d)


def generate_FD_mat(Ntau, a, b):
    """Generates a sparse first derivative matrix in `csr` format with second-order accuracy
    on [a,b] with `Ntau` grid points.
    We use second order forward or backward differences at the boundaries.

    Parameters
    ----------
    Nphi : int
        Number of grid nodes.
    a : float, optional
        Start of diffentiation interval.
    b : float, optional
        End of diffentiation interval.

    Returns
    -------
    array
        The differentiation grid.
    sparse 2darray
        The finite difference matrix.

    """
    tau_arr = np.linspace(a, b, Ntau)
    h = tau_arr[1] - tau_arr[0]

    diagonal = np.ones(Ntau) / (2 * h)
    first_deriv = sc.sparse.diags(diagonal[:-1], 1, format="lil")
    first_deriv.setdiag(-diagonal[:-1], -1)

    first_deriv[0, 0] = -3 / (2 * h)
    first_deriv[0, 1] = 2 / h
    first_deriv[0, 2] = -1 / (2 * h)
    first_deriv[-1, -1] = 3 / (2 * h)
    first_deriv[-1, -2] = -2 / h
    first_deriv[-1, -3] = 1 / (2 * h)

    return tau_arr, first_deriv.asformat("csr")
  

def atleast_2d_append(*arys):
    """View inputs as arrays with at least two dimensions. 
    Dimensions are added, when necessary, to the back of the shape tuple rather than to the front.
        
        This is exactly NumPy's `atleast_2d` function but altered to add dimensions to the back of the shape tuple rather than to the front.
        View the documentation for NumPy's `atleast_2d` function at https://numpy.org/doc/stable/reference/generated/numpy.atleast_2d.html.

    Parameters
    ----------
    arys1, arys2, ... : array_like
        One or more array-like sequences.  Non-array inputs are converted
        to arrays.  Arrays that already have two or more dimensions are
        preserved.

    Returns
    -------
    res, res2, ... : ndarray
        An array, or list of arrays, each with ``a.ndim >= 2``.
        Copies are avoided where possible, and views with two or more
        dimensions are returned.

    """
    res = []
    for ary in arys:
        ary = np.asanyarray(ary)
        if ary.ndim == 0:
            result = ary.reshape(1, 1)
        elif ary.ndim == 1:
            result = ary[:, np.newaxis]
        else:
            result = ary
        res.append(result)
    if len(res) == 1:
        return res[0]
    else:
        return res
        
        
def generate_diff_act_flux_funcs(u0):
    """Generates respectively the up and down diffuse actinic flux functions.
    This a use case of the u0 function that is an output of pydisort.
    The reclassification of delta-scaled actinic flux is automatically performed.

    Parameters
    ----------
    u0 : func
        Zeroth Fourier mode of the intensity.
        See the fourth "return" of the `pydisort` function.

    Returns
    -------
    function
        Actinic flux function with argument tau (type: array) for positive (upward) mu values.
        Returns the diffuse flux magnitudes (type: array).
    function
        Actinic flux function with argument tau (type: array) for negative (downward) mu values.
        Returns the diffuse flux magnitudes (type: array).

    """
    N = np.shape(u0(0))[0] // 2
    GL_weights = Gauss_Legendre_quad(N, 0, 1)[1]

    # Note that the zeroth axis of the array u0(tau) captures variation with mu
    flux_act_up = lambda tau: 2 * pi * GL_weights @ u0(tau)[:N]
    def flux_act_down_diffuse(tau):
        u0_cache, act_dscale_reclassification = u0(tau, True)
        result_without_reclassification = 2 * pi * GL_weights @ u0_cache[N:]
        return result_without_reclassification + act_dscale_reclassification
    
    return flux_act_up, flux_act_down_diffuse


def _mathscr_v(tau, l, s_poly_coeffs, Nscoeffs, G, K, G_inv, mu_arr):
    """Particular solution for isotropic internal sources.
    It has many seemingly redundant arguments to maximize precomputation
    in the `pydisort` function which calls it.

    """
    n = Nscoeffs - 1

    def mathscr_b(i):
        j = np.arange(i + 1)
        s_poly_coeffs_nj = s_poly_coeffs[:, n - j]
        return np.sum(
            (sc.special.factorial(n - j) / sc.special.factorial(n - i))[None, None, :]
            * K[:, :, None] ** -(i - j + 1)[None, None, :]
            * s_poly_coeffs_nj[:, None, :],
            axis=-1,
        )

    mathscr_v_coeffs = np.array(list(map(mathscr_b, range(Nscoeffs))))
    return np.einsum(
        "tik, tc, ctk, tkj, j -> it",
        G[l, :, :],
        tau[:, None] ** np.flip(np.arange(Nscoeffs))[None, :],
        mathscr_v_coeffs[:, l, :],
        G_inv[l, :, :],
        1 / mu_arr,
        optimize=True,
    )
    
    
def _compare(results, mu_to_compare, reorder_mu, flux_up, flux_down, u):
    """Performs pointwise comparisons between results from Stamnes' DISORT,
    which are stored in .npz files, against results from PythonicDISORT. Used in our PyTests.

    """
    uu = results["uu"]
    flup = results["flup"]
    rfldn = results["rfldn"]
    rfldir = results["rfldir"]

    # Load comparison points
    tau_test_arr = results["tau_test_arr"]
    phi_arr = results["phi_arr"]

    # Perform and print the comparisons
    # --------------------------------------------------------------------------------------------------
    print("Max pointwise differences")
    print()

    # Upward (diffuse) fluxes
    print("Upward (diffuse) fluxes")
    diff_flux_up = np.abs(flup - flux_up(tau_test_arr))
    ratio_flux_up = np.divide(
        diff_flux_up,
        flup,
        out=np.zeros_like(diff_flux_up),
        where=flup > 1e-8,
    )
    print("Difference =", np.max(diff_flux_up))
    print("Difference ratio =", np.max(ratio_flux_up))
    print()

    # Downward (diffuse) fluxes
    print("Downward (diffuse) fluxes")
    diff_flux_down_diffuse = np.abs(rfldn - flux_down(tau_test_arr)[0])
    ratio_flux_down_diffuse = np.divide(
        diff_flux_down_diffuse,
        rfldn,
        out=np.zeros_like(diff_flux_down_diffuse),
        where=rfldn > 1e-8,
    )
    print("Difference =", np.max(diff_flux_down_diffuse))
    print(
        "Difference ratio =",
        np.max(ratio_flux_down_diffuse),
    )
    print()

    # Direct (downward) fluxes
    print("Direct (downward) fluxes")
    diff_flux_down_direct = np.abs(rfldir - flux_down(tau_test_arr)[1])
    ratio_flux_down_direct = np.divide(
        diff_flux_down_direct,
        rfldir,
        out=np.zeros_like(diff_flux_down_direct),
        where=rfldir > 1e-8,
    )
    print("Difference =", np.max(diff_flux_down_direct))
    print(
        "Difference ratio =",
        np.max(ratio_flux_down_direct),
    )
    print()

    # Intensity
    diff = np.abs(uu - u(tau_test_arr, phi_arr)[reorder_mu])[mu_to_compare]
    diff_ratio = np.divide(
        diff,
        uu[mu_to_compare],
        out=np.zeros_like(diff),
        where=uu[mu_to_compare] > 1e-8,
    )
    max_diff_tau_index = np.argmax(np.max(np.max(diff, axis=0), axis=1))
    max_ratio_tau_index = np.argmax(np.max(np.max(diff_ratio, axis=0), axis=1))
    
    diff_tau_pt = tau_test_arr[max_diff_tau_index]
    ratio_tau_pt = tau_test_arr[max_ratio_tau_index]
    print("Intensities")
    print("At tau = " + str(diff_tau_pt))
    print("Max pointwise difference =", np.max(diff[:, max_diff_tau_index, :]))
    print("At tau = " + str(ratio_tau_pt))
    print("Max pointwise difference ratio =", np.max(diff_ratio[:, max_ratio_tau_index, :]))
    print()

    return (
        diff_flux_up,
        ratio_flux_up,
        diff_flux_down_diffuse,
        ratio_flux_down_diffuse,
        diff_flux_down_direct,
        ratio_flux_down_direct,
        diff,
        diff_ratio,
    )