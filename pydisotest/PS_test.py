import numpy as np
from scipy.integrate import quad
import PythonicDISORT
from math import pi
import pytest

# ============================================================================================================
# Test Problem PS: Pseudo-spherical correction (`radii_arr`)
# ============================================================================================================
# This subsumes test problem 16 of Stamnes' DISORT (`disotest.f90`), see case C of `test_PS_vs_Stamnes_DISORT`

# Six layers between an altitude of 60 km and the surface of the Earth, with the sun 5 degrees above the horizon
earth_radius = 6371
altitudes = np.array([60, 40, 25, 15, 8, 3, 0])
radii_arr = earth_radius + altitudes
tau_arr = np.array([0.05, 0.15, 0.4, 0.9, 1.6, 2.5])
NLayers = len(tau_arr)
omega_arr = np.array([0.99, 0.95, 0.9, 0.97, 0.85, 0.9])
NQuad = 16
mu0 = np.cos(85 * pi / 180)
I0 = 1
phi0 = 1
albedo = 0.2

# A: Rayleigh-like scattering; B: Henyey-Greenstein scattering
Leg_coeffs_all_A = np.zeros((NLayers, NQuad + 1))
Leg_coeffs_all_A[:, 0] = 1
Leg_coeffs_all_A[:, 2] = 0.1
Leg_coeffs_all_B = np.array([0.2, 0.5, 0.7, 0.8, 0.75, 0.6])[:, None] ** np.arange(NQuad + 1)[None, :]

tau_test_arr = np.array([0.02, 0.1, 0.3, 0.7, 1.2, 2.3])  # None of which is on a layer boundary
phi_arr = np.array([0, 1, 2.5])
mu_test_arr = np.array([-0.95, -0.6, -0.25, -0.05, 0.05, 0.3, 0.7, 1])

# C: test problem 16 of Stamnes' DISORT: 13 Rayleigh layers at 412 nm between an altitude of 70 km and a black
# surface, nearly conservative scattering, 30 streams, the same sun
radii_arr_C = earth_radius + np.array([70, 50, 40, 30, 24, 20, 16, 12, 10, 8, 6, 4, 2, 0])
tau_arr_C = np.cumsum([
    0.234663486e-03, 0.765681267e-03, 0.315219164e-02, 0.598260760e-02, 0.857555866e-02, 0.162254870e-01,
    0.301396251e-01, 0.226671547e-01, 0.286127925e-01, 0.357458740e-01, 0.442614332e-01, 0.543503761e-01,
    0.665201172e-01,
])
NQuad_C = 30
Leg_coeffs_all_C = np.zeros((len(tau_arr_C), NQuad_C + 1))
Leg_coeffs_all_C[:, 0] = 1
Leg_coeffs_all_C[:, 2] = 0.47821149 / 5


def _pydisort_C(**kwargs):
    kwargs.setdefault("radii_arr", radii_arr_C)
    return PythonicDISORT.pydisort(
        tau_arr_C, np.full(len(tau_arr_C), 0.99999),
        NQuad_C,
        Leg_coeffs_all_C,
        mu0, I0, 0,
        **kwargs,
    )


def _pydisort(Leg_coeffs_all, **kwargs):
    kwargs.setdefault("radii_arr", radii_arr)
    return PythonicDISORT.pydisort(
        tau_arr, omega_arr,
        NQuad,
        Leg_coeffs_all,
        mu0, I0, phi0,
        f_arr=Leg_coeffs_all[:, NQuad],
        BDRF_Fourier_modes=[albedo],
        **kwargs,
    )


def test_PS_vs_Stamnes_DISORT():
    # Stamnes' DISORT (single precision) with DO_PSEUDO_SPHERE = TRUE, EARTH_RADIUS = `earth_radius`,
    # H_LYR = `altitudes`, a Lambertian surface and USRANG = TRUE.
    # Stamnes' DISORT always delta-M scales with the moment of degree NSTR, which is zero in cases A and C.
    # Case C is test problem 16 of Stamnes' DISORT, which evaluates only the upward intensities at the top of the
    # atmosphere; here both hemispheres (including the direction of the beam), depths down to the surface and the
    # fluxes are compared too.
    print()
    results = np.load("Stamnes_results/PS_test.npz")
    rel_diff = lambda ours, theirs: np.max(np.abs(ours - theirs)) / np.max(np.abs(theirs))

    solvers = {
        "A": lambda **kwargs: _pydisort(Leg_coeffs_all_A, **kwargs),
        "B": lambda **kwargs: _pydisort(Leg_coeffs_all_B, **kwargs),
        "C": _pydisort_C,
    }
    for case, solve in solvers.items():
        grid = "C" if case == "C" else ""
        umu, phi, utau = results["umu" + grid], results["phi" + grid], results["utau" + grid]
        mu_arr, flux_up, flux_down, u0, u = solve()
        u_at_mu = solve(interpolate_mu=True)[-1]
        flup, rfldn, rfldir = results["flup" + case], results["rfldn" + case], results["rfldir" + case]

        diffs = {
            "upward flux": rel_diff(flux_up(utau), flup),
            "total downward flux": rel_diff(np.sum(flux_down(utau), axis=0), rfldn + rfldir),
            "direct flux": rel_diff(flux_down(utau)[1], rfldir),
            "diffuse downward flux": rel_diff(flux_down(utau)[0], rfldn),
            "intensity at the quadrature nodes": rel_diff(u(utau, phi)[np.argsort(mu_arr)], results["uu_nodes" + case]),
            "intensity at other polar angles": rel_diff(u_at_mu(umu, utau, phi), results["uu" + case]),
        }
        for name, diff in diffs.items():
            print("Case " + case + ", " + name + ": max difference / max =", diff)

        # With delta-M scaling (case B), Stamnes' DISORT attenuates the direct beam with the layer cosines of the
        # delta-scaled atmosphere, whereas PythonicDISORT uses the slant optical depth of the unscaled atmosphere.
        # The total downward flux is unaffected.
        # In case C (nearly conservative scattering, 30 streams) the upward flux of Stamnes' DISORT differs by up to
        # 3e-5 of its maximum from the flux obtained by integrating its own intensities at the quadrature nodes,
        # which PythonicDISORT matches to 8e-6; this single-precision error sets the largest difference.
        tol_direct = 5e-4 if case == "B" else 5e-5
        assert diffs["upward flux"] < 5e-5
        assert diffs["total downward flux"] < 5e-5
        assert diffs["direct flux"] < tol_direct
        assert diffs["diffuse downward flux"] < tol_direct
        assert diffs["intensity at the quadrature nodes"] < 5e-5
        assert diffs["intensity at other polar angles"] < 5e-5

        # The plane-parallel solution is far from these results, i.e. the correction matters for this test
        assert rel_diff(solve(radii_arr=None)[1](utau), flup) > 5e-2


def test_PS_plane_parallel_limits():
    # The correction vanishes as the radii grow, and for an overhead sun
    print()
    for kwargs in ({}, {"interpolate_mu": True}):
        args = (mu_test_arr,) if kwargs else ()
        for mu0_test, radii_test in ((mu0, 1e12 * earth_radius + altitudes), (1, radii_arr)):
            outputs = []
            for radii in (None, radii_test):
                outputs.append(
                    PythonicDISORT.pydisort(
                        tau_arr, omega_arr,
                        NQuad,
                        Leg_coeffs_all_B,
                        mu0_test, I0, phi0,
                        f_arr=Leg_coeffs_all_B[:, NQuad],
                        BDRF_Fourier_modes=[albedo],
                        radii_arr=radii,
                        **kwargs,
                    )
                )
            (_, Fp, Fm, _, u), (_, Fp_PS, Fm_PS, _, u_PS) = outputs
            for tau_derivative_order in (0, -1, 1):
                assert np.allclose(Fp_PS(tau_test_arr, tau_derivative_order), Fp(tau_test_arr, tau_derivative_order), rtol=1e-8)
                assert np.allclose(Fm_PS(tau_test_arr, tau_derivative_order), Fm(tau_test_arr, tau_derivative_order), rtol=1e-8)
                ref = u(*args, tau_test_arr, phi_arr, tau_derivative_order)
                res = u_PS(*args, tau_test_arr, phi_arr, tau_derivative_order)
                assert np.max(np.abs(res - ref)) < 1e-8 * np.max(np.abs(ref))


def test_PS_direct_beam():
    # The direct flux is mu0 * I0 * exp(-slant optical depth); at the layer boundaries the slant optical depth
    # is the sum over the layers above of their optical thickness times the slant-to-vertical ratio of the path
    # of the beam across them, which is computed here from the geometry of the spherical shells
    print()
    flux_down = _pydisort(Leg_coeffs_all_B)[2]
    thickness_arr = np.diff(tau_arr, prepend=0)
    for l in range(NLayers):
        closest = radii_arr[l + 1] * np.sqrt(1 - mu0**2)  # Closest distance of the beam to the centre of the Earth
        along_beam = np.sqrt(radii_arr[: l + 2] ** 2 - closest**2)
        slant_tau = np.sum(thickness_arr[: l + 1] * -np.diff(along_beam) / -np.diff(radii_arr[: l + 2]))
        assert np.isclose(flux_down(tau_arr[l])[1], mu0 * I0 * np.exp(-slant_tau), rtol=1e-10, atol=0)
    assert np.isclose(flux_down(0)[1], mu0 * I0, rtol=1e-13, atol=0)


def test_PS_derivatives():
    # The tau-derivative against central finite differences, and the tau-integral over each layer
    # (telescoping idiom) against adaptive quadrature
    print()
    mu_arr, flux_up, flux_down, u0, u = _pydisort(Leg_coeffs_all_B)
    u_at_mu = _pydisort(Leg_coeffs_all_B, interpolate_mu=True)[-1]
    funcs = (
        flux_up,
        lambda tau, *args: flux_down(tau, *args)[0],
        lambda tau, *args: flux_down(tau, *args)[1],
        lambda tau, *args: u(tau, phi_arr, *args),
        lambda tau, *args: u_at_mu(mu_test_arr, tau, phi_arr, *args),
    )
    h = 1e-5
    for func in funcs:
        derivative = func(tau_test_arr, 1)
        finite_diff = (func(tau_test_arr + h) - func(tau_test_arr - h)) / (2 * h)
        assert np.max(np.abs(derivative - finite_diff)) < 1e-6 * np.max(np.abs(derivative))

    tops = np.concatenate([[0], np.nextafter(tau_arr[:-1], np.inf)])
    for func in funcs[:3]:
        for top, bottom in zip(tops, tau_arr):
            integral = func(bottom, -1) - func(top, -1)
            assert np.isclose(integral, quad(func, top, bottom, epsabs=0, epsrel=1e-10)[0], rtol=1e-8, atol=0)


def test_PS_input_checks_and_NT():
    print()
    with pytest.raises(ValueError):
        _pydisort(Leg_coeffs_all_B, radii_arr=radii_arr[:-1])  # One radius per layer boundary is required
    with pytest.raises(ValueError):
        _pydisort(Leg_coeffs_all_B, radii_arr=radii_arr[::-1])  # The radii must decrease from top to bottom
    with pytest.raises(ValueError):
        _pydisort(Leg_coeffs_all_B, radii_arr=altitudes)  # Altitudes are not radii: the bottom one is zero

    # As in Stamnes' DISORT, the NT corrections are not performed with the pseudo-spherical correction
    Leg_coeffs_all = np.array([0.2, 0.5, 0.7, 0.8, 0.75, 0.6])[:, None] ** np.arange(2 * NQuad)[None, :]
    u = _pydisort(Leg_coeffs_all)[-1]
    u_NT = _pydisort(Leg_coeffs_all, NT_cor=True)[-1]
    assert np.array_equal(u_NT(tau_test_arr, phi_arr), u(tau_test_arr, phi_arr))
    # They are performed without it
    u_NT_plane_parallel = _pydisort(Leg_coeffs_all, NT_cor=True, radii_arr=None)[-1]
    u_plane_parallel = _pydisort(Leg_coeffs_all, radii_arr=None)[-1]
    assert not np.allclose(u_NT_plane_parallel(tau_test_arr, phi_arr), u_plane_parallel(tau_test_arr, phi_arr))
