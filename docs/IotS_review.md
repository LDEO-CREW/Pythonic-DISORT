# Integration of the source function (IotS): exploration, prototype, accuracy and cost

Written 2026-11-11. Annotate items in ALL CAPS as usual. Nothing has been committed.

## 0. Short answers

1. **Implemented, with one new optional keyword:** `subroutines.interpolate(u, NT_cor=None, method="polynomial")`. `method="IotS"` switches to the integration of the source function, for `u` and `u0` as output by `pydisort`. The default is unchanged. All 98 existing call paths are bit-identical to HEAD, and the solve time is unchanged (the IotS data are lazy closures).
2. **Speed:** the output stage costs **2–10x** the polynomial path for 1 polar angle (+2–70 ms), **1.7–23x** for 10 and **2–68x** for 60. Measured on whole sessions (solve + outputs), the ratio is **1.02–1.36** for one angle (the typical use), 1.07–1.42 for 10 angles and 1.3–2.1 for 60 angles. Re-evaluating the same angles at new tau costs 1.1–1.7x the polynomial path.
3. **Accuracy:** IotS is much more accurate almost everywhere that matters. It is also the only method that agrees with Stamnes' DISORT at user angles (to single precision). The polynomial method has two errors that do not go away as NQuad increases:
   - **Nadir/zenith:** 1.5–13% at |mu| = 1 for any beam problem, even at NQuad 16–64.
   - **Grazing angles at the boundaries:** 0.5–5%.

   IotS stays at the error level of the quadrature-node values. In thermal IR without scattering it is **exact at any NQuad**, whereas the polynomial method is off by 7–15% (4 streams) or about 4% (8 streams), roughly 0.1–2.5 K in brightness temperature.
4. **Where the polynomial method wins:** only when delta-M (or a truncated phase function) is used **without** NT corrections. IotS then faithfully reproduces the truncated single-scattering phase function, ringing included, while the polynomial method smooths it between the nodes. The polynomial method is then up to about 1.7x better at some angles. Both methods are poor in that regime (5–15% errors), and with NT corrections IotS wins.
5. **Side findings (pre-existing, not fixed):**
   - **the IMS correction has the wrong sign** (section 5.1), which is important;
   - `(N, NFourier)` BC arrays crash `pydisort` (section 5.2).

## 1. What IotS is, in PythonicDISORT's terms

For Fourier mode m and layer l, the quadrature solution is a sum of exponentials in the delta-scaled tau:
`u^m(t, mu_j) = sum_k GC[m,l,j,k] exp(K_k (t - ref_k)) + B[m,l,j] exp(-t/mu0) + v_l(t)_j`.

The source function at an arbitrary `mu` is the scattering integral of that solution, discretized with the same quadrature as the solve, plus the direct-beam and isotropic sources. It is a sum of the same exponentials, with coefficients `S[m,l,k](mu) = sum_j D^m_l(mu, mu_j) w_j GC[m,l,j,k]`. The transfer equation along `mu` is then integrated exactly through the layers, from the boundary, and the BCs and BDRF are evaluated at `mu`. This is Stamnes (1982), and DISORT's `TERPEV`/`TERPSO`/`USRINT`.

Properties:
- At the quadrature nodes it reproduces the discrete ordinates solution. Verified to <= 2e-14, including antiderivatives, `u0` and the Fourier error.
- It satisfies the boundary conditions at every `mu`.

Implementation notes (all in the new `src/PythonicDISORT/_IotS.py`):
- **Kernels:** every kernel is a divided difference of exponentials, (exp(-a d) - exp(-b d)) / (b - a). It is evaluated as exp(-min(a,b) d) * d * phi1(-|b - a| d) with `expm1`. This needs no branching and cannot overflow. The removable singularities at mu = -mu0 (beam) and mu = 1/K (eigenvalue resonance) need no special case; checked continuous to 1e-14.
- **Antiderivative:** along any direction mu du/dt = u - J, so mu u + (antiderivative of J) is an antiderivative of u. That gives closed-form per-layer antiderivatives, matching the node antiderivatives exactly. Telescoped layer integrals agree with adaptive quadrature to <= 5e-14, with and without NT.
- **Isotropic source:** polynomial sources of any degree. The particular solution along the ray is a finite sum, re-expanded about the start boundary so that the constant term uses `expm1`.
- **No-scattering shortcut:** `(mode, layer)` pairs that took the shortcut contribute no source and are skipped, which helps gas/Rayleigh layers.
- **Contraction order:** "kernel-first" is used for few angles and "moment-first" when Nmu > NLeg - m.
- **Caching:** everything that depends only on `mu` (source coefficients, boundary intensities at every interface) is cached by value. Repeat calls at new tau are cheap.

## 2. API and wiring (uncommitted working-tree changes)

| File | Change |
|---|---|
| `src/PythonicDISORT/_IotS.py` (new, ~440 lines) | `_build_IotS(...)` returns the closures `IotS_u(mu, tau, phi, is_antiderivative_wrt_tau, return_Fourier_error)` and `IotS_u0(mu, tau, is_antiderivative_wrt_tau)` |
| `_assemble_intensity_and_fluxes.py:576-599` | builds the closures (nothing is computed until called) and attaches them as `u._IotS`, `u0._IotS` |
| `_NT_corrections.py:283` | `u_corrected._IotS = u_star._IotS`, which is always the uncorrected intensity |
| `subroutines.py:598, 658, 816` | `interpolate(..., method="polynomial")`; `method="IotS"` branches to the new `_IotS_interpolant` before the untouched polynomial code |
| `pydisotest/IotS_test.py` + `Stamnes_results/IotS_test.npz` (new) | node reproduction, antiderivative vs `quad_vec`, and DISORT `USRANG` comparison (no NT) |

Semantics chosen. Each is a decision for you.
- **2.1** With `method="IotS"`, NT corrections are always added at the evaluation points. So `True`/`"quad"` act like `"eval"`, which is what DISORT does. If `pydisort` ran with `NT_cor=True` and `interpolate`'s `NT_cor` is None, the corrections are applied (at the evaluation points). `"quad"` has no IotS meaning short of polynomially interpolating the corrections, which seemed pointless. Alternatively, raise on `"quad"`.
- **2.2** The Fourier error is computed at the evaluation points. The polynomial path computes it at the nodes.
- **2.3** Vector/array `b_pos`/`b_neg` are only known at the nodes, so IotS interpolates them barycentrically; scalar BCs are exact. A callable BC would be an API extension; not done.
- **2.4** `mu = 0` raises, as in DISORT. The intensity is discontinuous there at the boundaries.
- **2.5** Not autograd-traceable. The polynomial `interpolate` isn't either (`BarycentricInterpolator`).
- **2.6** Zero-API-change alternative: make IotS the default in `interpolate` for `pydisort` outputs, keeping `"polynomial"` as the fallback for other functions or for speed with very many angles. The signature would be unchanged, but interpolated values would change by the interpolation error (1e-3–1e-1). My recommendation, given section 3: do this at the next version bump, with a changelog line. Also see 2.7.
- **2.7** `_IotS_interpolant` duplicates ~20 lines of `interpolate`'s `NT_cor` parsing. This keeps the polynomial path byte-identical for the prototype; if adopted, factor it out.

## 3. Accuracy

**Method:** both methods are evaluated at 99 mu per hemisphere (|mu| in [0.02, 1]), at TOA and BOA, for phi in {0, pi/2, pi}.
- The reference is IotS at NQuad 128–320 without delta-M, so the phase function is fully resolved. The poly-vs-IotS spread at the reference NQuad was checked.
- With NT, the IMS sign was corrected to NT1988/DISORT for this study (section 5.1), otherwise the aureole numbers are dominated by that bug. Both methods share the NT terms at the evaluation points (`"eval"`).
- Table entries are the max relative error in each angular region.
- "Zenith/nadir" means |mu| >= 0.95, "mid" means 0.2 <= |mu| < 0.95, and "grazing" means |mu| < 0.2.
- TOA-down and BOA-up over a black surface are identically zero for both methods. BOA-up over a Lambertian surface is isotropic, so both methods are exact interpolators and their errors are identical (the DO flux error).

### 3.1 Results (poly / IotS)

| Scenario | NQuad | TOA up: nadir | TOA up: mid | TOA up: grazing | BOA down: zenith | BOA down: mid | BOA down: grazing |
|---|---|---|---|---|---|---|---|
| Rayleigh tau 0.1, SZA 60, albedo 0.05 | 16 | 3.8e-2 / 2.0e-4 | 3.7e-3 / 4.3e-4 | 3.1e-2 / 4.4e-4 | 9.1e-2 / 3.9e-4 | 1.1e-2 / 5.1e-4 | 4.7e-2 / 5.1e-4 |
| | 32 | 2.1e-2 / 2.0e-6 | 8.1e-4 / 4.1e-6 | 3.6e-3 / 4.2e-6 | 3.7e-2 / 3.8e-6 | 1.3e-3 / 4.8e-6 | 6.0e-3 / 4.7e-6 |
| Aerosol HG 0.7, tau 0.3, w 0.92, dM + NT | 16 | 5.6e-2 / 4.4e-5 | 1.8e-2 / 1.4e-4 | 1.0e-2 / 4.4e-4 | 2.4e-1 / 7.4e-5 | 3.6e-2 / 1.2e-4 | 6.9e-3 / 3.0e-4 |
| | 32 | 8.0e-3 / 1.2e-7 | 9.1e-5 / 6.3e-7 | 1.1e-3 / 9.4e-6 | 1.5e-1 / 1.4e-7 | 2.2e-3 / 5.9e-7 | 1.4e-3 / 6.3e-6 |
| Cloud C.1 (Mie), tau 10, w 0.999, dM + NT | 16 | 1.3e-1 / 1.6e-3 | 5.4e-2 / 3.4e-2 * | 4.1e-2 / 4.1e-2 | 5.3e-2 / 2.3e-4 | 3.1e-1 / 3.1e-1 * | 3.5e-2 / 3.5e-2 |
| | 32 | 3.7e-2 / 1.3e-3 | 3.4e-2 / 2.0e-2 * | 1.3e-2 / 1.4e-2 | 3.0e-2 / 2.1e-4 | 3.9e-2 / 4.0e-2 * | 1.3e-2 / 1.2e-2 |
| Thermal IR window, 20 absorbing layers, tau 1, no scattering | 4 | 7.0e-2 / 2.6e-5 | 5.9e-2 / 2.5e-5 | 7.9e-2 / 9.6e-7 | 1.5e-1 / 0 | 9.6e-2 / 0 | 7.8e-2 / 0 |
| | 8 | 5.3e-3 / 2.8e-7 | 1.7e-2 / 2.6e-7 | 3.9e-2 / 1.0e-8 | 3.4e-4 / 0 | 9.4e-4 / 0 | 9.2e-4 / 0 |
| | 16 | 3.1e-4 / 3.4e-9 | 5.0e-4 / 3.2e-9 | 1.4e-3 / 1.2e-10 | 3.0e-4 / 0 | 1.2e-4 / 0 | 1.4e-4 / 0 |
| IR + cirrus (tau 1, w 0.5, g 0.8), dM | 4 | 4.0e-3 / 3.9e-4 | 2.2e-2 / 3.6e-3 | 3.6e-2 / 3.2e-3 | 1.0e-1 / 4.8e-4 | 6.6e-2 / 3.5e-4 | 9.0e-2 / 1.6e-5 |
| | 8 | 2.8e-4 / 9.0e-5 | 1.1e-2 / 3.9e-4 | 4.5e-2 / 1.9e-4 | 7.8e-3 / 1.8e-5 | 3.4e-3 / 2.1e-5 | 2.2e-3 / 4.0e-6 |
| | 16 | 7.2e-5 / 4.0e-6 | 1.6e-4 / 1.4e-5 | 1.7e-3 / 4.3e-5 | 3.3e-4 / 3.5e-7 | 1.3e-4 / 5.9e-7 | 1.7e-4 / 1.2e-7 |

\* The cloud "mid" cells contain the glory (TOA, Theta = 180 deg) or the aureole (BOA). The error there is the delta-M + NT limitation and is identical for both methods.
- Excluding +-15 deg around the aureole and glory, the cloud TOA max errors are 3.7e-2 / 1.4e-2 (NQuad 32) and 2.9e-2 / 2.6e-3 (NQuad 64).
- Cloud TOA median errors are 2.1e-3 / 2.6e-4 (NQuad 32) and 1.7e-3 / 2.7e-4 (NQuad 64).
- Rayleigh over cloud (tau 0.1 over C.1 tau 10), TOA, excluding aureole and glory: 9.8e-2 / 1.2e-3 (NQuad 16) and 2.3e-2 / 1.0e-3 (NQuad 32).

### 3.2 When each method wins, and why

1. **Zenith/nadir (polynomial fails; important in practice).** Odd Fourier modes carry a factor sqrt(1 - mu^2) = sin(theta), which is not polynomial in mu at mu = +-1. The polynomial interpolant extrapolates it past the outermost Gauss node. The errors converge only algebraically, as about NQuad^-1 (Rayleigh: 3.8% -> 2.1% from 16 -> 32 streams). The interpolated nadir radiance even depends on phi, which is physically impossible.
   - IotS evaluates `Pbar_l^m(mu)` exactly, so modes m >= 1 vanish at mu = +-1.
   - Satellite nadir viewing and zenith sky radiance are the most common geometries.
   - Diagnostic: removing sqrt(1 - mu^2) from the odd modes before interpolating ("mode-wise polynomial") cuts these errors 2–25x. It still leaves them 10–1000x above IotS and does nothing at mid or grazing angles, so it isn't worth implementing instead.
2. **Grazing angles at the boundaries (polynomial fails).** TOA-up and BOA-down radiances have an exp(-Delta tau / mu) boundary layer in mu, which a degree N-1 polynomial cannot follow. IotS integrates it exactly.
3. **Thermal emission, i.e. limb darkening (polynomial fails, IotS exact).** Without scattering, IotS is the exact formal solution at any NQuad. The polynomial errors are 7–15% at 4 streams, up to 4% at 8 and about 0.1% at 16, i.e. about 0.1–2.5 K brightness temperature at 10 um (1% radiance = 0.6 K).
4. **With NT corrections (polynomial fails in the aureole).** In `"eval"` mode the polynomial result is `Poly[u*] - u1*(mu) + u1~(mu)`. The interpolation error of the truncated single scattering `Poly[u1*] - u1*` remains, and it is sharply peaked near the beam. IotS computes `u1*` exactly, so it cancels exactly against TMS. Aerosol BOA example: 24% / 15% (NQuad 16 / 32) for polynomial vs 7e-5 / 1e-7 for IotS.
5. **Equal: where the node (DO) error dominates.** Examples: the low-NQuad cloud aureole, the glory (multiply-scattered peak effects that delta-M + NT does not treat), and BOA grazing angles under thick clouds. IotS turns interpolated radiances into radiances as accurate as at the nodes. It cannot fix the nodes.
6. **Polynomial wins: delta-M or truncation without NT.** Examples: cloud TOA at NQuad 64, 5.7% vs 9.6% max error (glory); aerosol without delta-M at NQuad 16, BOA oblique angles, 3.3% vs 3.8%. IotS reproduces the truncated phase function's single scattering exactly, oscillations included, while the polynomial method smooths between nodes. Both are far outside a useful budget there, and NT corrections are the fix. DISORT always applies them with delta-M (`CORINT`). This regime is not an argument for the polynomial method.

### 3.3 Practical relevance

Typical budgets are:
- reflective-band radiances: ~1–3% radiometric uncertainty; forward models for retrievals usually aim for <~0.5–1%;
- thermal IR: ~0.1–0.5 K.

The polynomial method's nadir/zenith errors (2–13% at NQuad 16–32 for beam problems) and its IR limb errors at 4–8 streams exceed these budgets, in geometries users actually request. IotS keeps the angular error at or below the node error in every case tested. The cases where the methods are equal (aureole/glory at low NQuad) need more streams, not a different interpolation.

### 3.4 Validation against Stamnes' DISORT (`USRANG = TRUE`, single precision)

| Case | IotS: max abs diff / max intensity | polynomial |
|---|---|---|
| 1 layer, truncated HG, Lambertian, beam | 2.7e-6 | 1.3e-2 (max rel 13%) |
| 1 layer, HG 0.85, delta-M + NT (DISORT's IMS convention mimicked) | 7.1e-6 | 1.5e-2 (max rel 120%) |
| 3 layers, thermal + beam + Lambertian | 7.7e-6 | 1.2e-2 |
| 4 layers, delta-M + NT, -mu0 among user angles | 2.2e-4 (known IMS-averaging difference at interior tau) | 3.2e-2 |

## 4. Cost

DISORT conda env (NumPy 1.26, netlib BLAS). Idle machine, min-of-repeats, polynomial and IotS interleaved. Each measurement is the first `interpolate`+call on a fresh solution (cold caches), with NT `"eval"`, 2 tau, 5 phi. HG layers, so no layer takes the no-scattering shortcut; realistic gas + cloud columns would be cheaper.

| NQuad | NLayers | solve (ms) | poly out, Nmu = 1/10/60 (ms) | IotS out, Nmu = 1/10/60 (ms) | session ratio, Nmu = 1/10/60 |
|---|---|---|---|---|---|
| 8 | 1 | 4.3 | 2.3/3.6/3.8 | 4.3/6.3/7.2 | 1.30/1.34/1.41 |
| 8 | 10 | 6.8 | 2.5/4.8/6.1 | 4.7/8.1/12.1 | 1.23/1.29/1.46 |
| 8 | 50 | 11.5 | 3.1/6.8/10.9 | 6.0/12.9/29.0 | 1.19/1.33/1.81 |
| 16 | 1 | 7.5 | 2.4/3.8/4.0 | 6.0/8.3/10.3 | 1.36/1.40/1.54 |
| 16 | 10 | 17.5 | 2.7/4.9/6.2 | 6.7/11.2/23.7 | 1.20/1.28/1.72 |
| 16 | 50 | 47.0 | 3.3/7.0/11.2 | 8.8/22.9/74.5 | 1.10/1.28/2.09 |
| 32 | 1 | 17.5 | 2.8/4.3/4.6 | 9.8/13.6/22.9 | 1.33/1.42/1.80 |
| 32 | 10 | 79.3 | 3.3/5.5/6.9 | 11.8/24.7/71.9 | 1.10/1.23/1.75 |
| 32 | 50 | 360.5 | 3.8/7.4/11.9 | 17.9/65.9/325.3 | 1.04/1.16/1.84 |
| 64 | 1 | 73.7 | 6.3/8.9/9.3 | 21.9/32.7/80.5 | 1.19/1.29/1.86 |
| 64 | 10 | 819.4 | 7.5/10.4/13.4 | 30.2/100.5/374.3 | 1.03/1.07/1.43 |
| 64 | 50 | 4180.7 | 7.6/13.8/21.3 | 79.3/321.0/1440.6 | 1.02/1.07/1.34 |

- **Solve:** `pydisort` itself is unchanged. Over three alternating rounds against HEAD the ratio was 1.00 +- 0.02.
- **Scaling:** the IotS cost scales as NFourier × NLayers × Nmu × NQuad × (NLeg + NQuad). The flop count matches DISORT's `TERPEV` + `USRINT`. Here it is about half netlib-BLAS matmuls and half elementwise `expm1` kernels.
- **Optimizations already applied:** the shortcut-pair skip, the moment-first contraction, and precomputed exponentials with a dense `expm1`. Together they cut the NQuad 32 / 50-layer / 60-angle case by about 1.5x (573 -> 373 ms under the profiler).
- **Cached angles:** repeat calls at the same `mu` and new tau cost 1.1–1.7x the polynomial path (e.g. 7.2 vs 6.3 ms).

## 5. Side findings (pre-existing; NOT fixed — decide)

### 5.1 The IMS correction has the wrong sign (important)

- **Code:** `_NT_corrections.py:216` (and :213, the autograd branch) adds `_IMS_correction(...)`.
- **NT1988 (p. 57, Eq. 23):** `u_hat = u - u_TMS ≈ -(1 - f w) w_hat^2 [2 P_hat - P_hat^2] h(tau, mu, mu0*, mu0*)` with h >= 0, i.e. the term is subtracted.
- **DISORT:** `INTCOR` does `UU = UU - DUIMS`. Its `SECSCA` computes the same quantity as `_IMS_correction`.
- **Notebook:** cell 222 (id `45f68b9a`) writes `u_true ≈ u_TMS + u_IMS` with a positive `u_IMS`.

Evidence:
- **vs DISORT `USRANG`** (1 layer, HG 0.85, tau 2): DISORT - TMS + IMS_py equals IotS u* to 1e-5 at every in-window angle.
- **vs a 256-stream truth** (same case, aureole), max relative error:

| IMS treatment | NQuad 8 | NQuad 16 |
|---|---|---|
| current sign | 112% | 9.7% |
| NT1988/DISORT sign | 4.1% | 0.2% |
| TMS only | 54% | 4.8% |

So the current sign makes IMS worse than not applying it. This likely explains the earlier "IMS overcorrects badly ... worse than TMS-only" observation in the IMS-averaging study.

Why the tests missed it:
- DISORT applies IMS only for downward |theta - theta0| <= 10 deg (`DTHETA` in `INTCOR`), while PythonicDISORT applies it to all downward directions.
- The pytests skip angles within 10 deg of the beam.

The fix:
- **Code:** a one-character sign flip in both branches. It covers values and antiderivatives, because `_IMS_correction` returns either.
- **Notebook:** a minus sign in cell 222, plus re-running the NT cells.
- **Optional:** consider DISORT's 10-degree window, a separate design question.

### 5.2 `(N, NFourier)` BC arrays always crash

- **Where:** `pydisort.py:297` and `:303` use `elif len(b_pos) == N:` to detect the vector case. That condition is also true for a 2-D `(N, NFourier)` array, so the documented array form goes down the vector branch and fails in `_solve_for_coeffs` with "could not broadcast (N, NFourier) into (N,)". This was reproduced at HEAD.
- **Fix:** `elif np.ndim(b_pos) == 1 and len(b_pos) == N:` (and the same for `b_neg`). I applied it only in my scratch copy, to test IotS with array BCs.

### 5.3 Notebook wording

- Cell 196 (id `b7f00a70`) calls the method the "*iteration* of the source-function method"; it is the *integration*.
- The same cell says "We may implement this in PythonicDISORT in the future". If IotS is adopted, that cell needs a short practical paragraph: what IotS is, that it is exact at the nodes and in non-scattering atmospheres, when to prefer it, and its cost. I have not edited the notebook.

## 6. Decisions

1. Keep the IotS implementation (working tree), revise it, or drop it?
2. API: keep `method="IotS"` opt-in, or make it the default (2.6)? Rename `method` or its values?
3. `NT_cor="quad"` under IotS: act as `"eval"` (current), or raise (2.1)?
4. IMS sign (5.1): fix? Also adopt DISORT's 10-degree aureole window?
5. BC-array bug (5.2): fix?
6. Keep `pydisotest/IotS_test.py` and `Stamnes_results/IotS_test.npz`?
7. Notebook text for IotS and the "iteration" typo (5.3)?

Reproduction scripts, all in this session's scratchpad (temporary; I can copy them into the repo on request):
- `t_nodes.py`: node reproduction;
- `t_disort.py`, `t_disort2.py`: DISORT `USRANG`;
- `acc.py`, `run_acc.py`, `modewise.py`, `diag*.py`: accuracy study;
- `bench_eval.py`, `bench_solve.py`, `bench_cached.py`: timing;
- `bitid.py`: bit-identity;
- `chk_IMS_truth.py`, `dbg_TMS.py`: IMS sign evidence.
