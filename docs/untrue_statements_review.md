# Untrue statements in v1.9: notebook, docstrings, comments, README

Written 2026-11-12 from the v1.9 audit. Annotate each item in ALL CAPS on its `RESPONSE:` line. Nothing in this list has been edited, except where an item says so.

Each item gives the location, the statement, why it is untrue and a suggested correction. Notebook cells are given by section, index and cell id (indices count from 0 and match the notebook as of today).

**Omitted because the bug fixes make them true:**
- §1.4 (cell 62, `88d2926d`): "so $\mu_0$ need not avoid quadrature angles". This is true after the fix to bug 3. However, the warning "`-1/mu0` is close to an eigenvalue ... Consider slightly perturbing `mu0`" still fires whenever `mu0` is a quadrature node, which contradicts the sentence. Should the warning stay as it is?
- `pydisort` docstring, `Fm`: the direct flux is "of the same type and size as `tau`". This is true after the fix to bug 9.
- `subroutines.cache_BDRF_Fourier_modes` docstring: the cached scalar modes are functions "which output is a scalar". The fix to bug 6 returns scalars as scalars, and the docstring was updated with it.

## A. Notebook (`docs/Pythonic-DISORT.ipynb`)

### A1. §1.3, cell 47 (`06b14d0e`): Legendre coefficients "each between $-1$ and $1$ inclusive"
- **Why untrue:** `pydisort` raises unless $-1 < g_\ell < 1$ for $\ell \geq 1$ (`pydisort.py`, the check after "Value checks on the phase function Legendre coefficients"). Only $g_0$ may equal 1.
- **Suggested:** "each strictly between $-1$ and $1$, except $g_0 = 1$".

RESPONSE:

### A2. §1.3.1, cell 55 (`c9f5735d`): "$f \in [0, 1]$"
- **Why untrue:** two cells later, cell 57 says $f \in [0, 1)$. The code divides by $1 - f$, and after the fix to bug 11 it rejects $f = 1$.
- **Suggested:** "$f \in [0, 1)$".

RESPONSE:

### A3. §1.6, cell 76 (`bdeac418`): a scalar `b_neg` equals `FISOT` "plus blackbody emission from the bottom boundary"
- **Why untrue:** `b_neg` is the BC at the top boundary. In Stamnes' DISORT the matching thermal term is the top-boundary emission (`TTEMP`, `TEMIS`). The bottom-boundary emission (`BTEMP`) enters `b_pos`.
- **Suggested:** "plus blackbody emission from the top boundary (`TTEMP` and `TEMIS`)".

RESPONSE:

### A4. §1.7, cell 83 (`67fbe54d`): citation numbers and Planck units
- **Why untrue:**
  - The cell cites "[4, section 3.2.3]" and "[4, section 2.5]", but both links point to `#cite-STWLE2000`, which is reference [8]. Reference [4] is Stamnes & Conklin (1984).
  - $B$ is said to have "units $W\,m^{-2}\,sr^{-1}$", but it is integrated over wavenumber in the next formula. So it is per unit wavenumber: $W\,m^{-2}\,sr^{-1}\,(cm^{-1})^{-1}$. `subroutines.Planck` returns exactly that (see C3).
- **Suggested:** "[8, section 3.2.3]", "[8, section 2.5]", and "units $W\,m^{-2}\,sr^{-1}$ per $cm^{-1}$".

RESPONSE:

### A5. §2, cell 91 (`17a83658`): the antiderivative example
- **Statement:** "`flux_up(tau0, is_antiderivative_wrt_tau=True) - flux_up(0, is_antiderivative_wrt_tau=True)` will give the integral of `flux_up` over $\tau \in [0, \tau_0]$".
- **Why untrue:** this holds only when $\tau_0$ lies in the first layer. Each layer's antiderivative has its own constant of integration (see A10).
  - In a three-layer test with $\tau_0 = 5$, the expression gives 1.655, while the integral is 0.360.
  - Summing the per-layer differences gives 0.360, as the `pydisort` docstring idiom does.
- **Suggested:** say "for $\tau_0$ in the first layer". Alternatively, show the per-layer idiom with `np.nextafter` (see C2).

RESPONSE:

### A6. §3.3, cell 120 (`5a99d60e`): "$f \delta(\mu - \mu_0)\delta(\phi - \phi_0)$"
- **Why untrue:** the forward peak of the beam scatters into the beam direction $-\mu_0$. The same cell then writes $u_\text{direct}(0, \mu, \phi) = I_0 \delta(\mu + \mu_0) \delta(\phi - \phi_0)$.
- **Suggested:** $\delta(\mu + \mu_0)$.

RESPONSE:

### A7. §3.4.2, cell 141 (`24e35e87`), code: `weighted_Leg_coeffs_l = ((2 * np.arange(NLeg) + 1) * Leg_coeffs[l, m:])`
- **Why untrue:** for any $m > 0$ the two factors have lengths `NLeg` and `NLeg - m`, so the line raises a shape error. It runs in the notebook only because $m = 0$ there.
- **Suggested:** `(2 * np.arange(m, NLeg) + 1) * Leg_coeffs[l, m:]`.

RESPONSE:

### A8. §3.4.2, cell 143 (`f6822d0c`), code: BDRF modes called with `-mu_arr_pos` and `-np.array([mu0])`
- **Why untrue:** the package calls them with positive second arguments: `BDRF_Fourier_modes_m(mu_arr_pos, mu_arr_pos)` and `(mu_arr_pos, np.array([mu0]))` in `_solve_for_coeffs.py`. The `pydisort` docstring also says the arguments are "`mu, -mu_p`", where $\mu_p < 0$ is the incident direction. The notebook code therefore evaluates any non-symmetric BDRF at the wrong angles. (The notebook's Lambertian example is constant, so its outputs are unaffected.)
- **Suggested:** `BDRF_Fourier_modes[m](mu_arr_pos, mu_arr_pos)` and `BDRF_Fourier_modes[m](mu_arr_pos, np.array([mu0]))`.

RESPONSE:

### A9. §3.5, cell 151 (`77a3ccdb`) and the comment in cell 153 (`79a3aa59`): eigenvalues "arrange[d] ... as negative then positive, from largest to smallest magnitude"
- **Why untrue:** `np.linalg.eig` returns eigenvalues in no particular order, and neither the notebook nor the package sorts them. "Negative then positive" is true. Nothing in the package relies on the magnitude order (see the chat explanation). The same comment is in `_solve_for_gen_and_part_sols.py:277`.
- **Suggested:** "arrange its entries as negative then positive, with the $j$th negative entry equal to minus the $j$th positive one". Alternatively, sort them (`np.argsort` on `K_squared` before the square root), if you want the statement as written.
- **Also in this cell:** "eigenvalues $k$ which are always real". As the comment in `_solve_for_gen_and_part_sols.py` says, a truncated phase function can give complex eigenvalues. You previously chose not to put this nuance in the notebook, so this is listed only for completeness.

RESPONSE:

### A10. §3.7, cell 200 (`779e1cdd`): "Since every output function of `pydisort` is continuous, its antiderivatives are also continuous."
- **Why untrue:** the antiderivatives are per-layer closed forms with a different constant in each layer, so they jump at the layer interfaces. In a three-layer test, the `flux_up` antiderivative jumps by 1.31 at $\tau = 1$, while `flux_up` itself is continuous to $10^{-9}$. Integrals are still correct when taken within a layer (the `pydisort` docstring idiom).
- **Suggested:** "Each antiderivative is continuous within a layer but generally not across layer interfaces, so integrate layer by layer (see the `pydisort` docstring)."

RESPONSE:

### A11. §3.7.1, cell 212 (`2d0d37b0`), and §3.7.3, cell 238 (`268745db`): "this verification will not reflect quadrature and truncation errors"
- **Why untrue:** the verified residual is almost entirely truncation error.
  - With the full HG phase function, the RTE residual is 1.59 (0.209 with NT corrections).
  - With the truncated series it is $1.7 \times 10^{-5}$.
  - Quadrature errors are indeed not reflected.
- **Suggested:** "this verification will not reflect quadrature errors; it does reflect the truncation of the phase function".

RESPONSE:

### A12. §3.7.2, cell 222 (`45f68b9a`): "$p_r = p - p^*$ is the residual phase function"
- **Why untrue:** the residual used in $u_\text{IMS}$ (with the $f$ factors pulled out in front) is normalized to unit integral: $p_r = (p - (1 - f)p^*)/f$. Its Legendre coefficients are 1 for $\ell <$ `NLeg` and $g_\ell/f$ above, which for one layer are the $\bar{g}_\ell$ of the next formula and of the code. $p - p^*$ has coefficients $g_\ell - g^*_\ell$ instead.
- **Note:** the sign of the IMS term in this cell was corrected with bug 1 (REVIEW note in the cell).
- **Suggested:** "$p_r = \left(p - (1 - f)p^*\right)/f$".

RESPONSE:

### A13. §3.7, cell 196 (`b7f00a70`): polynomial interpolation paragraph (affected by the IotS switch)
- **Why untrue or outdated:**
  - (a) "Since $\mu_i$ are Legendre points, uniform convergence is guaranteed as $N \rightarrow \infty$". At $\mu = \pm 1$ the error decays only like $1/\text{NQuad}$, because the odd Fourier modes carry $\sqrt{1 - \mu^2}$. The interpolated nadir radiance also depends on $\phi$. For example, with HG $g = 0.5$ the nadir error at the bottom is 13% at NQuad 16 and 3% at NQuad 64. So the statement is misleading rather than false.
  - (b) "*iteration* of the source-function method": Stamnes (1982), the paper cited as [11], calls it the *integration* of the source function.
  - (c) "We may implement this in PythonicDISORT in the future" and "under the hood it uses `scipy.interpolate.BarycentricInterpolator`" become untrue once IotS replaces the polynomial interpolation.
- **Suggested:** rewrite the paragraph for IotS: what it is, that it is exact at the nodes and in non-scattering layers, and its cost. I can draft it.
- **Done (2026-11-12):** after your decision to switch to IotS, cell 196 was rewritten (REVIEW note in the cell), which resolves (a) to (c).

RESPONSE:

### A14. §5, cell 298 (`072db819`): "PythonicDISORT has for-loops over the atmospheric layers $l$ and over the Fourier modes $m$"
- **Why outdated:** the solve (eigenproblems, particular solutions and BC system assembly) no longer loops over layers; it is batched. It still loops over the Fourier modes. The remaining layer loops are the TMS scan of the NT corrections and the layer propagation of IotS.
- **Suggested:** "PythonicDISORT loops over the Fourier modes $m$ (the layers are vectorized)".

RESPONSE:

### A15. §6, cell 300 (`071bbadb`), and §6.4, cell 328 (`42a61a60`): the $10^\circ$ window around the beam
- **Statements:**
  - Cell 300: "even with NT corrections, there tends to be large inaccuracies at $\mu$ points that are extremely close to the direct beam".
  - Cell 328: "For some tests, we do not compare intensity values at polar angles that are within $10^\circ$ of the direct beam".
- **Why untrue:** the inaccuracies were the wrong sign of the IMS correction (bug 1). After the fix, test 5b matches DISORT at the in-window nodes to 0.44%, and `pydisotest/5_test.py`, the only test with a window, now uses $0^\circ$. `pydisotest/5_test.ipynb` still uses $10^\circ$.
- **Note:** I added a REVIEW note to cell 300.
- **Suggested:** delete the sentence in both cells, and set the window in `5_test.ipynb` to 0.

RESPONSE:

### A16. §6.4, cell 328 (`42a61a60`): "Except for test problems 11 and I ..., we compare the solutions ... against corresponding solutions from Stamnes' DISORT"
- **Why untrue:** test problem NT (`NT_test.py`) does not compare against DISORT either; it checks internal consistency.
- **Suggested:** "Except for test problems 11, I and NT". If you keep `IotS_test.py`, it does compare against DISORT.

RESPONSE:

### A17. §6.5, cell 338 (`aed4cb45`), and its cross-reference in cell 196: why DISORT is slow
- **Statement:** "DISORT takes a long time to compute the intensity at a large number of points because it applies NT corrections to every point; PythonicDISORT's `NT_cor="quad"` addresses such a scenario".
- **Why untrue:** in that timing both codes correct the same $16 \times 1000 \times 100$ points. Stamnes' DISORT runs with `USRANG = False` (cell 331), so it returns and corrects the intensity at its 16 computational angles, and PythonicDISORT's `u` is corrected at its 16 nodes. So the difference is not where the corrections are applied.
- **Suggested:** drop the causal claim.
- **Partly done (2026-11-12):** the second half of the sentence (about `NT_cor="quad"`) became untrue with the IotS switch and was removed (REVIEW note in cell 338). The first half remains for your response.

RESPONSE:

### A18. Stale stored outputs
- **Cells 166, 172 and 191** (verification of the particular solutions and the full solution) print errors of about $1.7$, $2.9$ and $3.1 \times 10^{-6}$. Re-running them now gives $1.6 \times 10^{-12}$, $1.2 \times 10^{-11}$ and $1.8 \times 10^{-11}$.
- **Cell 182** prints a condition number of 305.37; it is now 540.35.
- **NT-corrected cells:** after the IMS fix, their stored outputs also change.
- **Suggested:** re-run the notebook before the release.

RESPONSE:

## B. `pydisort` docstring (`src/PythonicDISORT/pydisort.py`)

### B1. Line 51: "Each coefficient should be between 0 and 1 inclusive."
- **Why untrue:** negative coefficients are accepted, and $\pm 1$ is rejected for $\ell \geq 1$ (as in A1).
- **Suggested:** "Each coefficient must be strictly between -1 and 1, except the zeroth which must be 1."

RESPONSE:

### B2. Line 121: `Fp(tau_arr, True) - Fp(np.insert(tau_arr[:-1] + 1e-15, 0, 0), True)` gives the integral over each layer
- **Why untrue:** for interfaces at $\tau \geq 16$, `tau + 1e-15 == tau` in double precision, so the lower limit stays in the layer above. With interfaces at 10, 20 and 30, the third-layer integral comes out as 4.598 instead of 4.595.
- **Suggested:** `np.insert(np.nextafter(tau_arr[:-1], np.inf), 0, 0)`, which gave the exact values.

RESPONSE:

## C. Other docstrings and comments

### C1. `subroutines.transform_interval`, `transform_weights` and `generate_FD_mat`: `a`, `b` "optional"
- **Where:** `subroutines.py:44/46`, `70/72` and `188/190`.
- **Why untrue:** the parameters have no defaults.
- **Suggested:** drop "optional".

RESPONSE:

### C2. `subroutines.generate_FD_mat` docstring: parameter `Nphi`
- **Where:** `subroutines.py:186`.
- **Why untrue:** the parameter is named `Ntau`.

RESPONSE:

### C3. `subroutines.Planck`, units "W / (m^2 sr)"
- **Where:** `subroutines.py:307` and `320`.
- **Why untrue:** it is a spectral density in wavenumber: W / (m^2 sr cm^-1). The output of `blackbody_contrib_to_BCs`, its integral over wavenumber, is in W / (m^2 sr), as that docstring says. The notebook has the same omission (A4).

RESPONSE:

### C4. `_NT_corrections.py:573`: "16 coefficients of the Taylor series of phi2"
- **Why untrue:** `1.0 / np.cumprod(np.arange(2, 19))` has 17 entries.

RESPONSE:

### C5. `_solve_for_gen_and_part_sols.py:277`: "Eigenvalues arranged negative then positive, from largest to smallest magnitude"
- **Why untrue:** as in A9.
- **Suggested:** "Eigenvalues arranged negative then positive; the j-th negative one is minus the j-th positive one".

RESPONSE:

### C6. Internal shape tables
- `G_inv_mu_inv` is listed as `NLayers x NQuad x NQuad` (`_solve_for_coeffs.py:46`, `_assemble_intensity_and_fluxes.py:90`). It is `NLayers x NQuad`.
- `eigenvecs_GpG_arr` and `eigenvecs_GmG_arr` are listed as `NFourier*NLayers x NQuad/2 x NQuad` (`_solve_for_gen_and_part_sols.py:129-130`, also missing the closing backtick). They are `Nns x NQuad/2 x NQuad/2`, where Nns is the number of (mode, layer) pairs without the no-scattering shortcut.

RESPONSE:

## D. README

### D1. Line 65: "`numpy >= 1.8.0`"
- **Why untrue:** `np.einsum(..., optimize=True)`, used in `_assemble_intensity_and_fluxes.py` and `_IotS.py`, needs NumPy >= 1.12, and the required `scipy >= 1.8.0` itself needs NumPy >= 1.17.3.
- **Suggested:** "`numpy >= 1.17.3`" (or whatever your CI's oldest environment uses).

RESPONSE:

### D2. Line 12: "the only **missing features** are $\delta$-M+ scaling ..., the Chapman function ..., and pseudo-spherical corrections"
- **Why untrue:** Stamnes' DISORT also has the following, which PythonicDISORT lacks:
  - `IBCND = 1` (albedo and transmissivity of the whole medium for illumination from the top and bottom);
  - built-in BRDF models (`BDREF`: Hapke, RPV, Cox-Munk, Ross-Li);
  - automatic azimuthal convergence, where the Fourier series stops when the modes become small (`ACCUR`);
  - before the IotS switch, the source-function method for intensities at user angles.
- **Suggested:** list them, or write "the main missing features".

RESPONSE:
