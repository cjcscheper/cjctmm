# Theory of the transfer-matrix method

This note derives what `cjctmm` computes, step by step, for students who know
plane waves and Maxwell's equations but have not used the transfer-matrix
method (TMM) before. It also fixes the conventions (time dependence, the sign
of `r_p`) and explains the finite-substrate-pass model and when it is valid.

The derivation follows Born & Wolf, *Principles of Optics*, ch. 1.5–1.6, and
Macleod, *Thin-Film Optical Filters*, ch. 2; Hecht, *Optics*, ch. 4 and 9, is a
gentler introduction to the Fresnel equations and thin-film interference.

## 1. The problem

A planar stack of homogeneous, isotropic, non-magnetic layers lies between two
semi-infinite media. A monochromatic plane wave of angular frequency $\omega$
comes in from the first medium (index 0) at angle $\theta_0$ and leaves into
the last medium (index $N-1$):

```
  z →
  incident medium | layer 1 | layer 2 | ... | layer N-2 | exit medium
  n_0, θ_0          n_1, d_1  n_2, d_2         n_{N-2}     n_{N-1}
```

Each layer $j$ has a thickness $d_j$ and a complex refractive index
$n_j(\omega) = n_j' + i\kappa_j$. We want the complex amplitude reflection and
transmission coefficients $r$, $t$ and the power fractions $R$, $T$ and
$A = 1 - R - T$. Because every layer is linear, each frequency is solved on its
own; `solve_tmm` does all frequencies at once with numpy arrays.

## 2. Polarization: s and p, TE and TM

The plane of incidence contains the wavevector and the normal to the layers
(here the $xz$ plane; $y$ is normal to it). Any polarization is a sum of two
independent ones that do not mix in an isotropic planar stack:

| name                                 | electric field                   | magnetic field              |
|--------------------------------------|----------------------------------|-----------------------------|
| **s** (German *senkrecht*) = **TE**  | perpendicular to the plane ($y$) | in the plane                |
| **p** (*parallel*) = **TM**          | in the plane                     | perpendicular to the plane ($y$) |

"Transverse electric" means E is transverse to the plane of incidence. At
normal incidence there is no plane of incidence and s and p are the same wave.

## 3. Time convention and complex index

Fields are written as $\mathbf{E}(\mathbf{r}, t) = \mathrm{Re}\,[\mathbf{E}\,e^{i(\mathbf{k}\cdot\mathbf{r} - \omega t)}]$,
i.e. time dependence $e^{-i\omega t}$ (the physics convention, used by Born &
Wolf and Jackson). A wave travelling along $+z$ in a medium of index $n$ is
$e^{i n \omega z / c}$, so with $n = n' + i\kappa$ it decays as
$e^{-\kappa \omega z/c}$: **$\kappa > 0$ means absorption**, and the complex
permittivity is $\varepsilon = n^2 = \varepsilon_\infty + i\sigma/(\varepsilon_0\omega)$
for a conductivity $\sigma$.

Engineering texts and part of the thin-film literature (Macleod) use
$e^{+i\omega t}$ with $n = n' - i\kappa$. Every complex amplitude in that
convention is the complex conjugate of the one here. To use cjctmm with indices
in that convention, conjugate them before passing them in, and pass
`conjugate_model=True` to get $r^*$, $t^*$ back. $R$, $T$, $A$ are the same in
both.

## 4. Snell's law and why the solver needs $\cos\theta$

The interface conditions must hold at every point of every interface, so all
waves share the same dependence on $x$: the wavevector component along the
layers, $k_x = (\omega/c)\, n_j \sin\theta_j$, is the same in every layer. That
is Snell's law, and it lets every layer's angle be computed directly from the
incident medium:

$$ \sin\theta_j = \frac{n_0 \sin\theta_0}{n_j}. $$

The component normal to the layers is $k_{z,j} = (\omega/c)\, n_j\cos\theta_j$.
That, and not $\theta_j$ itself, is what the TMM needs: it gives the phase a
wave gains across a layer and it tilts the admittance (section 5). So
`snell_cos_theta` returns $\cos\theta_j = \sqrt{1 - \sin^2\theta_j}$ directly.

The square root has two roots, $\pm\cos\theta_j$. The physical one makes the
wave decay away from the interface it came through: $\mathrm{Im}(\cos\theta_j)
\ge 0$, so that $e^{i k_z z}$ decays along $+z$. Two cases matter:

- **absorbing layer**: $\cos\theta_j$ is complex; the forward wave is attenuated;
- **total internal reflection**: beyond the critical angle
  $\theta_c = \arcsin(n_2/n_1)$, $\sin\theta_2 > 1$ and $\cos\theta_2$ is purely
  imaginary: the transmitted wave is evanescent and carries no power, $R = 1$.

For passive media and a real Snell invariant this root is the principal square
root (both its real and imaginary parts are $\ge 0$), so $\mathrm{Im}(n\cos\theta) \ge 0$
holds as well. The implementation only flips a root that lies clearly in the
lower half-plane, so that rounding noise of $10^{-17}$ in a propagating wave
does not turn it around (see the `snell_cos_theta` docstring).

The angle itself (`snell_theta`) is complex in general. `arccos` is fine only
for real indices below the critical angle; in general both $\sin\theta$ (from
the invariant) and $\cos\theta$ (from the decaying root) are fixed first, and
$\theta = -i\ln(\cos\theta + i\sin\theta)$, which inverts
$e^{i\theta} = \cos\theta + i\sin\theta$.

## 5. Boundary conditions and the admittance

Without surface currents, the field components **parallel** to an interface
($E_x, E_y, H_x, H_y$) are continuous across it. So the TMM tracks one
tangential E component and one tangential H component in each polarization.
The tangential H is taken with the sign that makes the power flow normal to the
layers $S_z = \tfrac12\mathrm{Re}(E_\mathrm{tan} H_\mathrm{tan}^*)$:
$(E_y, -H_x)$ for s and $(E_x, H_y)$ for p.

For one plane wave travelling forward, $H_\mathrm{tan} = \eta\, E_\mathrm{tan}$
with the **tilted optical admittance** (`admittance`)

$$ \eta_s = \frac{n\cos\theta}{Z_0}, \qquad \eta_p = \frac{n}{Z_0\cos\theta}, $$

where $Z_0 = \mu_0 c \approx 376.73\ \Omega$. Why: in a plane wave
$|\mathbf H| = n|\mathbf E|/Z_0$. For s, E is entirely tangential but only the
part $H\cos\theta$ of H is; for p, H is entirely tangential but only
$E\cos\theta$ of E is. A backward wave has $H_\mathrm{tan} = -\eta E_\mathrm{tan}$.
At normal incidence both reduce to $n/Z_0$.

## 6. One interface: the Fresnel coefficients

Let a wave from medium 1 meet medium 2, with tangential amplitudes $E_i$
(incident), $E_r$ (reflected) and $E_t$ (transmitted). Continuity of tangential
E and H gives

$$ E_i + E_r = E_t, \qquad \eta_1 (E_i - E_r) = \eta_2 E_t, $$

so the ratios of the **tangential** fields are

$$ r_\mathrm{tan} = \frac{\eta_1 - \eta_2}{\eta_1 + \eta_2}, \qquad
   t_\mathrm{tan} = \frac{2\eta_1}{\eta_1 + \eta_2}. $$

For s, E is tangential, so these are the textbook
$r_s = \dfrac{n_1\cos\theta_1 - n_2\cos\theta_2}{n_1\cos\theta_1 + n_2\cos\theta_2}$ and
$t_s = \dfrac{2 n_1\cos\theta_1}{n_1\cos\theta_1 + n_2\cos\theta_2}$.

For p the textbook coefficients are ratios of the *full* E amplitudes, measured
against reference directions that must be stated (section 9). With the Hecht /
Born & Wolf choice,

$$ r_p = -r_\mathrm{tan} = \frac{n_2\cos\theta_1 - n_1\cos\theta_2}{n_2\cos\theta_1 + n_1\cos\theta_2}, \qquad
   t_p = t_\mathrm{tan}\,\frac{\cos\theta_1}{\cos\theta_2} = \frac{2 n_1\cos\theta_1}{n_2\cos\theta_1 + n_1\cos\theta_2}. $$

The factor $\cos\theta_1/\cos\theta_2$ converts tangential parts
$E\cos\theta$ back to full amplitudes, and the minus sign in $r_p$ comes from
the reference directions (section 9). `fresnel_r`, `fresnel_t` compute exactly
this, from the same admittances as the solver.

Two special angles follow (`brewster_angle`, `critical_angle`):

- **Brewster angle** $\theta_B = \arctan(n_2/n_1)$: $r_p = 0$, and $r_p$
  changes sign across it (air to glass: 56.31°).
- **Critical angle** $\theta_c = \arcsin(n_2/n_1)$ for $n_1 > n_2$: total
  internal reflection beyond it (glass to air: 41.81°).

## 7. The characteristic matrix of a layer

Inside layer $j$ (local coordinate $0 \le z \le d$) the field is a forward and a
backward wave:

$$ E(z) = E^+ e^{ik_z z} + E^- e^{-ik_z z}, \qquad
   H(z) = \eta\,(E^+ e^{ik_z z} - E^- e^{-ik_z z}). $$

Write the fields at the front face (a, $z=0$) in terms of those at the back
face (b, $z=d$). With the **phase thickness** (`propagation_phase`)

$$ \delta = k_z d = \frac{\omega}{c}\, n\, d \cos\theta, $$

eliminating $E^\pm$ gives

$$ \begin{pmatrix} E_a \\ H_a \end{pmatrix} =
   \underbrace{\begin{pmatrix} \cos\delta & -\dfrac{i}{\eta}\sin\delta \\ -i\eta\sin\delta & \cos\delta \end{pmatrix}}_{M_j}
   \begin{pmatrix} E_b \\ H_b \end{pmatrix}. $$

Check: a purely forward wave has $H_b = \eta E_b$ and then
$E_a = (\cos\delta - i\sin\delta)E_b = e^{-i\delta}E_b$, i.e. it gained
$e^{+i\delta}$ crossing the layer, as it must for $e^{-i\omega t}$.
$\mathrm{Re}\,\delta$ sets interference ($\delta = \pi/2$ is a quarter-wave layer);
$e^{-\mathrm{Im}\,\delta}$ is the amplitude attenuation of one crossing.

## 8. The stack: product, r and t

Because tangential E and H are continuous at every interface, the back face of
one layer is the front face of the next, and the matrices simply multiply:

$$ M = M_1 M_2 \cdots M_{N-2}. $$

In the exit medium only a transmitted wave exists, so there
$(E, H) = (1, \eta_\mathrm{exit})$ times its tangential amplitude. Carry that
to the front of the stack:

$$ \begin{pmatrix} B \\ C \end{pmatrix} = M \begin{pmatrix} 1 \\ \eta_\mathrm{exit} \end{pmatrix}. $$

In the incident medium these fields are incident plus reflected wave,
$B = E_i + E_r$ and $C = \eta_0(E_i - E_r)$, so

$$ r_\mathrm{tan} = \frac{\eta_0 B - C}{\eta_0 B + C}, \qquad
   t_\mathrm{tan} = \frac{2\eta_0}{\eta_0 B + C}, $$

which are converted to the textbook $r$, $t$ exactly as for one interface:
nothing changes for s; for p, $r = -r_\mathrm{tan}$ and
$t = t_\mathrm{tan}\cos\theta_0/\cos\theta_\mathrm{exit}$. With no interior
layers, $M$ is the identity and this is the Fresnel result of section 6.

## 9. Power: R, T, A

The time-averaged power flow normal to the layers of a single wave is
$S_z = \tfrac12\mathrm{Re}(\eta)\,|E_\mathrm{tan}|^2$. Hence

$$ R = |r|^2, \qquad
   T = \frac{\mathrm{Re}(\eta_\mathrm{exit})}{\mathrm{Re}(\eta_0)}\,|t_\mathrm{tan}|^2, \qquad
   A = 1 - R - T. $$

$T$ is not $|t|^2$: the transmitted wave travels in another medium and at
another angle, which changes how much power the same field amplitude carries.
Beyond the critical angle $\eta_\mathrm{exit}$ is purely imaginary and $T = 0$.
$R$, $T$, $A$ do not depend on any sign or time convention. The incident medium
should be lossless; otherwise incident and reflected waves interfere in the
power flow and $R$, $T$ lose their simple meaning.

## 10. Conventions, and how to convert

### The sign of $r_p$

Textbooks disagree on the sign of $r_p$ because it depends on which direction
counts as "positive" for the reflected electric field, which is a choice. cjctmm
uses the choice of Hecht, Born & Wolf and Wikipedia:

> the magnetic fields of the incident, reflected and transmitted waves all
> point along the **same** direction $\hat{\mathbf y}$ (normal to the plane of
> incidence), and each electric field follows from $\mathbf E = Z\,\mathbf H \times \hat{\mathbf k}$,
> with $Z = Z_0/n$.

For the incident wave $\hat{\mathbf k}_i = (\sin\theta_1, 0, \cos\theta_1)$ this
gives $\mathbf E_i \parallel (\cos\theta_1, 0, -\sin\theta_1)$; for the
reflected wave $\hat{\mathbf k}_r = (\sin\theta_1, 0, -\cos\theta_1)$ gives
$\mathbf E_r \parallel (-\cos\theta_1, 0, -\sin\theta_1)$. Their tangential
($x$) parts point in **opposite** directions, which is the minus sign in
$r_p = -r_\mathrm{tan}$.

Consequences that look odd at first:

- At normal incidence, air to glass: $r_s = -0.2$ but $r_p = +0.2$. Both
  describe the same reflected wave (at normal incidence s and p are
  identical); the p reference direction for the reflected field is simply the
  opposite of the incident one, so the same physical field gets the opposite
  sign.
- At grazing incidence both go to $-1$.

The test `test_fresnel_matches_boundary_conditions` pins this from first
principles: it builds the three plane waves with explicit $\mathbf k$ vectors
and the field directions above, solves the continuity of tangential E and H
numerically, and compares with `fresnel_r`, `fresnel_t` at several angles,
also for absorbing media.

**The other convention**, in which the reflected p reference direction is
flipped so that $r_p = r_s$ at normal incidence, is obtained with `-result.r` for p
polarization (and `-fresnel_r(..., "p")`). $t_p$ is the same in both.

### Time convention

See section 3: inputs always as $n' + i\kappa$; `conjugate_model=True` returns
$r^*$, $t^*$, the coefficients in the $e^{+i\omega t}$ convention.

## 11. Numerical notes

- **Opaque layers.** $\cos\delta$ and $\sin\delta$ grow like $e^{|\mathrm{Im}\,\delta|}$.
  A thick metal layer would overflow float64 (near $e^{709}$), also after the
  matrix products. cjctmm clamps $|\mathrm{Im}\,\delta| \le 40$
  (`MAX_IMAG_PHASE`): a single pass is then attenuated by $e^{-40} \approx 4\times10^{-18}$,
  so the layer is opaque either way and the result does not change measurably.
- **Why characteristic matrices.** The alternative "interface matrix times
  propagation matrix" formulation contains $e^{-i\delta}$ for the backward wave
  and overflows sooner in lossy layers; the characteristic matrix only contains
  $\cos\delta$ and $\sin\delta$ and the admittances.
- **Many layers.** All layer matrices are built at once and multiplied
  pairwise (neighbours first, keeping their order), so a stack of thousands of
  thin sublayers (e.g. a discretized depth profile) costs a few dozen array
  operations instead of one Python loop iteration per layer.

## 12. Finite substrate passes: time-gated echoes

A pulse that enters a substrate of thickness $d$ is partly reflected back and
forth between its faces, producing a train of echoes delayed by multiples of
the round-trip time

$$ \Delta t = \frac{2\,\mathrm{Re}(n)\, d\cos\theta}{c} = \frac{2\,\mathrm{Re}\,\delta}{\omega}. $$

A time-domain measurement often cuts the trace with a time window before the
first echo arrives. The fully coherent TMM ($N\to\infty$, the default) includes
all echoes and so describes an untruncated measurement. With
`num_substrate_passes = N`, the substrate (the second-to-last layer) contributes
only the first $N$ passes in transmission and the matching echoes in reflection.
With 1 = the incident medium together with all front layers (one effective
interface, computed coherently with the characteristic matrix), 2 = the
substrate and 3 = the exit medium, and the substrate's one-pass phase factor
$e^{i\delta}$:

$$ t_N = t_{12}\,t_{23}\,e^{i\delta} \sum_{k=0}^{N-1} \left(r_{21} r_{23} e^{2i\delta}\right)^k, \qquad
   r_N = r_{12} + t_{12}\,r_{23}\,t_{21}\,e^{2i\delta} \sum_{k=0}^{N-2} \left(r_{21} r_{23} e^{2i\delta}\right)^k. $$

$N = 1$ is the direct pulse only; for an absorbing substrate $N \to \infty$
reproduces the coherent result.

### When is this valid?

The model truncates **only** the substrate's echoes; the layers in front of it
are still summed coherently, i.e. all their echoes are kept. A real gate at
$t_\mathrm{gate} \approx N\,\Delta t_\mathrm{sub}$ would also cut the front
layers' echoes that arrive later than $t_\mathrm{gate}$. For a thin film
($\Delta t_\mathrm{film} \ll t_\mathrm{gate}$) those are negligible; for a front
layer about as thick as the substrate they are not.

`solve_tmm` estimates that error. For each front layer $j$ an echo is
multiplied per round trip by

$$ \rho_j = r_\mathrm{front}\, r_\mathrm{back}\, e^{2i\delta_j}, $$

with the bare-interface coefficients inside the layer towards its two
neighbours. The first echo beyond the gate is number
$k_\mathrm{min} = \lceil t_\mathrm{gate}/\Delta t_j \rceil$, and all later ones
together have a relative amplitude of about

$$ \epsilon_j \approx \frac{|\rho_j|^{k_\mathrm{min}}}{1 - |\rho_j|}. $$

The largest $\epsilon_j$ over frequencies and layers is compared with
`substrate_pass_tolerance` (default $10^{-3}$); above it a `UserWarning` names
the layer and the estimate. This is an estimate: it uses the phase index
rather than the group index (dispersion shifts echo times) and ignores echoes
that bounce in more than one layer.

### Not covered: incoherent thick substrates

In visible and infrared optics a thick substrate (much thicker than the
coherence length of the light, or not flat to within a wavelength) is usually
treated **incoherently**: the multiple reflections are summed in intensity,
e.g. $T = T_{12} T_{23} / (1 - R_{21} R_{23})$, discarding their phases. That is
a different model from truncating a coherent echo series in time, and cjctmm
does not implement it.

## References

- M. Born and E. Wolf, *Principles of Optics*, 7th ed., Cambridge University
  Press (1999), ch. 1.5 (Fresnel formulae) and 1.6 (wave propagation in a
  stratified medium, characteristic matrix).
- H. A. Macleod, *Thin-Film Optical Filters*, 5th ed., CRC Press (2018), ch. 2
  (admittance formulation, characteristic matrix); note it uses the
  $e^{+i\omega t}$ convention.
- E. Hecht, *Optics*, 5th ed., Pearson (2017), ch. 4.6 (Fresnel equations,
  sign convention) and 9.7 (thin films, multilayers).
- M. Tinkham, "Energy gap interpretation of experiments on infrared
  transmission through superconducting films", Phys. Rev. **104**, 845 (1956):
  the thin-film transmission formula used in the tests and the examples.
